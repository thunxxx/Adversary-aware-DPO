# Copyright 2024 HuggingFace Inc. and the LlamaFactory team.
#
# This code is inspired by the HuggingFace's transformers library.
# https://github.com/huggingface/transformers/blob/v4.40.0/src/transformers/trainer_seq2seq.py
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import os
from types import MethodType
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from transformers import Seq2SeqTrainer
from typing_extensions import override

from ...extras import logging
from ...extras.constants import IGNORE_INDEX
from ...extras.packages import is_transformers_version_greater_than
from ..callbacks import SaveProcessorCallback
from ..trainer_utils import create_custom_optimizer, create_custom_scheduler
from .utilis import GDAdversary, AdversaryWrapper, add_hooks
import math
import torch.nn.functional as F
# import pdb
# pdb.set_trace()

if TYPE_CHECKING:
    from torch.utils.data import Dataset
    from transformers import PreTrainedModel, PreTrainedTokenizer, ProcessorMixin
    from transformers.trainer import PredictionOutput

    from ...hparams import FinetuningArguments



logger = logging.get_logger(__name__)

def zero_nan_grads(model):
    flag = False
    for name, p in model.named_parameters():
        if p.grad is not None:
            if torch.isnan(p.grad).any():
                flag = True
                p.grad[torch.isnan(p.grad)] = 0.
    if flag:
        pass


def clear_hooks(model):
    for name, module in model.named_children():
        if isinstance(module, AdversaryWrapper):
            setattr(model, name, module.module)
            clear_hooks(module.module)
        else:
            clear_hooks(module)


def PGD(model, rejected_features, adversaries):
    params = []
    for adv in adversaries:
        params += list(adv.parameters())
    
    # Define optimization utils
    adv_optim = torch.optim.AdamW(params, lr=5e-2)

    for i in range(10):
        adv_optim.zero_grad()
        loss = model(**rejected_features).loss
        loss.backward()

        zero_nan_grads(adv)

        torch.nn.utils.clip_grad_norm_(
            adv.parameters(), 1.0)
            
        adv_optim.step()
        for adv in adversaries:
            adv.clip_attack()        

def pad_to_max_length(tensor, max_seq_len, pad_value=0):
    current_len = tensor.size(1)
    if current_len < max_seq_len:
        pad_len = max_seq_len - current_len
        # 对于形状为 (batch_size, seq_len) 的 tensor，
        # pad 参数 (0, pad_len) 表示在最后一维右侧填充 pad_len 个 pad_value
        tensor = F.pad(tensor, (0, pad_len), value=pad_value)
    return tensor


class CustomSeq2SeqTrainer(Seq2SeqTrainer):
    r"""
    Inherits Seq2SeqTrainer to compute generative metrics such as BLEU and ROUGE.
    """

    def __init__(
        self, finetuning_args: "FinetuningArguments", processor: Optional["ProcessorMixin"], **kwargs
    ) -> None:
        if is_transformers_version_greater_than("4.46"):
            kwargs["processing_class"] = kwargs.pop("tokenizer")
        else:
            self.processing_class: "PreTrainedTokenizer" = kwargs.get("tokenizer")

        super().__init__(**kwargs)
        self.finetuning_args = finetuning_args

        if processor is not None:
            self.add_callback(SaveProcessorCallback(processor))

        if finetuning_args.use_badam:
            from badam import BAdamCallback, clip_grad_norm_old_version  # type: ignore

            self.accelerator.clip_grad_norm_ = MethodType(clip_grad_norm_old_version, self.accelerator)
            self.add_callback(BAdamCallback)

        self.AT_alpha = self.finetuning_args.AT_alpha



    @override
    def create_optimizer(self) -> "torch.optim.Optimizer":
        if self.optimizer is None:
            self.optimizer = create_custom_optimizer(self.model, self.args, self.finetuning_args)
        return super().create_optimizer()



    @override
    def create_scheduler(
        self, num_training_steps: int, optimizer: Optional["torch.optim.Optimizer"] = None
    ) -> "torch.optim.lr_scheduler.LRScheduler":
        create_custom_scheduler(self.args, num_training_steps, optimizer)
        return super().create_scheduler(num_training_steps, optimizer)



    @override
    def _get_train_sampler(self) -> Optional["torch.utils.data.Sampler"]:
        if self.finetuning_args.disable_shuffling:
            return torch.utils.data.SequentialSampler(self.train_dataset)

        return super()._get_train_sampler()

    

    @override
    def compute_loss(
        self, model: "PreTrainedModel", inputs, return_outputs: bool = False, **kwargs
    ) -> Union["torch.Tensor", Tuple["torch.Tensor", List["torch.Tensor"]]]:
        r"""
        Fixes the loss value. See https://github.com/huggingface/transformers/pull/35438 for details.

        It should be removed after https://github.com/huggingface/transformers/pull/35651 is merged.
        """

        clear_hooks(model)

        utility_features = inputs['utility']
        chosen_features = inputs['chosen']
        rejected_features = inputs['rejected']
        win_loss, rej_loss, adv_loss, utility_loss = (
            torch.tensor(0, device = model.device),
            torch.tensor(0, device = model.device),
            torch.tensor(0, device = model.device),
            torch.tensor(0, device = model.device)
        )
        
        if utility_features is not None:
            utility_loss = super().compute_loss(model, utility_features, return_outputs, **kwargs)
        
        if chosen_features is not None:

            self.disable_model_gradients()
            pgd_layers=[8, 16, 24, 30]
            max_seq_len = max(chosen_features['input_ids'].shape[1], rejected_features['input_ids'].shape[1])

            chosen_features['input_ids'] = pad_to_max_length(chosen_features['input_ids'], max_seq_len, pad_value=32001)
            rejected_features['input_ids'] = pad_to_max_length(rejected_features['input_ids'], max_seq_len, pad_value=32001)
            chosen_features['labels'] = pad_to_max_length(chosen_features['labels'], max_seq_len, pad_value=-100)
            rejected_features['labels'] = pad_to_max_length(rejected_features['labels'], max_seq_len, pad_value=-100)
            chosen_features['attention_mask'] = pad_to_max_length(chosen_features['attention_mask'], max_seq_len, pad_value=0)
            rejected_features['attention_mask'] = pad_to_max_length(rejected_features['attention_mask'], max_seq_len, pad_value=0)


            input_mask = (rejected_features['input_ids']==rejected_features['labels']).cumsum(dim=0) == 0

            adversaries, hooks = add_hooks(model, pgd_layers, input_mask)
            PGD(model, rejected_features, adversaries)
            self.enable_model_gradients(model)

            adv_loss = super().compute_loss(model, chosen_features, return_outputs, **kwargs)
            if adv_loss < 0.5:
                adv_loss = 0.5 + 0.0001 * adv_loss

            
        loss = adv_loss + self.AT_alpha * utility_loss

        # if kwargs.get("num_items_in_batch") and not getattr(self, "model_accepts_loss_kwargs", False):
        #     if return_outputs:
        #         loss = (loss[0] / self.args.gradient_accumulation_steps, *loss[1:])
        #     else:
        #         loss = loss / self.args.gradient_accumulation_steps

        return loss


    def disable_model_gradients(self):
        for param in self.model.parameters():
            param.requires_grad_(False)
    
    def enable_model_gradients(self, model):
        n_layers = 32
        for i in range(n_layers):
            for name, param in model.get_submodule('module.language_model.base_model.layers')[i].named_parameters():
                if "lora_" in name:
                    param.requires_grad_(True)


    @override
    def prediction_step(
        self,
        model: "torch.nn.Module",
        inputs: Dict[str, Union["torch.Tensor", Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
        **gen_kwargs,
    ) -> Tuple[Optional[float], Optional["torch.Tensor"], Optional["torch.Tensor"]]:
        r"""
        Removes the prompt part in the generated tokens.

        Subclass and override to inject custom behavior.
        """
        if self.args.predict_with_generate:  # do not pass labels to model when generate
            labels = inputs.pop("labels", None)
        else:
            labels = inputs.get("labels")

        loss, generated_tokens, _ = super().prediction_step(
            model, inputs, prediction_loss_only=prediction_loss_only, ignore_keys=ignore_keys, **gen_kwargs
        )
        if generated_tokens is not None and self.args.predict_with_generate:
            generated_tokens[:, : inputs["input_ids"].size(-1)] = self.processing_class.pad_token_id
            generated_tokens = generated_tokens.contiguous()

        return loss, generated_tokens, labels

    def save_predictions(
        self, dataset: "Dataset", predict_results: "PredictionOutput", skip_special_tokens: bool = True
    ) -> None:
        r"""
        Saves model predictions to `output_dir`.

        A custom behavior that not contained in Seq2SeqTrainer.
        """
        if not self.is_world_process_zero():
            return

        output_prediction_file = os.path.join(self.args.output_dir, "generated_predictions.jsonl")
        logger.info_rank0(f"Saving prediction results to {output_prediction_file}")

        labels = np.where(
            predict_results.label_ids != IGNORE_INDEX, predict_results.label_ids, self.processing_class.pad_token_id
        )
        preds = np.where(
            predict_results.predictions != IGNORE_INDEX,
            predict_results.predictions,
            self.processing_class.pad_token_id,
        )

        for i in range(len(preds)):
            pad_len = np.nonzero(preds[i] != self.processing_class.pad_token_id)[0]
            if len(pad_len):  # move pad token to last
                preds[i] = np.concatenate((preds[i][pad_len[0] :], preds[i][: pad_len[0]]), axis=-1)

        decoded_inputs = self.processing_class.batch_decode(dataset["input_ids"], skip_special_tokens=False)
        decoded_preds = self.processing_class.batch_decode(preds, skip_special_tokens=skip_special_tokens)
        decoded_labels = self.processing_class.batch_decode(labels, skip_special_tokens=skip_special_tokens)

        with open(output_prediction_file, "w", encoding="utf-8") as f:
            for text, pred, label in zip(decoded_inputs, decoded_preds, decoded_labels):
                f.write(json.dumps({"prompt": text, "predict": pred, "label": label}, ensure_ascii=False) + "\n")