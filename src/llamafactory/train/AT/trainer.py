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

if TYPE_CHECKING:
    from torch.utils.data import Dataset
    from transformers import PreTrainedModel, PreTrainedTokenizer, ProcessorMixin
    from transformers.trainer import PredictionOutput

    from ...hparams import FinetuningArguments


logger = logging.get_logger(__name__)

def normalize(images):
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).to(images.device)
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).to(images.device)
    images = images - mean[None, :, None, None]
    images = images / std[None, :, None, None]
    return images



def denormalize(images):
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).to(images.device)
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).to(images.device)
    images = images * std[None, :, None, None]
    images = images + mean[None, :, None, None]
    return images


def PGD(model, input_ids, attention_mask, images, labels, image_sizes = None, ep = 0.2, alpha = 0.0125, num_steps = 30):

    
    x_adv = denormalize(images)
    original_images = x_adv.clone()

    x_adv.requires_grad_(True)
    min_loss = float('inf')
    best_adv_image = None

    for step in range(num_steps):
        
        if image_sizes is None:
            loss = model(input_ids=input_ids, labels=labels, pixel_values=normalize(x_adv),attention_mask=attention_mask).loss
        else:
            loss = model(input_ids=input_ids, labels=labels, pixel_values=normalize(x_adv), image_sizes = image_sizes, attention_mask=attention_mask).loss            
        if loss.item() < min_loss:
            best_adv_image = x_adv.clone().detach()
            min_loss = loss.item()

        grad = torch.autograd.grad(
            outputs = loss,
            inputs = x_adv,
            retain_graph = False,
            create_graph = False,
            only_inputs = True,
            allow_unused=True                
        )[0]

        x_adv = x_adv - alpha * grad.sign()

        x_adv = torch.clamp(original_images-x_adv, -ep, ep)
        x_adv = torch.clamp(x_adv, 0, 1)

        # print(f"Step {step+1}/{num_steps} - Loss: {loss.item()}, Min Loss: {min_loss}")

    best_adv_image = normalize(best_adv_image)    


    return best_adv_image


class CustomSeq2SeqTrainer(Seq2SeqTrainer):
    r"""
    Inherits Seq2SeqTrainer to compute generative metrics such as BLEU and ROUGE.
    """

    def __init__(
        self, finetuning_args: "FinetuningArguments", processor: Optional["ProcessorMixin"], model_name: str , **kwargs
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
        self.model_name = model_name
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
            rejected_input_ids, rejected_attention_mask, rejected_pixel_values, rejected_labels = rejected_features['input_ids'], rejected_features['attention_mask'], rejected_features['pixel_values'], rejected_features['labels']
            if self.model_name == 'llava_next':
                rejected_image_sizes = rejected_features['image_sizes']
                adv_images = PGD(model, rejected_input_ids, rejected_attention_mask, rejected_pixel_values, rejected_labels, rejected_image_sizes)
            else:
                adv_images = PGD(model, rejected_input_ids, rejected_attention_mask, rejected_pixel_values, rejected_labels)
            chosen_features['pixel_values'] = adv_images
            rejected_features['pixel_values'] = adv_images
            win_loss = super().compute_loss(model, chosen_features, return_outputs, **kwargs)

            if win_loss < 0.5:
                win_loss = 0.5 + 0.0001 * win_loss



            adv_loss = win_loss

        # self.log({"win_loss": win_loss,
        #           "rej_loss": rej_loss,
        #           "utility_loss": utility_loss})

        loss = adv_loss + self.AT_alpha * utility_loss



        return loss

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