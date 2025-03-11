# Adversary-aware DPO
This is the official repository of paper "Adversary-Aware DPO: Enhancing Safety Alignment in Vision Language Models via Adversarial Training".

## Getting Started
### Installation
```
conda create -n ADPO python=3.11.11
conda activate ADPO
pip install -e ".[torch,metrics]"
pip install deepspeed==0.15.4
```
### Usage
#### Using AT to train a reference model
```
bash scripts/at.sh
```
#### DPO training with adversary-aware loss
```
bash scripts/adpo.sh
```
The arguments:
- model_name_or_path: Path to the model weight or identifier from huggingface.co/models
- output_dir: output path of the fine-tuned model
## Acknowlegement and citation
We thank the following open-source repositories.
```
[1] https://github.com/hiyouga/LLaMA-Factory
```
If you find ADPO useful in your research, please consider citing our paper:
```
@article{weng2025adversary,
  title={Adversary-Aware DPO: Enhancing Safety Alignment in Vision Language Models via Adversarial Training},
  author={Weng, Fenghua and Lou, Jian and Feng, Jun and Huang, Minlie and Wang, Wenjie},
  journal={arXiv preprint arXiv:2502.11455},
  year={2025}
}
```