# Humanizing Machine-Generated Content

This repository contains resources of our paper:
- [Humanizing Machine-Generated Content: Evading AI-Text Detection through Adversarial Attack](https://arxiv.org/abs/2404.01907)

---

## How to reporduce our result
1. Download and unzip dataset from [Google Drive](https://drive.google.com/file/d/15rdZfNmnVeqEFKSu1A01DIvhYL30vadi)

2. Run
```
python evaluation/eval_accuracy.py \
    --detector hc3 \
    --tests ./output/hc3/**/*.jsonl \
    --output_file /tmp/hc3_evaluation.csv
```


## Do attacks on your own data
1. Distill sample labels from your target victim detector, train a surrogate model with `train_detector.py`

2. Follow `attack.multi_flint_attack` to start multi-process attacking


## Citation
If you find our paper/resources useful, please cite:
```
@inproceedings{Zhou2024_COLING,
 author = {Ying Zhou and
           Ben He and
           Le Sun},
 title = {Humanizing Machine-Generated Content: Evading AI-Text Detection through Adversarial Attack},
 booktitle = {Proceedings of the 2024 Joint International Conference on Computational Linguistics, Language Resources and Evaluation.},
 year = {2024},
}
```
