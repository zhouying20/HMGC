import json
import random
import argparse
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from tqdm import tqdm
from glob import glob
from pathlib import Path
from datasets import load_dataset
from transformers import (
    RobertaConfig,
    RobertaTokenizer,
    RobertaForSequenceClassification,
)

from evaluation.models.checkgpt import CheckGPTDetectorForPipeline
from evaluation.metrics import calc_classification_metrics


def set_seed(args):
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(seed)


def set_distribution(args):
    if args.local_rank == -1:  # single-node multi-gpu -> DP (or cpu)
        args.n_gpu = torch.cuda.device_count()
        args.local_world_size = 1
        args.device = str(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        args.batch_size = args.batch_size_per_device * args.n_gpu
        args.batch_size = max(args.batch_size, args.batch_size_per_device) # in case of cpu-mode
    else:  # distributed mode -> DDP
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1
        args.local_world_size = dist.get_world_size()
        args.device = str(torch.device("cuda", args.local_rank))
        raise NotImplementedError("DDP is not implemented yet...")


def get_distributed_model(
    model: nn.Module,
    device: object,
    optimizer: torch.optim.Optimizer = None,
    n_gpu: int = 1,
    local_rank: int = -1,
    fp16: bool = False,
    fp16_opt_level: str = "O1",
) -> (nn.Module, torch.optim.Optimizer):
    model.to(device)

    if fp16:
        try:
            import apex
            from apex import amp

            apex.amp.register_half_function(torch, "einsum")
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")

        model, optimizer = amp.initialize(model, optimizer, opt_level=fp16_opt_level)

    if local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[device if device else local_rank],
            output_device=local_rank,
            find_unused_parameters=True,
        )
    elif n_gpu > 1:
        model = torch.nn.DataParallel(model)

    return model, optimizer


class CheckGPTBlackBoxEvaluator():
    def __init__(self, config):
        self.config = config
        self.checkgpt_path="./data/CheckGPT/model/Unified_Task123.pth"

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = RobertaTokenizer.from_pretrained("roberta-large")
        config = RobertaConfig.from_pretrained("roberta-large", num_labels=2)
        model = CheckGPTDetectorForPipeline.from_pretrained("roberta-large", config=config, device=self.device)
        model.classifier.load_state_dict(torch.load(self.checkgpt_path), strict=True)
        model = model.to(self.device)
        model.eval()
        self.model = model

        self.lb_mapping = {
            "gpt": 0,
            "human": 1,
        }
        self.labels = ["gpt", "human"]

    def predict_one(self, input):
        item = input.replace("\n", " ").replace("  ", " ").strip()
        tokens = self.tokenizer.encode(item)
        if len(tokens) > 512:
            tokens = tokens[:512]
            print("!!!Input too long. Truncated to first 512 tokens.")
        inputs = torch.tensor(tokens).unsqueeze(0).to(self.device)
        outputs = self.model(inputs)
        pred = torch.max(outputs.data, 1)[1]
        gpt_prob, hum_prob = F.softmax(outputs.data, dim=1)[0]
        return pred[0].data, 100 * gpt_prob, 100 * hum_prob

    def load_test_samples(self, test_file):
        ori_key = self.config.origin_key
        att_key = self.config.attacked_key
        label_key = self.config.label_key

        samples = list()
        labels = list()
        with open(test_file, "r") as rf:
            for line in tqdm(rf, dynamic_ncols=True):
                l_j = json.loads(line)
                labels.append(self.lb_mapping[l_j[label_key]])
                if l_j[label_key] == "gpt":
                    samples.append(l_j[att_key] or l_j[ori_key])
                else:
                    samples.append(l_j[ori_key])
        return samples, labels

    def run_eval(self, test_file):
        samples, labels = self.load_test_samples(test_file)

        preds = list()
        logits = list()
        for _text in samples:
            pred, gpt_prob, hum_prob = self.predict_one(_text)
            gpt_prob = gpt_prob.cpu().item()
            hum_prob = hum_prob.cpu().item()
            logits.append((gpt_prob, hum_prob))
            if np.isclose(gpt_prob, hum_prob) or gpt_prob > hum_prob:
                preds.append(self.lb_mapping["gpt"])
            else:
                preds.append(self.lb_mapping["human"])

        preds = np.array(preds)
        labels = np.array(labels)
        metric_report = calc_classification_metrics(preds, labels, target_names=self.labels)
        print("*************************")
        print(f"[{test_file}]")
        print(json.dumps(metric_report, indent=4, ensure_ascii=False))
        print("*************************")
        print("\n")

        return metric_report


class HC3WhiteBoxEvaluator(object):
    def __init__(self, config) -> None:
        self.config = config
        self.model_path = "Hello-SimpleAI/chatgpt-detector-roberta"

        self.local_rank = config.local_rank if config.local_rank != -1 else 0
        self.world_size = config.local_world_size
        self.batch_size = config.batch_size

        self.label2id = {
            "human": 0,
            "gpt": 1,
        }
        self.labels = ["human", "gpt"]

        model_config = RobertaConfig.from_pretrained(self.model_path, num_labels=2)
        tokenizer = RobertaTokenizer.from_pretrained(self.model_path, use_fast=False)
        model = RobertaForSequenceClassification.from_pretrained(self.model_path, config=model_config)
        model, _ = get_distributed_model(
            model, config.device,
            n_gpu=config.n_gpu,
            local_rank=config.local_rank,
        )
        self.model = model
        self.tokenizer = tokenizer

    def prepare_data(
        self,
        data_file,
        padding="max_length",
        max_seq_length=512,
        batch_size=64,
        shuffle=False,
        pin_memory=False,
        num_workers=0
    ) -> DataLoader:
        ori_key = self.config.origin_key
        att_key = self.config.attacked_key
        prefix_key = self.config.prefix_key
        label_key = self.config.label_key

        def tokenize_func(examples):
            # Tokenize the texts
            token_args = list()
            for idx, lb in enumerate(examples[label_key]):
                if lb == "gpt": # only gpt samples were attacked
                    _text = examples[att_key][idx] or examples[ori_key][idx]
                    token_args.append(
                        _text if prefix_key is None else (examples[prefix_key][idx], _text)
                    )
                else:
                    token_args.append(
                        examples[ori_key][idx] if prefix_key is None else (examples[prefix_key][idx], examples[ori_key][idx])
                    )
            result = self.tokenizer(token_args, padding=padding, max_length=max_seq_length, truncation=True)

            # Map labels to IDs (not necessary for GLUE tasks)
            if self.label2id is not None and label_key in examples:
                result["label"] = [(self.label2id[l] if l != -1 else -1) for l in examples[label_key]]
            return result

        def label_split_collate_fn(batch):
            input_ids = torch.stack([d['input_ids'] for d in batch], dim=0).to(self.config.device)
            attn_masks = torch.stack([d['attention_mask'] for d in batch], dim=0).to(self.config.device)

            labels = [d['label'] for d in batch]

            return {
                'input_ids': input_ids,
                'attention_mask': attn_masks
            }, labels

        _data = {"data": data_file}
        dataset = load_dataset("json", data_files=_data)["data"]
        dataset = dataset.map(
            tokenize_func,
            batched=True,
            desc="Running tokenizer on dataset",
        )
        dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

        if self.world_size > 1:
            sampler = DistributedSampler(
                dataset,
                shuffle=shuffle,
                rank=self.local_rank,
                num_replicas=self.world_size,
                drop_last=False
            )
        else:
            sampler = None

        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            pin_memory=pin_memory,
            num_workers=num_workers,
            drop_last=False,
            sampler=sampler,
            collate_fn=label_split_collate_fn,
        )

    def do_eval(self, test_file):
        model = self.model
        dataloader = self.prepare_data(test_file, batch_size=self.batch_size)

        all_labels = list()
        all_preds = list()
        for batch_inputs, batch_labels in tqdm(dataloader):
            all_labels += batch_labels

            with torch.no_grad():
                logits = model(**batch_inputs).logits
                logits = logits.cpu().numpy()

            preds = np.argmax(logits, axis=1).tolist()
            all_preds += preds

        metric_report = calc_classification_metrics(all_preds, all_labels, target_names=self.labels)
        print("*************************")
        print(f"[{self.model_path}]")
        print(f"[{test_file}]")
        print(json.dumps(metric_report, indent=4, ensure_ascii=False))
        print("*************************")
        print("\n")

        return metric_report


def main(args):
    if args.detector.lower() == "checkgpt":
        evaluator = CheckGPTBlackBoxEvaluator()
    elif args.detector.lower() == "hc3":
        evaluator = HC3WhiteBoxEvaluator(args)

    test_files = list()
    for test_pattern in args.tests:
        test_files += list(glob(test_pattern))

    res_evals = list()
    for test in sorted(test_files):
        eval_meta = {
            "test": Path(test).stem,
            "model": args.detector,
        }
        eval_metric = evaluator.do_eval(test)
        res_evals.append(dict(list(eval_meta.items()) + list(eval_metric.items())))
    res_df = pd.DataFrame.from_records(res_evals)

    if args.output_file is not None:
        res_df.to_csv(args.output_file, index=False)
    else:
        print(res_df.to_markdown())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A simple argument parser example")
    parser.add_argument("--local-rank", "--local_rank", type=int, default=-1,)
    parser.add_argument("--seed", type=int, default=42,)

    parser.add_argument("--detector", type=str, required=True,)
    parser.add_argument("--tests", nargs='+', default=[],)
    parser.add_argument("--output_file", type=str, default=None,)

    parser.add_argument("--batch_size_per_device", type=int, default=256,)
    parser.add_argument("--origin_key", type=str, default="origin_text",)
    parser.add_argument("--attacked_key", type=str, default="attacked_text",)
    parser.add_argument("--prefix_key", type=str, default=None,)
    parser.add_argument("--label_key", type=str, default="label",)
    args = parser.parse_args()

    set_distribution(args)
    set_seed(args)
    main(args)
