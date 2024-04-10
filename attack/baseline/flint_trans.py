import json
import argparse
from textflint.engine import Engine
from textflint.adapter import auto_config


def flint_transform(config_file, test_file):
    engine = Engine()
    config = auto_config(config=config_file)

    data_list = list()
    with open(test_file, "r") as rf:
        for line in rf:
            r_j = json.loads(line.strip())
            if "text" in r_j:
                x_text = r_j["text"]
            elif "answer" in r_j:
                x_text = r_j["answer"]
            else:
                raise NotImplementedError(f"not supported x field in {r_j}")
            data_list.append({
                "x": x_text,
                "y": r_j["label"],
            })
    engine.run(data_list, config=config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A simple argument parser example")
    parser.add_argument("--config", help="TextFlint config file path", default="./attack/baseline/config/checkgpt_baseline.json")
    parser.add_argument("--test_file", help="TextFlint test file path", default="./data/CheckGPT/data/test.jsonl")
    args = parser.parse_args()

    flint_transform(args.config, args.test_file)
