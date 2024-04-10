import os
import json
import argparse
from tqdm import tqdm
from textflint.adapter import auto_dataset
from attack.methods.transformations.dipper_paraphrase import Paraphrase


def flint_transform(test_file, out_dir):
    trans_method = Paraphrase()

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
    dataset = auto_dataset(data_input=data_list, task="UT")

    method_name = trans_method.__repr__()
    trans_file = os.path.join(out_dir, f"trans_{method_name}_{len(dataset)}.json")

    print('******Start {0}!******'.format(trans_method))
    with open(trans_file, "w") as wt:
        for sample in tqdm(dataset, dynamic_ncols=True):
            trans_rst = trans_method.transform(
                sample, n=1, field="x",
            )
            for trans_one in trans_rst:
                wt.write(json.dumps(trans_one.dump(), ensure_ascii=False) + "\n")
    print('******Finish {0}!******'.format(trans_method))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A simple argument parser example")
    parser.add_argument("--out_dir", default="./output",)
    parser.add_argument("--test_file", default="./data/CheckGPT/test_rnd10000_gpt.jsonl",)
    args = parser.parse_args()

    flint_transform(args.test_file, args.out_dir)

