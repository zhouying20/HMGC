import os
import json
from glob import glob
from copy import deepcopy
from pathlib import Path


def main():
    origin_file = "./data/CheckGPT/test_rnd10k.jsonl"
    base_dir = "./output/checkgpt"

    tests = list()
    with open(origin_file, "r") as rf:
        for line in rf:
            lj = json.loads(line)
            tests.append(lj)

    temp_dir = os.path.join(base_dir, "temp")
    for cls in ["baseline", "ablation", "main"]:
        ori_pattern = os.path.join(temp_dir, cls, "trans_*")
        save_dir = os.path.join(base_dir, cls)
        for file in glob(ori_pattern):
            save_file = os.path.join(save_dir, Path(file).stem.lstrip("trans_") + ".jsonl")

            trans = dict()
            with open(file, "r") as rf:
                for line in rf:
                    one = json.loads(line)
                    trans[int(one["sample_id"])] = one

            missing_attack = 0
            with open(save_file, "w") as wf:
                for idx in range(len(tests)):
                    to_save = deepcopy(tests[idx])
                    if "question" in to_save.keys():
                        to_save["prefix"] = to_save.pop("question")
                    if "answer" in to_save.keys():
                        to_save["origin_text"] = to_save.pop("answer")
                    elif "text" in to_save.keys():
                        to_save["origin_text"] = to_save.pop("text")
                    assert "origin_text" in to_save.keys()

                    if idx in trans.keys():
                        one = trans[idx]
                        assert to_save["label"] == "gpt"
                        to_save["attacked_text"] = one["x"]
                    elif to_save["label"] == "gpt":
                        to_save["attacked_text"] = None
                        missing_attack += 1

                    wf.write(json.dumps(to_save, ensure_ascii=False) + "\n")
            print(f"{file} done, missing {missing_attack} attacks")


if __name__ == "__main__":
    main()
