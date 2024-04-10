import json
import argparse
import textstat
import numpy as np

from glob import glob


class TextStatEvaluator(object):
    def calc_metrics(self, samples):
        res_each_scores = {
            "flesch_reading_ease": list(),
            "flesch_kincaid_grade": list(),
            "gunning_fog": list(),
            "smog_index": list(),
            "automated_readability_index": list(),
            "coleman_liau_index": list(),
            "linsear_write_formula": list(),
            "dale_chall_readability_score": list(),
            "text_standard": list(),
            "spache_readability": list(),
            "difficult_words": list(),
        }
        for sample in samples:
            res_each_scores["flesch_reading_ease"].append(textstat.flesch_reading_ease(sample))
            res_each_scores["flesch_kincaid_grade"].append(textstat.flesch_kincaid_grade(sample))
            res_each_scores["gunning_fog"].append(textstat.gunning_fog(sample))
            res_each_scores["smog_index"].append(textstat.smog_index(sample))
            res_each_scores["automated_readability_index"].append(textstat.automated_readability_index(sample))
            res_each_scores["coleman_liau_index"].append(textstat.coleman_liau_index(sample))
            res_each_scores["linsear_write_formula"].append(textstat.linsear_write_formula(sample))
            res_each_scores["dale_chall_readability_score"].append(textstat.dale_chall_readability_score(sample))
            res_each_scores["text_standard"].append(textstat.text_standard(sample, float_output=True))
            res_each_scores["spache_readability"].append(textstat.spache_readability(sample))
            res_each_scores["difficult_words"].append(textstat.difficult_words(sample))

        return res_each_scores

    def do_eval(self, samples):
        perturbed_scores = self.calc_metrics(samples)

        metric_report = dict()
        for metric in perturbed_scores.keys():
            ps = np.array(perturbed_scores[metric])
            metric_report[metric] = np.mean(ps)

        return metric_report


def load_attacked_texts(test_file):
    res_texts = list()
    with open(test_file, "r") as rf:
        for line in rf:
            lj = json.loads(line)
            if lj["label"] == "gpt":
                attacked_text = lj["attacked_text"] or lj["origin_text"]
                res_texts.append(attacked_text)
    return res_texts


def main(args):
    test_files = list()
    for test_pattern in args.tests:
        test_files += list(glob(test_pattern))

    text_stat = TextStatEvaluator()
    for file in test_files:
        attacked_samples = load_attacked_texts(file)
        metric_report = text_stat.do_eval(attacked_samples)

        print(
            "*************************\n"
            f"[{file}]\n"
            f"{metric_report}\n"
            "*************************\n",
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A simple argument parser example")
    parser.add_argument("--tests", nargs='+', default=[],)
    args = parser.parse_args()
    main(args)
