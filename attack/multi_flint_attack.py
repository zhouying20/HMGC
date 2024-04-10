import os
import json
import logging
import argparse
import tensorflow as tf
import torch.multiprocessing as mp
mp.set_start_method("spawn", force=True)

from tqdm import tqdm
from datetime import datetime
from textflint.adapter import auto_dataset
from textattack.shared import AttackedText
from textattack.goal_function_results import GoalFunctionResultStatus

from attack.methods.models import SurrogateDetectionModel
from utils.conf_util import setup_logger


attacking = None

logger = logging.getLogger()
setup_logger(logger)


def init(model_path, attacking_method="dualir", n_gpu=3):
    global attacking

    # get unique process id by calling current_process, to set CUDA visiability
    p_idx = int(mp.current_process()._identity[0]) # base started with 1
    gpu_i = (p_idx - 1) * n_gpu
    # gpus = list(map(str, range(gpu_i, gpu_i+n_gpu)))
    # os.environ["CUDA_CUDA_VISIBLE_DEVICES"] = ",".join(gpus)
    tf_gpus = tf.config.list_physical_devices("GPU")

    if n_gpu == 3:
        os.environ["VICTIM_DEVICE"] = f"cuda:{gpu_i + 0}"
        os.environ["PPL_DEVICE"] = f"cuda:{gpu_i + 1}"
        os.environ["TA_DEVICE"] = f"cuda:{gpu_i + 2}"
        # text_attack 会在一开始继承父进程的 TA_DEVICE 并初始化该 device，因此需要进行额外 patch
        from textattack.shared import utils
        utils.device = f"cuda:{gpu_i + 2}"
        tf.config.set_visible_devices(tf_gpus[gpu_i+2], "GPU")
    elif n_gpu == 1:
        os.environ["VICTIM_DEVICE"] = f"cuda:{gpu_i}"
        os.environ["PPL_DEVICE"] = f"cuda:{gpu_i}"
        os.environ["TA_DEVICE"] = f"cuda:{gpu_i}"
        from textattack.shared import utils
        utils.device = f"cuda:{gpu_i}"
        tf.config.set_visible_devices(tf_gpus[gpu_i], "GPU")
    else:
        raise NotImplementedError()

    if attacking_method == "dualir":
        from attack.recipes.rspmu_mlm_dualir import get_recipe
        recipe_func = get_recipe
    elif attacking_method == "wir":
        from attack.recipes.rspmu_mlm_wir import get_recipe
        recipe_func = get_recipe
    elif attacking_method == "greedy":
        from attack.recipes.rspmu_mlm_greedy import get_recipe
        recipe_func = get_recipe
    elif attacking_method == "no_pos":
        from attack.recipes.ablation_rsmu_mlm_dualir_no_pos import get_recipe
        recipe_func = get_recipe
    elif attacking_method == "no_use":
        from attack.recipes.ablation_rspm_mlm_dualir_no_use import get_recipe
        recipe_func = get_recipe
    elif attacking_method == "no_max_perturbed":
        from attack.recipes.ablation_rspu_mlm_dualir_no_max_perturbed import get_recipe
        recipe_func = get_recipe
    else:
        raise NotImplementedError(f"not supported attacking recipe -> {attacking_method}")

    # init model specific arguments
    # TODO, we set human class index to 1 in early surrogate model training on CheckGPT
    if "checkgpt" in model_path.lower():
        label2id = {"gpt": 0, "human": 1, "tied": 0}
        target_cls = 0
    else:
        label2id = {"human": 0, "gpt": 1, "tied": 1}
        target_cls = 1

    victim_model = SurrogateDetectionModel(model_path, batch_size=128, label2id=label2id)
    attacking = recipe_func(target_cls)
    attacking.init_goal_function(victim_model)

    logger.info(f"run attacking with {recipe_func}")
    logger.info(
        "*******************************\n"
        f"initializing process-{p_idx}...]\n"
        f"\t gpus: {os.environ.get('CUDA_CUDA_VISIBLE_DEVICES', None)}\n"
        f"\t victim on {os.environ.get('VICTIM_DEVICE', None)}\n"
        f"\t ppl on {os.environ.get('PPL_DEVICE', None)}\n"
        f"\t textattack on {os.environ.get('TA_DEVICE', None)}\n"
        f"\t attacking recipe: \n"
        f"{attacking.print()}\n"
        "*******************************\n",
        # flush=True,
    )


class MultiProcessingHelper:
    def __init__(self):
        self.total = None

    def __call__(self, data_samples, trans_save_path, func, workers=None, init_fn=None, init_args=None):
        self.total = len(data_samples)
        with mp.Pool(workers, initializer=init_fn, initargs=init_args) as pool, \
             tqdm(pool.imap(func, data_samples), total=self.total, dynamic_ncols=True) as pbar, \
             open(trans_save_path, "wt") as w_trans:
                for trans_res in pbar:
                    if trans_res is None: continue
                    w_trans.write(json.dumps(trans_res.dump(), ensure_ascii=False) + "\n")


def init_sample_from_textattack(ori):
    text_input, label_str = ori.to_tuple()
    label_output = attacking.goal_function.model.label2id[label_str]
    attacked_text = AttackedText(text_input)
    if attacked_text.num_words <= 2:
        logger.debug(f"The initial text -> [{attacked_text.text}] is less than 2 words, will skip this sample now!!!")
        goal_function_result = None
    else:
        goal_function_result, _ = attacking.goal_function.init_attack_example(
            attacked_text, label_output
        )
    return goal_function_result


def do_attack_one(ori_one):
    goal_function_result = init_sample_from_textattack(ori_one)

    if goal_function_result is None or \
        goal_function_result.goal_status == GoalFunctionResultStatus.SKIPPED:
        return None
    else:
        result = attacking.attack_one(goal_function_result)
        train_data_dict = result.perturbed_result.attacked_text._text_input
        trans_sample = ori_one.replace_fields(
            list(train_data_dict.keys()),
            list(train_data_dict.values()),
        )
        return trans_sample


def main(args):
    # prepare test data
    sample_list = list()
    with open(args.data_file, "r") as rf:
        for line in rf:
            r_j = json.loads(line.strip())
            sample_list.append({
                "x": r_j[args.text_key],
                "y": r_j[args.label_key],
            })
    dataset = auto_dataset(sample_list, task="SA")

    output_file = os.path.join(
        args.output_dir,
        "attacked-{}-{}.jsonl".format(args.attacking_method, datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    )
    worker = MultiProcessingHelper()
    worker(
        dataset,
        output_file,
        func=do_attack_one,
        workers=args.num_workers,
        init_fn=init,
        init_args=(args.model_name_or_path, args.attacking_method, args.num_gpu_per_process, ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A simple argument parser example")
    parser.add_argument("--model_name_or_path", required=True, help="TextFlint Model path")
    parser.add_argument("--data_file", required=True, help="TextFlint test file path")
    parser.add_argument("--output_dir", required=True, help="Directory to save the attacked samples")

    parser.add_argument("--attacking_method", type=str, default="dualir", help="Attacking method")
    parser.add_argument("--num_gpu_per_process", type=int, default=3, help="Number of gpus of one process")
    parser.add_argument("--num_workers", type=int, default=2, help="Total gpu usage is num_workers * num_gpu_per_process")
    parser.add_argument("--text_key", type=str, default="text", help="Text key for json object")
    parser.add_argument("--label_key", type=str, default="label", help="Label key for json object")
    args = parser.parse_args()
    main(args)
