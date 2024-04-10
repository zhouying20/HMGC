"""Generate answers with local models.
"""
import os
import json
import time
import torch
import argparse

from tqdm import tqdm
from fastchat.model import load_model, get_conversation_template


rewriting_prompt = """
Please rewrite the following article while incorporating the word usage patterns:
{origin_article}

Aim to diverge from the original text's style and expression.
""".strip()


def load_articles(article_file):
    res = list()
    with open(article_file) as rf:
        for idx, line in enumerate(rf):
            lj = json.loads(line)
            if "text" in lj:
                _text = lj["text"]
            elif "answer" in lj:
                _text = lj["answer"]
            else:
                raise NotImplementedError(f"no text field in {article_file}")
            res.append({
                "sample_id": idx,
                "text": _text,
            })
    return res


def run_rewrite(
    model_path,
    model_id,
    test_file,
    output_file,
    max_new_token,
    num_choices,
    num_gpus_per_model,
    num_gpus_total,
    max_gpu_memory,
    prompt_mode,
    temperature,
):
    articles = load_articles(test_file)

    # random shuffle the articles to balance the loading
    # random.shuffle(articles)

    # Split the question file into `num_gpus` files
    assert num_gpus_total % num_gpus_per_model == 0
    use_ray = num_gpus_total // num_gpus_per_model > 1

    if use_ray:
        get_answers_func = ray.remote(num_gpus=num_gpus_per_model)(gen_model_answers).remote
    else:
        get_answers_func = gen_model_answers

    chunk_size = len(articles) // (num_gpus_total // num_gpus_per_model) // 2
    ans_handles = []
    for i in range(0, len(articles), chunk_size):
        ans_handles.append(
            get_answers_func(
                model_path,
                model_id,
                articles[i : i+chunk_size],
                output_file,
                max_new_token,
                num_choices,
                num_gpus_per_model,
                max_gpu_memory,
                prompt_mode,
                temperature,
            )
        )

    if use_ray:
        ray.get(ans_handles)


@torch.inference_mode()
def gen_model_answers(
    model_path,
    model_id,
    articles,
    output_file,
    max_new_token,
    num_choices,
    num_gpus_per_model,
    max_gpu_memory,
    prompt_mode,
    temperature=0.75,
):
    model, tokenizer = load_model(
        model_path,
        device="cuda",
        num_gpus=num_gpus_per_model,
        max_gpu_memory=max_gpu_memory,
        load_8bit=False,
        cpu_offloading=False,
        debug=False,
    )
    assert num_choices == 1 # TODO now only support 1 article to be generated

    for art in tqdm(articles, dynamic_ncols=True):
        choices = []
        for i in range(num_choices):
            torch.manual_seed(i)
            conv = get_conversation_template(model_id)
            user_prompt = rewriting_prompt.format(origin_article=art["text"])
            conv.append_message(conv.roles[0], user_prompt)
            conv.append_message(conv.roles[1], None)
            if prompt_mode == "plain":
                prompt_for_tok = user_prompt.strip()
            else:
                prompt_for_tok = conv.get_prompt()
            input_ids = tokenizer([prompt_for_tok]).input_ids

            if temperature < 1e-4:
                do_sample = False
            else:
                do_sample = True

            # some models may error out when generating long outputs
            try:
                output_ids = model.generate(
                    torch.as_tensor(input_ids).cuda(),
                    do_sample=do_sample,
                    temperature=temperature,
                    max_new_tokens=max_new_token,
                    use_cache=True,
                )
                if model.config.is_encoder_decoder:
                    output_ids = output_ids[0]
                else:
                    output_ids = output_ids[0][len(input_ids[0]) :]

                # be consistent with the template's stop_token_ids
                if conv.stop_token_ids:
                    stop_token_ids_index = [
                        i
                        for i, id in enumerate(output_ids)
                        if id in conv.stop_token_ids
                    ]
                    if len(stop_token_ids_index) > 0:
                        output_ids = output_ids[: stop_token_ids_index[0]]

                output = tokenizer.decode(
                    output_ids,
                    spaces_between_special_tokens=False,
                )
                if conv.stop_str and output.find(conv.stop_str) > 0:
                    output = output[: output.find(conv.stop_str)]
                for special_token in tokenizer.special_tokens_map.values():
                    if isinstance(special_token, list):
                        for special_tok in special_token:
                            output = output.replace(special_tok, "")
                    else:
                        output = output.replace(special_token, "")
                output = output.strip()

                if conv.name == "xgen" and output.startswith("Assistant:"):
                    output = output.replace("Assistant:", "", 1).strip()
            except RuntimeError as e:
                print("ERROR question ID: ", art["sample_id"])
                output = "ERROR"
            conv.messages[-1][-1] = output
            choices.append({"index": i, "output": output})

        # Dump generations
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(os.path.expanduser(output_file), "a") as fout:
            for c in choices:
                ans_json = {
                    "sample_id": art["sample_id"],
                    "x": c["output"],
                    "choice_id": c["index"],
                    "model_id": model_id,
                    "tstamp": time.time(),
                }
                fout.write(json.dumps(ans_json, ensure_ascii=False) + "\n")


def reorg_answer_file(answer_file):
    """Sort by question id and de-duplication"""
    answers = {}
    with open(answer_file, "r") as fin:
        for l in fin:
            qid = json.loads(l)["sample_id"] # TODO now num_choices should be 1
            answers[qid] = l

    qids = sorted(list(answers.keys()))
    with open(answer_file, "w") as fout:
        for qid in qids:
            fout.write(answers[qid])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", type=str, required=True)
    parser.add_argument("--model-path", type=str, required=True, help="The path to the weights. This can be a local folder or a Hugging Face repo ID.")

    parser.add_argument("--test-file", type=str, help="The input article file.")
    parser.add_argument("--output-file", type=str, help="The output generation file.")
    parser.add_argument("--temperature", type=float, default=0.75, help="The maximum number of new generated tokens.")
    parser.add_argument("--max-new-token", type=int, default=1024, help="The maximum number of new generated tokens.",)
    parser.add_argument("--num-choices", type=int, default=1, help="How many completion choices to generate.")
    parser.add_argument("--num-gpus-per-model", type=int, default=1, help="The number of GPUs per model.")
    parser.add_argument("--num-gpus-total", type=int, default=1, help="The total number of GPUs.")
    parser.add_argument("--max-gpu-memory", type=str, help="Maxmum GPU memory used for model weights per GPU.",)
    parser.add_argument(
        "--prompt-mode", type=str, default="conversation", choices=["plain", "conversation"],
        help=(
            "How to construct user prompt. "
            "`plain` directly use question as input to generate output. "
            "`conversation` use FastChat model_adapter's Conversation to construct. "
        ),
    )
    args = parser.parse_args()

    if args.num_gpus_total // args.num_gpus_per_model > 1:
        import ray
        ray.init()

    run_rewrite(
        args.model_path,
        args.model_id,
        args.test_file,
        args.output_file,
        args.max_new_token,
        args.num_choices,
        args.num_gpus_per_model,
        args.num_gpus_total,
        args.max_gpu_memory,
        args.prompt_mode,
        args.temperature,
    )
    reorg_answer_file(args.output_file)
