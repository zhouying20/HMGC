import os
import torch

from abc import ABC
from torch.nn import CrossEntropyLoss
from transformers import GPTNeoXForCausalLM, AutoTokenizer


class HuggingFaceModel(ABC):
    def __init__(self, seq_max_len, batch_size, device="cuda") -> None:
        super().__init__()
        self.model = None
        self.tokenizer = None

        self.seq_max_len = seq_max_len
        self.batch_size = batch_size
        device = os.environ.get("PPL_DEVICE", device)
        self.device = torch.device(device)

    def eval_ppl(
        self, data_samples,
        add_start_token: bool = False,
    ):
        r"""
        :param list[str] data_samples: list of input text
        :return: list obj of ppls of self.model
        """
        batch_size = self.batch_size
        max_length = self.seq_max_len

        # if batch_size > 1 (which generally leads to padding being required), and
        # if there is not an already assigned pad_token, assign an existing
        # special token to also be the padding token
        if self.tokenizer.pad_token is None and batch_size > 1:
            existing_special_tokens = list(self.tokenizer.special_tokens_map_extended.values())
            # check that the model already has at least one special token defined
            assert (
                len(existing_special_tokens) > 0
            ), "If batch_size > 1, model must have at least one special token to use for padding. Please use a different model or set batch_size=1."
            # assign one of the special tokens to also be the pad token
            self.tokenizer.add_special_tokens({"pad_token": existing_special_tokens[0]})

        if add_start_token and max_length:
            # leave room for <BOS> token to be added:
            assert (
                self.tokenizer.bos_token is not None
            ), "Input model must already have a BOS token if using add_start_token=True. Please use a different model, or set add_start_token=False"
            max_tokenized_len = max_length - 1
        else:
            max_tokenized_len = max_length

        encodings = self.tokenizer(
            data_samples,
            add_special_tokens=False,
            padding=True,
            truncation=True if max_tokenized_len else False,
            max_length=max_tokenized_len,
            return_tensors="pt",
            return_attention_mask=True,
        ).to(self.device)

        encoded_texts = encodings["input_ids"]
        attn_masks = encodings["attention_mask"]

        # check that each input is long enough:
        if add_start_token:
            assert torch.all(torch.ge(attn_masks.sum(1), 1)), "Each input text must be at least one token long."
        else:
            assert torch.all(
                torch.ge(attn_masks.sum(1), 2)
            ), "When add_start_token=False, each input text must be at least two tokens long. Run with add_start_token=True if inputting strings of only one token, and remove all empty input strings."

        ppls = []
        loss_fn = CrossEntropyLoss(reduction="none")

        for start_index in range(0, len(encoded_texts), batch_size):
            end_index = min(start_index + batch_size, len(encoded_texts))
            encoded_batch = encoded_texts[start_index:end_index]
            attn_mask = attn_masks[start_index:end_index]

            if add_start_token:
                bos_tokens_tensor = torch.tensor(
                    [[self.tokenizer.bos_token_id]] * encoded_batch.size(dim=0)
                ).to(self.device)
                encoded_batch = torch.cat([bos_tokens_tensor, encoded_batch], dim=1)
                attn_mask = torch.cat(
                    [torch.ones(bos_tokens_tensor.size(), dtype=torch.int64).to(self.device), attn_mask],
                    dim=1
                )

            labels = encoded_batch

            with torch.no_grad():
                out_logits = self.model(encoded_batch, attention_mask=attn_mask).logits

            shift_logits = out_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            shift_attention_mask_batch = attn_mask[..., 1:].contiguous()
            perplexity_batch = torch.exp(
                (loss_fn(shift_logits.transpose(1, 2), shift_labels) * shift_attention_mask_batch).sum(1)
                / shift_attention_mask_batch.sum(1)
            )

            ppls += perplexity_batch.cpu().tolist()

        return ppls


class PythiaPPLModel(HuggingFaceModel):
    def __init__(self, seq_max_len, batch_size, device="cuda") -> None:
        super().__init__(seq_max_len, batch_size, device)
        self.prepare_model()

    def prepare_model(self,):
        self.model = GPTNeoXForCausalLM.from_pretrained(
            "EleutherAI/pythia-2.8b-deduped",
        )
        self.model = self.model.to(self.device)

        self.tokenizer = AutoTokenizer.from_pretrained(
            "EleutherAI/pythia-2.8b-deduped"
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token


def main():
    pythia = PythiaPPLModel(512, 64)
    samples = [
        "lorem ipsum", "Happy Birthday!", "Bienvenue zhichi zhihu searcl"
    ]
    print(pythia.eval_ppl(samples, add_start_token=False))


if __name__ == "__main__":
    main()
