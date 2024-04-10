import os
import torch
import numpy as np

from torch.nn import CrossEntropyLoss
from textattack.models.wrappers import PyTorchModelWrapper
from textflint.input.model.flint_model.torch_model import TorchModel
from textflint.input.model.metrics.metrics import accuracy_score as Accuracy
from transformers import (
    RobertaConfig,
    RobertaForSequenceClassification,
    RobertaTokenizer,
)


class SurrogateDetectionModel(TorchModel):
    """
    Model wrapper for Surrogate Detection Model implemented by pytorch.
    """
    def __init__(
        self,
        model_name_or_path,
        batch_size=64,
        device="cuda",
        label2id={"gpt": 0, "human": 1, "tied": 0}
    ):
        detection_config = RobertaConfig.from_pretrained(model_name_or_path, num_labels=2)

        super().__init__(
            model=RobertaForSequenceClassification.from_pretrained(
                model_name_or_path,
                config=detection_config,
            ),
            tokenizer=RobertaTokenizer.from_pretrained(model_name_or_path),
            task='SA',
            batch_size=batch_size,
        )
        self.label2id = label2id

        device = os.environ.get("VICTIM_DEVICE", device)
        self.model_device = torch.device(device)
        self.model = self.model.to(self.model_device)

    def __call__(self, batch_texts):
        r"""
        Tokenize text, convert tokens to id and run the model.

        :param batch_texts: (batch_size,) batch text input
        :return: numpy.array()
        """
        inputs = self.encode(batch_texts)

        with torch.no_grad():
            logits = self.model(**inputs).logits

        return logits.detach().cpu().numpy()
    
    def _get_eval_metrics(self, golds, preds, prefix):
        all_labels = np.array(golds)
        all_outputs = np.array(preds)
        gpt_labels = all_labels[all_labels == 0]
        gpt_outputs = all_outputs[all_labels == 0]
        human_labels = all_labels[all_labels == 1]
        human_outputs = all_outputs[all_labels == 1]

        res = {}
        if len(gpt_labels) != 0:
            res[prefix + "gpt_accuracy"] = Accuracy(gpt_labels, gpt_outputs)
        if len(human_labels) != 0:
            res[prefix + "human_accuracy"] = Accuracy(human_labels, human_outputs)
        res[prefix + "accuracy"] = Accuracy(all_labels, all_outputs)

        return res

    def evaluate(self, data_samples, prefix=''):
        r"""
        :param list[Sample] data_samples: list of Samples
        :param str prefix: name prefix to add to metrics
        :return: dict obj to save metrics result

        """
        outputs = []
        labels = []
        i = 0

        while i < len(data_samples):
            batch_samples = data_samples[i: i + self.batch_size]
            batch_inputs, batch_labels = self.unzip_samples(batch_samples)
            labels += batch_labels
            logits = self.__call__(batch_inputs)
            predicts = np.argmax(logits, axis=1)
            outputs += predicts.tolist()
            i += self.batch_size

        return self._get_eval_metrics(labels, outputs, prefix)

    def encode(self, inputs):
        r"""
        Tokenize inputs and convert it to ids.

        :param inputs: model original input
        :return: list of inputs ids

        """
        return self.tokenizer(
            inputs,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        ).to(self.model_device)

    def _tokenize(self, inputs):
        """Helper method for `tokenize`"""
        assert isinstance(inputs, (list, tuple))
        assert len(inputs) == 1
        return self.tokenizer.tokenize(
            inputs[0]
        )

    def tokenize(self, inputs, strip_prefix=False):
        """Helper method that tokenizes input strings
        Args:
            inputs (list[str]): list of input strings
            strip_prefix (bool): If `True`, we strip auxiliary characters added to tokens as prefixes (e.g. "##" for BERT, "Ġ" for RoBERTa)
        Returns:
            tokens (list[list[str]]): List of list of tokens as strings
        """
        tokens = self._tokenize(inputs)
        if strip_prefix:
            # `aux_chars` are known auxiliary characters that are added to tokens
            strip_chars = ["##", "Ġ", "__"]
            # TODO: Find a better way to identify prefixes. These depend on the model, so cannot be resolved in ModelWrapper.

            def strip(s, chars):
                for c in chars:
                    s = s.replace(c, "")
                return s

            tokens = [[strip(t, strip_chars) for t in x] for x in tokens]

        return tokens

    def unzip_samples(self, data_samples):
        r"""
        Unzip sample to input texts and labels.

        :param list[Sample] data_samples: list of Samples
        :return: (inputs_text), labels.
        """
        x = []
        y = []

        for sample in data_samples:
            x.append(sample['x'])
            y.append(self.label2id[sample['y']])

        return x, y

    def get_grad(self, *inputs):
        r"""
        Get gradient of loss with respect to input tokens.

        :param tuple inputs: tuple of original texts
        """
        return self.get_model_grad(inputs)

    def get_model_grad(self, text_inputs, loss_fn=CrossEntropyLoss()):
        r"""
        Get gradient of loss with respect to input tokens.

        :param str|[str] text_inputs: input string or input string list
        :param torch.nn.Module loss_fn: loss function.
            Default is `torch.nn.CrossEntropyLoss`
        :return: Dict of ids, tokens, and gradient as numpy array.
        """
        if not hasattr(self.model, "get_input_embeddings"):
            raise AttributeError(
                f"{type(self.model)} must have method `get_input_embeddings` "
                f"that returns `torch.nn.Embedding` object that represents "
                f"input embedding layer"
            )

        if not isinstance(loss_fn, torch.nn.Module):
            raise ValueError("Loss function must be of type `torch.nn.Module`.")

        if isinstance(text_inputs, str):
            _texts = [text_inputs]
        elif isinstance(text_inputs, (list, tuple)):
            _texts = text_inputs
        else:
            raise NotImplementedError(f"not supported input type -> f{type(text_inputs)}")

        self.model.train()

        embedding_layer = self.model.get_input_embeddings()
        original_state = embedding_layer.weight.requires_grad
        embedding_layer.weight.requires_grad = True

        emb_grads = []
        def grad_hook(module, grad_in, grad_out):
            emb_grads.append(grad_out[0])
        emb_hook = embedding_layer.register_backward_hook(grad_hook)

        self.model.zero_grad()
        inputs = self.encode(_texts)
        logits = self.model(**inputs).logits

        output = logits.argmax(dim=1)
        loss = loss_fn(logits, output)
        loss.backward()

        # grad w.r.t to word embeddings
        if emb_grads[0].shape[1] == 1:
            grad = torch.transpose(emb_grads[0], 0, 1)[0].cpu().numpy()
        else:
            # gradient has shape [1,max_sequence,_]
            grad = emb_grads[0][0].cpu().numpy()

        embedding_layer.weight.requires_grad = original_state
        emb_hook.remove()
        self.model.eval()

        res = {"ids": inputs["input_ids"][0].cpu().tolist(), "gradient": grad}
        return res
