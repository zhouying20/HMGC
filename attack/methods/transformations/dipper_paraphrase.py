r"""
Paraphrasing class
==========================================================
"""

__all__ = ['Paraphrase']

import nltk
import torch

from nltk.tokenize import sent_tokenize
nltk.download('punkt')

from textflint.common import device as default_device
from textflint.generation.transformation import Transformation


class Paraphrase(Transformation):
    r"""
    Back Translation with hugging-face translation models.
    A sentence can only be transformed into one sentence at most.

    """
    def __init__(
        self,
        trans_iter=1,
        lex_diversity=40,
        order_diversity=40,
        sent_interval=3,
        use_prefix=False,
        model_name_or_path="kalpeshk2011/dipper-paraphraser-xxl",
        device=None,
        **kwargs,
    ):
        r"""
        :param str from_model_name: model to translate original language to
            target language
        :param str to_model_name: model to translate target language to
            original language
        :param device: indicate utilize cpu or which gpu device to
            run neural network

        """
        super().__init__()
        assert lex_diversity in [0, 20, 40, 60, 80, 100], "Lexical diversity must be one of 0, 20, 40, 60, 80, 100."
        assert order_diversity in [0, 20, 40, 60, 80, 100], "Order diversity must be one of 0, 20, 40, 60, 80, 100."

        self.trans_iter = trans_iter
        self.lex_diversity = lex_diversity
        self.order_diversity = order_diversity
        self.sent_interval = sent_interval
        self.use_prefix = use_prefix

        self.device = self.get_device(device) if device else default_device
        self.model_name_or_path = model_name_or_path
        self._model = None
        self._tokenizer = None

    def __repr__(self):
        return 'Paraphrase'

    @staticmethod
    def get_device(device):
        r"""
        Get gpu or cpu device.

        :param str device: device string
                           "cpu" means use cpu device.
                           "cuda:0" means use gpu device which index is 0.
        :return: device in torch.
        """
        if "cuda" not in device:
            return torch.device("cpu")
        else:
            return torch.device(device)

    def get_model(self):
        # accelerate load efficiency
        from transformers import T5Tokenizer, T5ForConditionalGeneration

        """ Load models of translation. """
        self._tokenizer = T5Tokenizer.from_pretrained("google/t5-v1_1-xxl")
        self._model = T5ForConditionalGeneration.from_pretrained(self.model_name_or_path)
        self._model = self._model.to(self.device)

    def dipper_paraphrase(
        self, input_text, prefix="",
    ):
        """Paraphrase a text using the DIPPER model.

        Args:
            input_text (str): The text to paraphrase. Make sure to mark the sentence to be paraphrased between <sent> and </sent> blocks, keeping space on either side.
            lex_diversity (int): The lexical diversity of the output, choose multiples of 20 from 0 to 100. 0 means no diversity, 100 means maximum diversity.
            order_diversity (int): The order diversity of the output, choose multiples of 20 from 0 to 100. 0 means no diversity, 100 means maximum diversity.
            **kwargs: Additional keyword arguments like top_p, top_k, max_length.
        """
        lex_diversity = self.lex_diversity
        order_diversity = self.order_diversity
        sent_interval = self.sent_interval

        input_text = " ".join(input_text.split())
        sentences = sent_tokenize(input_text)

        cur_prefix = " ".join(prefix.replace("\n", " ").split())
        cur_output = ""
        for sent_idx in range(0, len(sentences), sent_interval):
            curr_sent_window = " ".join(sentences[sent_idx:sent_idx + sent_interval])
            if "no-context" in self.model_name_or_path:
                final_input_text = f"lexical = {lex_diversity}, order = {order_diversity} {curr_sent_window}"
            else:
                final_input_text = f"lexical = {lex_diversity}, order = {order_diversity} {cur_prefix} <sent> {curr_sent_window} </sent>"

            final_input = self._tokenizer([final_input_text], return_tensors="pt")
            final_input = {k: v.to(self.device) for k, v in final_input.items()}

            with torch.inference_mode(): # use same config as origin paper
                outputs = self._model.generate(**final_input, do_sample=True, top_p=0.75, top_k=None, max_length=512)
            outputs = self._tokenizer.batch_decode(outputs, skip_special_tokens=True)

            cur_prefix += " " + outputs[0]
            cur_output += " " + outputs[0]

        return cur_output.strip()

    def _transform(self, sample, n=1, field='x', **kwargs):
        if self._model is None:
            self.get_model()

        trans_sample = sample.clone(sample)
        for _ in range(self.trans_iter):
            sents = trans_sample.get_sentences(field)
            if self.use_prefix:
                prefix = sents[0]
                text = " ".join(sents[1:])
            else:
                prefix = ""
                text = " ".join(sents)
            paraphrased_text = self.dipper_paraphrase(text, prefix=prefix)
            trans_sample = trans_sample.replace_field(field, " ".join([prefix, paraphrased_text,]).strip())

        return [trans_sample, ]
