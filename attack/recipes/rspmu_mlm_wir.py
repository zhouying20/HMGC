import functools

from textattack.goal_functions import TargetedClassification
from textattack.search_methods import GreedyWordSwapWIR
from textattack.constraints.grammaticality import PartOfSpeech
from textattack.constraints.overlap import MaxWordsPerturbed
from textattack.constraints.pre_transformation import RepeatModification, StopwordModification
from textattack.constraints.semantics.sentence_encoders import UniversalSentenceEncoder
from textattack.transformations import WordSwapMaskedLM
from textflint.generation.attack import Attack # Note that here we use the Attack from textflint

# from attacker.methods.search_methods import GreedyDualWIR
# from attacker.methods.models.llm_model import PythiaPPLModel


def get_recipe(target_cls):
    def init_indices_to_order(initial_text):
        len_text = initial_text.num_words
        indices_to_order = list(range(len_text))
        return len_text, indices_to_order

    goal_function = functools.partial(TargetedClassification, target_class=target_cls)

    constraints = [
        RepeatModification(), # no repeated modification
        StopwordModification(), # not modify stopword
        PartOfSpeech(),
        MaxWordsPerturbed(max_percent=0.4),
        UniversalSentenceEncoder(
            threshold=0.75,
            metric="cosine",
            compare_against_original=True,
            window_size=50,
        )
    ]

    transformation = WordSwapMaskedLM(batch_size=128)

    search_method = GreedyWordSwapWIR(wir_method="gradient")
    search_method.get_indices_to_order = init_indices_to_order

    attacking = Attack(goal_function, constraints, transformation, search_method)
    return attacking
