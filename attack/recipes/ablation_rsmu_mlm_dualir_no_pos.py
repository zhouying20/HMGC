import functools

from textattack.goal_functions import TargetedClassification
# from textattack.constraints.grammaticality import PartOfSpeech
from textattack.constraints.overlap import MaxWordsPerturbed
from textattack.constraints.pre_transformation import RepeatModification, StopwordModification
from textattack.constraints.semantics.sentence_encoders import UniversalSentenceEncoder
from textattack.transformations import WordSwapMaskedLM
from textflint.generation.attack import Attack # Note that here we use the Attack from textflint

from attack.methods.search_methods import GreedyDualWIR
from attack.methods.models.ppl_model import PythiaPPLModel


def get_recipe(target_cls):
    def init_ppl_model():
        return PythiaPPLModel(
            seq_max_len=512,
            batch_size=8,
        )

    goal_function = functools.partial(TargetedClassification, target_class=target_cls)

    constraints = [
        RepeatModification(), # no repeated modification
        StopwordModification(), # not modify stopword
        MaxWordsPerturbed(max_percent=0.4),
        UniversalSentenceEncoder(
            threshold=0.75,
            metric="cosine",
            compare_against_original=True,
            window_size=50,
        )
    ]

    transformation = WordSwapMaskedLM(batch_size=128)

    search_method = GreedyDualWIR(alpha=0.2, wir_method="gradient")
    search_method.load_llm_model = init_ppl_model

    attacking = Attack(goal_function, constraints, transformation, search_method)
    return attacking
