from transformers import BartConfig, BartTokenizer

from src.models.qa_modeling_bart import PrefixBartQAModel
from src.models.qa_modeling_bart import BartQAModel


MODEL_CLASSES = {
    'bart': (BartConfig, BartQAModel, BartTokenizer),
    'prefix': (BartConfig, PrefixBartQAModel, BartTokenizer)
}
