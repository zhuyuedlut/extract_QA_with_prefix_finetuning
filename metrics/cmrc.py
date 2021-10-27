import datasets

from .evalution import evaluate

_CITATION = """\
@inproceedings{cui-emnlp2019-cmrc2018,
    title = "A Span-Extraction Dataset for {C}hinese Machine Reading Comprehension",
    author = "Cui, Yiming  and
      Liu, Ting  and
      Che, Wanxiang  and
      Xiao, Li  and
      Chen, Zhipeng  and
      Ma, Wentao  and
      Wang, Shijin  and
      Hu, Guoping",
    booktitle = "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP)",
    month = nov,
    year = "2019",
    address = "Hong Kong, China",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/D19-1600",
    doi = "10.18653/v1/D19-1600",
    pages = "5886--5891",
}
"""

_DESCRIPTION = """
Stanford Question Answering Dataset (cmrc2018) is a reading comprehension dataset, consisting of questions posed by
crowdworkers, where the answer to every question is a segment of text, or span,
from the corresponding reading passage.
"""

_KWARGS_DESCRIPTION = """
Computes SQuAD scores (F1 and EM).
Args:
    predictions: List of question-answers dictionaries with the following key-values:
        - 'id': id of the question-answer pair as given in the references (see below)
        - 'prediction_text': the text of the answer
    references: List of question-answers dictionaries with the following key-values:
        - 'id': id of the question-answer pair (see above),
        - 'answers': a Dict in the SQuAD dataset format
            {
                'text': list of possible texts for the answer, as a list of strings
                'answer_start': list of start positions for the answer, as a list of ints
            }
            Note that answer_start values are not taken into account to compute the metric.
Returns:
    'exact_match': Exact match (the normalized answer exactly match the gold answer)
    'f1': The F-score of predicted tokens versus the gold answer
Examples:
    >>> predictions = [{'prediction_text': '1976', 'id': '56e10a3be3433e1400422b22'}]
    >>> references = [{'answers': {'answer_start': [97], 'text': ['1976']}, 'id': '56e10a3be3433e1400422b22'}]
    >>> squad_metric = datasets.load_metric("squad")
    >>> results = squad_metric.compute(predictions=predictions, references=references)
    >>> print(results)
    {'exact_match': 100.0, 'f1': 100.0}
"""

@datasets.utils.file_utils.add_start_docstrings(_DESCRIPTION, _KWARGS_DESCRIPTION)
class Cmrc(datasets.Metric):
    def _info(self):
        return datasets.MetricInfo(
            description=_DESCRIPTION,
            citation=_CITATION,
            inputs_description=_KWARGS_DESCRIPTION,
            features=datasets.Features(
                {
                    "predictions": {"id": datasets.Value("string"),
                                    "prediction_text": datasets.Value("string")},
                    "references": {
                        "id": datasets.Value("string"),
                        "answers": datasets.features.Sequence(
                            {
                                "text": datasets.Value("string"),
                                "answer_start": datasets.Value("int32"),
                            }
                        ),
                    },
                }
            ),
        )

    def _compute(self, predictions, references):
        pred_dict = {prediction["id"]: prediction["prediction_text"] for
                     prediction in predictions}
        dataset = [
            {
                "paragraphs": [
                    {
                        "qas": [
                            {
                                "answers": [{"text": answer_text} for
                                            answer_text in
                                            ref["answers"]["text"]],
                                "id": ref["id"],
                            }
                            for ref in references
                        ]
                    }
                ]
            }
        ]
        score = evaluate(dataset=dataset, predictions=pred_dict)
        return score