# src/eval/rouge_evaluator.py
from rouge_score import rouge_scorer

class RougeEvaluator:
    def __init__(self):
        self.scorer = rouge_scorer.RougeScorer(
            ["rouge1", "rouge2", "rougeL"],
            use_stemmer=True
        )

    def compute_scores(self, reference: str, candidate: str) -> dict:
        scores = self.scorer.score(reference, candidate)
        return {
            "rouge1": scores["rouge1"].fmeasure,
            "rouge2": scores["rouge2"].fmeasure,
            "rougeL": scores["rougeL"].fmeasure,
        }