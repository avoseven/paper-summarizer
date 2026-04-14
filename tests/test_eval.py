# tests/test_eval.py
from src.eval.rouge_evaluator import RougeEvaluator

def test_rouge_evaluator():
    evaluator = RougeEvaluator()
    ref = "This is a reference summary."
    cand = "This is a candidate summary."
    scores = evaluator.compute_scores(ref, cand)
    assert "rouge1" in scores
    assert 0 <= scores["rouge1"] <= 1
    # など