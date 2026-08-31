import pytest

from localllm_bench.evaluation import aggregate_records


def test_aggregate_records_by_prompt_arm() -> None:
    records = [
        {
            "prompt_arm": "zero-shot",
            "answer_type": "numeric",
            "ttft_ns": 10_000_000,
            "e2e_latency_ns": 100_000_000,
            "score": {
                "schema_valid": 1.0,
                "answer_accuracy": 1.0,
                "exact_match": 1.0,
                "token_f1": 1.0,
                "numeric_accuracy": 1.0,
                "unit_accuracy": 1.0,
                "parsed_answer": "1000",
            },
        },
        {
            "prompt_arm": "zero-shot",
            "answer_type": "classification",
            "ttft_ns": 20_000_000,
            "e2e_latency_ns": 300_000_000,
            "score": {
                "schema_valid": 0.0,
                "answer_accuracy": 0.0,
                "exact_match": 0.0,
                "token_f1": 0.5,
                "numeric_accuracy": None,
                "unit_accuracy": None,
                "parsed_answer": None,
            },
        },
    ]
    summary = aggregate_records(records)["zero-shot"]
    assert summary["items"] == 2
    assert summary["answer_accuracy"] == 0.5
    assert summary["scorable_response_rate"] == 0.5
    assert summary["median_ttft_ms"] == 15.0
    assert summary["quality_adjusted_answers_per_second"] == pytest.approx(2.5)
    assert summary["numeric_accuracy"] == 1.0
