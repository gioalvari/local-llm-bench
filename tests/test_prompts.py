from localllm_bench.config import PromptArm
from localllm_bench.dataset import EvaluationItem
from localllm_bench.prompts import build_evaluation_prompt


def _item() -> EvaluationItem:
    return EvaluationItem.model_validate(
        {
            "item_id": "item-1",
            "source_document_id": "doc",
            "source_url": "https://example.org",
            "source_revision": "1",
            "category": "grid",
            "context": "The answer is storage.",
            "question": "What is the answer?",
            "answer_type": "text",
            "canonical_answer": "storage",
            "evidence_spans": ["answer is storage"],
            "split": "smoke",
        }
    )


def test_prompt_arms_are_distinct_and_grounded() -> None:
    zero = build_evaluation_prompt(_item(), PromptArm.ZERO_SHOT)
    engineered = build_evaluation_prompt(_item(), PromptArm.ENGINEERED)
    assert zero != engineered
    assert "The answer is storage." in zero
    assert "using only the provided context" in engineered
