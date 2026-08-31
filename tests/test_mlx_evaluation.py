import pytest

from localllm_bench.mlx_evaluation import build_evaluation_prompt
from localllm_bench.training_data import TrainingItem


class FakeTokenizer:
    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
    ) -> str:
        assert tokenize is False
        assert add_generation_prompt is True
        return " | ".join(message["content"] for message in messages)


def test_build_evaluation_prompt_uses_chat_template() -> None:
    item = TrainingItem(
        item_id="test-1",
        source_document_id="doc",
        split="test",
        question="What is demand response?",
        answer="A change in electricity use.",
    )
    prompt = build_evaluation_prompt(FakeTokenizer(), item)
    assert "energy-market question" in prompt
    assert "What is demand response?" in prompt


def test_build_evaluation_prompt_rejects_token_ids() -> None:
    class BadTokenizer:
        def apply_chat_template(self, *args: object, **kwargs: object) -> list[int]:
            return [1, 2]

    item = TrainingItem(
        item_id="test-1",
        source_document_id="doc",
        split="test",
        question="Question?",
        answer="Answer.",
    )
    with pytest.raises(ValueError, match="did not return text"):
        build_evaluation_prompt(BadTokenizer(), item)
