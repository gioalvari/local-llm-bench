"""Frozen quality-evaluation prompt templates."""

from localllm_bench.config import PromptArm
from localllm_bench.dataset import EvaluationItem

PROMPT_VERSION = "1"


def build_evaluation_prompt(item: EvaluationItem, arm: PromptArm) -> str:
    """Render one frozen prompt arm for a source-grounded item."""
    if arm is PromptArm.ZERO_SHOT:
        return (
            f"Context:\n{item.context}\n\nQuestion: {item.question}\n"
            'Return JSON with keys "answer" and "unit".'
        )
    return (
        "Answer the question using only the provided context. Return exactly one "
        "JSON object and no markdown or explanation. Use this schema: "
        '{"answer":"short answer","unit":null}. For numeric answers, put only '
        'the number in "answer" and its unit in "unit". If the context does not '
        "contain the answer, use unknown.\n\n"
        f"CONTEXT\n{item.context}\n\nQUESTION\n{item.question}"
    )
