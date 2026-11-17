from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

from langchain_core.messages import AIMessage, ToolMessage

from deepeval import evaluate
from deepeval.evaluate.configs import AsyncConfig, DisplayConfig, ErrorConfig
from deepeval.metrics import (
    FaithfulnessMetric,
    AnswerRelevancyMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    ContextualRelevancyMetric,
)
from deepeval.models import DeepEvalBaseLLM
from deepeval.test_case import LLMTestCase


def _content_to_text(message: Any) -> str:
    """Return plain text from a LangChain message / string / list of blocks."""
    content = getattr(message, "content", message)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
        return "".join(parts)
    return str(content)


# ---------------------------------------------------------------------------
# DeepEval judge wrapper around a LangChain chat model
# ---------------------------------------------------------------------------
class LangChainJudge(DeepEvalBaseLLM):
    """Wrap a LangChain chat model so DeepEval can use it as a judge."""

    def __init__(self, llm, model_name: str = "langchain-judge"):
        self.llm = llm
        self._model_name = model_name

    def load_model(self):
        return self.llm

    def generate(self, prompt: str, schema=None):
        # DeepEval passes a pydantic `schema` -> ask the model for structured
        # output (far more reliable than parsing free text from a 7B model).
        if schema is not None:
            try:
                return self.llm.with_structured_output(schema).invoke(prompt)
            except Exception:
                pass  # fall back to plain text; DeepEval will parse the JSON
        return _content_to_text(self.llm.invoke(prompt))

    async def a_generate(self, prompt: str, schema=None):
        if schema is not None:
            try:
                return await self.llm.with_structured_output(schema).ainvoke(prompt)
            except Exception:
                pass
        return _content_to_text(await self.llm.ainvoke(prompt))

    def get_model_name(self) -> str:
        return self._model_name


# ---------------------------------------------------------------------------
# Extract contexts + final answer from the response of RagPipeline.run_pipeline
# ---------------------------------------------------------------------------
# Matches "Document 1:\nContent: ...\nMetadata: {...}"
_DOC_BLOCK_RE = re.compile(
    r"Document\s+\d+:\s*\nContent:\s*(.*?)\nMetadata:",
    re.DOTALL,
)


def _get_messages(response: Any) -> list[Any]:
    if isinstance(response, dict):
        return response.get("messages", [])
    if isinstance(response, list):
        return response
    return []


def _extract_final_answer(response: Any) -> str:
    """Last AIMessage that has content and no pending tool_calls."""
    messages = _get_messages(response)
    if not messages:
        return str(response)

    for msg in reversed(messages):
        if isinstance(msg, AIMessage):
            has_tool_calls = bool(getattr(msg, "tool_calls", None))
            if not has_tool_calls and msg.content:
                return _content_to_text(msg)
    return ""


def _extract_contexts(
        response: Any,
        tool_names: set[str] | None = None,
) -> list[str]:
    """
    Collect retrieval results from the ToolMessages inside the agent response.
    If `tool_names` is None, every ToolMessage is treated as retrieval output
    (the agent only has the tools from build_retrieval_tools).
    """
    contexts: list[str] = []

    for msg in _get_messages(response):
        if not isinstance(msg, ToolMessage):
            continue
        if tool_names is not None and (msg.name or "") not in tool_names:
            continue

        raw = _content_to_text(msg).strip()
        if not raw:
            continue
        if "No relevant documents found." in raw and len(raw) < 60:
            continue

        matches = _DOC_BLOCK_RE.findall(raw)
        if matches:
            contexts.extend(m.strip() for m in matches if m.strip())
        else:
            contexts.append(raw)  # fallback: whole output as one context

    # Deduplicate, preserve order
    seen: set[str] = set()
    unique: list[str] = []
    for c in contexts:
        if c not in seen:
            seen.add(c)
            unique.append(c)
    return unique


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def evaluate_rag(
        pipeline,
        questions: Sequence[str],
        reference_answers: Sequence[str],
        llm_for_judge=None,
        retrieval_tool_names: set[str] | None = None,
        output_dir: str = r"D:\AI Source\Projects\list_of_projects\TeamDocs-AI\evaluation_results",
        top_k: int = 5,
) -> dict[str, Any]:
    """
    Evaluate RagPipeline with DeepEval.

    Args:
        pipeline:
            An instance of RagPipeline (uses pipeline.run_pipeline(query)).
        questions:
            Evaluation questions.
        reference_answers:
            Ground-truth answers (same length as questions).
        llm_for_judge:
            LangChain chat model used as the DeepEval judge.
            Default: pipeline._chat_model.
        retrieval_tool_names:
            Only ToolMessages from these tools count as retrieved context.
            Default None = all tools of the agent.
        output_dir:
            Directory for result files.
        top_k:
            For reporting only. Actual k is set in build_retrieval_tools(...).
    """
    if not questions:
        raise ValueError("questions cannot be empty.")
    if len(questions) != len(reference_answers):
        raise ValueError(
            "questions and reference_answers must have the same length."
        )

    if llm_for_judge is None:
        llm_for_judge = getattr(pipeline, "_chat_model", None)
    if llm_for_judge is None:
        raise ValueError("llm_for_judge is required for DeepEval.")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 1. Run the pipeline for every question, capture contexts
    records: list[dict[str, Any]] = []

    for index, (question, reference) in enumerate(
            zip(questions, reference_answers), start=1
    ):
        print(f"[{index}/{len(questions)}] Evaluating: {question}")

        response = pipeline.run_pipeline(question)

        answer = _extract_final_answer(response)
        contexts = _extract_contexts(response, retrieval_tool_names)

        if not contexts:
            print("   ! no retrieved context captured for this question")

        records.append({
            "question": question,
            "answer": answer,
            "contexts": contexts,
            "ground_truth": reference,
            "retrieved_context_count": len(contexts),
        })

    # 2. Save raw outputs first
    raw_df = pd.DataFrame([
        {
            "question": r["question"],
            "answer": r["answer"],
            "reference_answer": r["ground_truth"],
            "retrieved_contexts": json.dumps(
                r["contexts"], ensure_ascii=False
            ),
            "retrieved_context_count": r["retrieved_context_count"],
        }
        for r in records
    ])
    raw_file = output_path / "rag_outputs.csv"
    raw_df.to_csv(raw_file, index=False, encoding="utf-8-sig")

    # 3. Wrap LLM as DeepEval judge
    judge_llm = LangChainJudge(
        llm_for_judge,
        model_name=(
                getattr(llm_for_judge, "model_name", None)
                or getattr(llm_for_judge, "model", None)
                or "langchain-judge"
        ),
    )

    # 4. Build DeepEval test cases
    test_cases = [
        LLMTestCase(
            input=r["question"],
            actual_output=r["answer"],
            expected_output=r["ground_truth"],
            retrieval_context=r["contexts"] or ["(no context retrieved)"],
        )
        for r in records
    ]

    # 5. Define RAG metrics
    metrics = [
        FaithfulnessMetric(threshold=0.5, model=judge_llm),
        AnswerRelevancyMetric(threshold=0.5, model=judge_llm),
        ContextualPrecisionMetric(threshold=0.5, model=judge_llm),
        ContextualRecallMetric(threshold=0.5, model=judge_llm),
        ContextualRelevancyMetric(threshold=0.5, model=judge_llm),
    ]

    # 6. Run evaluation (sequential: a local Ollama model can't handle
    #    DeepEval's default 20 concurrent requests)
    evaluation_result = evaluate(
        test_cases=test_cases,
        metrics=metrics,
        async_config=AsyncConfig(run_async=False),
        display_config=DisplayConfig(print_results=False, show_indicator=True),
        error_config=ErrorConfig(ignore_errors=True),
    )

    # 7. Collect per-case scores from the returned results
    per_case_rows: list[dict[str, Any]] = []
    for tc, test_result in zip(test_cases, evaluation_result.test_results):
        row: dict[str, Any] = {"question": tc.input}
        for md in (test_result.metrics_data or []):
            name = md.name.lower().replace(" ", "_")
            row[name] = md.score
            row[f"{name}_reason"] = md.reason or md.error
        per_case_rows.append(row)

    scores_df = pd.DataFrame(per_case_rows)

    detailed_df = pd.concat(
        [raw_df.reset_index(drop=True), scores_df.drop(columns=["question"])
        .reset_index(drop=True)],
        axis=1,
    )
    detailed_file = output_path / "deepeval_detailed_results.csv"
    detailed_df.to_csv(detailed_file, index=False, encoding="utf-8-sig")

    # 8. Aggregate summary
    metric_names = [
        "faithfulness",
        "answer_relevancy",
        "contextual_precision",
        "contextual_recall",
        "contextual_relevancy",
    ]
    summary: dict[str, Any] = {}

    for metric in metric_names:
        if metric in scores_df.columns:
            values = pd.to_numeric(scores_df[metric], errors="coerce")
            summary[metric] = {
                "mean": float(values.mean()) if values.notna().any() else None,
                "valid_samples": int(values.notna().sum()),
                "failed_samples": int(values.isna().sum()),
            }

    report = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "total_questions": len(questions),
        "top_k": top_k,
        "retrieval_tools": (
            sorted(retrieval_tool_names) if retrieval_tool_names else "all"
        ),
        "metrics": summary,
    }

    summary_file = output_path / "deepeval_summary.json"
    with summary_file.open("w", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2, allow_nan=False)

    print("\n" + "=" * 60)
    print("DEEPEVAL EVALUATION COMPLETED")
    print("=" * 60)
    for metric, values in summary.items():
        print(
            f"{metric:25s}: {values['mean']} "
            f"(valid={values['valid_samples']}, "
            f"failed={values['failed_samples']})"
        )
    print(f"\nRaw outputs: {raw_file}")
    print(f"Detailed scores: {detailed_file}")
    print(f"Summary: {summary_file}")

    return {
        "summary": report,
        "detailed_results": detailed_df,
        "raw_outputs": raw_df,
        "files": {
            "raw": str(raw_file),
            "detailed": str(detailed_file),
            "summary": str(summary_file),
        },
    }