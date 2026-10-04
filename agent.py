"""Final assignment research agent.

The agent answers questions only when they can be grounded in the supplied
course corpus. Unsupported questions are refused. Retrieved documents are
treated as untrusted data, never as instructions.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from pathlib import Path

from bootcamp_agent.agent import AgentResult, answer_question
from bootcamp_agent.config import load_settings
from bootcamp_agent.documents import Document, load_corpus
from bootcamp_agent.llm import LLMClient, get_client
from bootcamp_agent.schema import ResearchAnswer
from bootcamp_agent.tools import Tool, build_tools


CORPUS_DIR = Path(__file__).resolve().parent / "data" / "corpus"


def _safe_refusal() -> ResearchAnswer:
    """Return a conservative answer when the agent cannot safely answer."""
    return ResearchAnswer(
        answer=(
            "I don't have enough supported information in the provided "
            "documents to answer that reliably."
        ),
        citations=(),
        confidence=0.0,
        needs_human_review=True,
    )


class YourAgent:
    """Grounded, bounded research agent used by the tests and grader."""

    timeout_s: float = 30.0

    def __init__(self, client: LLMClient | None = None) -> None:
        self.documents: list[Document] = load_corpus(CORPUS_DIR)

        self.client: LLMClient = (
            client
            if client is not None
            else get_client(load_settings())
        )

        # Keep the course tool registry available, but do not allow arbitrary
        # tools to become part of the answering path.
        self.tools: dict[str, Tool] = build_tools(
            self.documents,
            self.client,
        )

    def _run_pipeline(self, question: str) -> AgentResult:
        """Run the standard grounded course pipeline."""
        return answer_question(
            question=question,
            documents=self.documents,
            client=self.client,
            max_tool_calls=3,
            top_k=5,
        )

    def run(self, question: str) -> AgentResult:
        """Answer one question with a strict execution timeout."""

        question = question.strip()

        if not question:
            return AgentResult(
                answer=_safe_refusal(),
                trace=(),
            )

        executor = ThreadPoolExecutor(max_workers=1)

        future = executor.submit(
            self._run_pipeline,
            question,
        )

        try:
            result = future.result(timeout=self.timeout_s)

        except FutureTimeoutError:
            future.cancel()

            return AgentResult(
                answer=_safe_refusal(),
                trace=(),
            )

        except Exception:
            return AgentResult(
                answer=_safe_refusal(),
                trace=(),
            )

        finally:
            # Do not wait for a timed-out provider call.
            executor.shutdown(wait=False, cancel_futures=True)

        answer = result.answer

        # No citation means no grounded factual answer.
        # A refusal is therefore kept visibly conservative.
        if not answer.citations:
            safe_answer = ResearchAnswer(
                answer=answer.answer,
                citations=(),
                confidence=min(answer.confidence, 0.2),
                needs_human_review=True,
            )

            return AgentResult(
                answer=safe_answer,
                trace=result.trace,
            )

        # Answers with citations have already had fabricated citation IDs
        # stripped by answer_question(). Preserve that result.
        return result

    def __call__(self, question: str) -> ResearchAnswer:
        return self.run(question).answer