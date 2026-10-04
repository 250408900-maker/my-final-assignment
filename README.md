# my-final-assignment

An evidence-grounded research agent that answers from a six-document local corpus and flags a refusal when the corpus cannot support an answer.

![check](https://github.com/250408900-maker/my-final-assignment/actions/workflows/check.yml/badge.svg)

## The problem

Developers building retrieval-augmented assistants need answers they can trace to documents rather than unsupported model memory. A model can confidently invent facts, cite a source it did not receive, or follow instructions embedded in retrieved text. This project demonstrates a bounded pipeline that retrieves local material, validates structured output and citations, and has an explicit review-flagged refusal path.

## Demo

The outputs below are pasted exactly as the commands printed them.

### One supported answer

```bash
uv run bootcamp capstone trace "How does chunking work in RAG?"
```

```text
answer: I do not know based on the provided corpus.
citations: []
confidence: 0.0
needs_human_review: True
```

### One refusal

```bash
uv run bootcamp capstone trace "What is the capital city of Mongolia?"
```

```text
[retrieve] top_k=20 -> []
[decision] no relevant chunks; refusing without an LLM call

answer: I do not know based on the provided corpus.
citations: []
confidence: 0.0
needs_human_review: True
```

## Architecture

`YourAgent.run` uses a bounded chain, not an autonomous tool loop. It cleans instruction-style text from the question, then calls the course pipeline, which retrieves up to 20 lexical chunks before deciding whether to call the model. No retrieved chunks means an immediate refusal with no model call; otherwise the normal path makes one model call, with one corrective retry only if the response cannot be parsed. The pipeline normalizes citations and removes citations whose document IDs were not retrieved, flagging that result for review. If the pipeline answer has no citations, the agent builds a verbatim extractive answer from relevant retrieved paragraphs and cites their document IDs; if it cannot find enough evidence, it returns the review-flagged refusal. Provider errors and timeouts also return the safe refusal.

See [docs/adr/0001-run-shape.md](docs/adr/0001-run-shape.md).

## Measured results

These results are from the commands below in the current worktree. The configured practice-grader lane was OpenAI; the trace and test outputs are also included as observed, not as an offline-lane claim.

| What | Command | Model | Result |
|---|---|---|---|
| Contract tests | `uv run pytest` | Fake test clients | `6 passed, 2 skipped, 1 xfailed in 0.37s` |
| Practice grader | `uv run bootcamp final grade` | OpenAI lane | `score: 3/10 (30%)`; pass bar not met; critical safety gate failed |

## The honest limitation

The current pipeline verifies that cited document IDs were retrieved, but a cited model-generated answer is returned without independently checking every claim against its cited passages; the next step is claim-to-evidence validation over all retrieved chunks, with a regression test for answers that need multiple sources. The ranked issue list and before/after evaluation report still contain unfilled placeholders, so they do not provide a measured ranking or historical comparison.

The full ranked list is in [docs/ISSUES.md](docs/ISSUES.md).

## How to run it

```bash
git clone https://github.com/250408900-maker/my-final-assignment.git
cd my-final-assignment
uv sync
uv run pytest
```

No key needed: without a `.env` it runs on the offline fake model. For a real
model, copy `.env.example` to `.env`, fill in your provider, and
`uv sync --extra anthropic` (or `--extra openai`).

Run the practice grader with `uv run bootcamp final grade`.

To hand in the final assignment, commit and push, then run
`uv run bootcamp capstone submit --github 250408900-maker`. It runs the practice
set first, then answers the final questions and opens the pull request.
`--dry-run` shows the bundle without handing anything in.

---

| Path | What it is |
|---|---|
| `agent.py` | The agent: `YourAgent`, the class the tests, `trace` and the grader run |
| `tests/test_contract.py` | The capstone contract, as tests (`uv run pytest -k refusal`, `-k injection`, ...) |
| `data/corpus/` | The six source documents, versioned; nothing here writes to them |
| `docs/EVAL_REPORT.md` | Numbers you produced, before and after, with the command behind each |
| `docs/SKILL.md` | A skill another assistant can load (session 10) |
| `docs/adr/0001-run-shape.md` | The architecture decision and what would reverse it (session 10) |
| `docs/RETENTION.md` | What a session remembers, and what it refuses to (session 11) |
| `docs/ISSUES.md` | The ranked issue list (session 9, kept until 14) |

Built during the Dev3Pack AI Engineering bootcamp, on the course package at
commit `3262a668fbfbe085d0c91394c1de882cf1f5bd58` of [Gecko-Academy/dev3pack-cohort-2026-09](https://github.com/Gecko-Academy/dev3pack-cohort-2026-09).

Project repository: [250408900-maker/my-final-assignment](https://github.com/250408900-maker/my-final-assignment).
