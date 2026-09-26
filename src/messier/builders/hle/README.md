# Humanity's Last Exam Description

[Humanity's Last Exam (HLE)](https://huggingface.co/datasets/cais/hle) evaluates models on expert-written questions across mathematics, science, engineering, medicine, the humanities, and other fields. Questions require either a short answer or a multiple-choice answer, and some include an image.

We include per-model judgments for 1,369 HLE task identifiers released by [Sup AI](https://github.com/supaihq/hle). These runs used the Sup AI scaffold with web search. The source also reports an ensemble answer named `main`, which MESSIER omits because it is not an individual model result. To protect the benchmark's integrity and respect the authors' intended use, the release preserves task identifiers and derived results without copying its questions, answers, rationales, or images. Users with access to the gated dataset can join these fields using `task_id`.

| | HLE |
| --- | --- |
| Task and environment | An expert-written question spanning mathematics, science, engineering, medicine, the humanities, and other specialist fields, sometimes with an image and access to live web search |
| Environment state | Live search results. Search calls do not change the web. |
| Action space | Web-search calls and text responses |
| Verifier | An LLM Judge (GPT-5.1) compares the final answer with the gold answer, accounting for equivalent wording and numerical tolerance |
| Scoring rule | The binary output from the verifier is used directly |
| Gold answer | The expected answer available from the gated HLE dataset and used by the source judge |

The following example contains fields from one imported task identifier and result.

```python
Task(
    benchmark="hle",
    task_id="66ee01a4126fac9cef29cb8b",
    environment_id="hle",
    environment_description=(
        "An expert-written question spanning mathematics, science, engineering, "
        "medicine, the humanities, and other specialist fields, sometimes with "
        "an image and access to live web search."
    ),
    environment_metadata={
        "action_space": {"types": ["tool_calls", "text"]},
        "environment_state": {"type": "live_web", "access": "read_only"},
    },
    task_description=None,
    task_metadata={},
    gold_answer=None,
    verifiers=[
        {
            "type": "llm_judge",
            "mode": "final",
            "judge_model": "openai/gpt-5.1",
        }
    ],
    scoring_rule="direct",
    source_platform="huggingface",
)

Record(
    benchmark="hle",
    task_id="66ee01a4126fac9cef29cb8b",
    agent_id="claude-opus-4-5+sup-ai",
    agent_model="claude-opus-4-5",
    agent_scaffold="sup-ai",
    model_date="2025-11-24",
    trial=0,
    result=1,
    metadata={"confidence": 95},
    source_platform="github",
)
```

## References

- Imported results: [Sup AI HLE evaluations](https://github.com/supaihq/hle)
- Original benchmark: [Humanity's Last Exam](https://huggingface.co/datasets/cais/hle) and its [paper](https://arxiv.org/abs/2501.14249)
