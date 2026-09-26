# LiveCodeBench Description

[LiveCodeBench](https://github.com/LiveCodeBench/LiveCodeBench) evaluates several coding capabilities using competitive-programming problems from LeetCode, Codeforces, and AtCoder. Here, we import only its code-generation scenario, comprising 1,055 tasks and 29,540 per-model pass@1 values from the public leaderboard. The model receives a fixed problem and returns a Python solution without execution feedback. An automated judge evaluates the solution afterward. The leaderboard reports pass@1 for each model and task rather than individual generated solutions, so these values remain source summaries and may be fractional when calculated from multiple generations (but we don't have that info).

| | LiveCodeBench |
| --- | --- |
| Task and environment | A competitive-programming problem from LeetCode, Codeforces, or AtCoder |
| Environment state | None. The model receives no execution feedback while generating its solution |
| Action space | Text responses |
| Verifier | An automated online judge runs the submitted solution against hidden tests |
| Scoring rule | The binary output from the verifier is used directly |
| Gold answer | Test cases containing program inputs and their expected outputs |

The following example contains fields from one task and source summary. The task description is shortened here.

```python
Task(
    benchmark="livecodebench",
    task_id="1883_B",
    environment_id="codeforces",
    task_description=(
        "You are given a string s and an integer k. Determine whether removing "
        "exactly k characters can leave a string that can form a palindrome. ..."
    ),
    environment_metadata={
        "action_space": {
            "types": ["text"],
            "description": "A Python solution submitted as text.",
        },
        "environment_state": {"type": "none", "access": "none"},
        "platform": "codeforces",
    },
    task_metadata={
        "question_title": "B. Chemistry",
        "extra_context": None,
        "n_public_tests": 1,
        "contest_id": "1883",
        "contest_date": "2023-09-22T00:00:00",
    },
    task_date="2023-09-22",
    verifiers=[{"type": "script", "mode": "final", "name": "online_judge"}],
    difficulty_label="medium",
    scoring_rule="direct",
    source_platform="huggingface",
)

Record(
    benchmark="livecodebench",
    task_id="1883_B",
    record_type="source_summary",
    agent_id="deepseek-v3",
    agent_model="deepseek-v3",
    agent_scaffold=None,
    model_date="2024-12-24",
    trial=None,
    result=0.6,
    metadata={"summary": "pass@1", "raw_model": "DeepSeek-V3"},
    source_platform="github",
)
```

## References

- Imported results: [LiveCodeBench code-generation dataset](https://huggingface.co/datasets/livecodebench/code_generation_lite)
- Original benchmark: [LiveCodeBench](https://github.com/LiveCodeBench/LiveCodeBench)
