# SWE-bench Description

[SWE-bench](https://www.swebench.com/) evaluates whether agents can resolve real GitHub issues in Python repositories. Each task provides an issue description and a repository checkout at a fixed commit. The agent inspects and edits the repository, runs tests, and produces a patch.

The SWE-bench test harness applies the patch and runs qa tests. A task succeeds when the tests exposing the issue pass and previously passing tests continue to pass. The harness produces one binary result, which is used directly as the trial result.

| | SWE-bench |
| --- | --- |
| Task and environment | A GitHub issue and a repository checkout at a fixed commit point |
| Environment state | Repository files and installed project dependencies |
| Action space | Shell commands for inspecting and editing files, running tests, and producing a patch |
| Verifier | A script checks that the issue is resolved without breaking previously passing tests |
| Scoring rule | The binary output from the verifier is used directly |
| Gold answer | The patch written by the developer who resolved the original issue |

The following example contains fields from one imported task and result. The issue, test patch, and gold patch are shortened here.

```python
Task(
    benchmark="swebench",
    task_id="django__django-10730",
    environment_id="django/django",
    environment_description=(
        "A checkout of django/django. "
        "The Web framework for perfectionists with deadlines"
    ),
    environment_metadata={
        "action_space": {"types": ["shell"]},
        "environment_state": {
            "type": "filesystem",
            "access": "read_write",
            "ref": "github://django/django@199025f...",
        },
    },
    task_description=(
        "Infinite loop in ExceptionReporter.get_traceback_frames()\n"
        "The following code generates a cause/context cycle ..."
    ),
    task_metadata={
        "split": "train",
        "test_patch": "diff --git a/tests/view_tests/tests/test_debug.py ...",
    },
    gold_answer="diff --git a/django/views/debug.py ...",
    task_date="2018-12-06",
    verifiers=[
        {
            "type": "script",
            "mode": "final",
            "name": "swebench_harness",
        }
    ],
    scoring_rule="direct",
    source_platform="github",
)

Record(
    benchmark="swebench",
    task_id="django__django-10730",
    agent_id="claude-2+rag",
    agent_model="claude-2",
    agent_scaffold="rag",
    model_date="2023-07-11",
    trial=0,
    result=1,
    metadata={
        "run_id": "public_swebench:test:20231010_rag_claude2:django__django-10730:attempt0",
        "source_reward": 1.0,
        "error_type": "success",
    },
    source_platform="github",
)
```

## References

- Imported tasks and results: [SWE-bench](https://github.com/SWE-bench/SWE-bench)
- Public evaluation results: [SWE-bench Experiments](https://github.com/swe-bench/experiments)
- Original paper: [SWE-bench: Can Language Models Resolve Real-World GitHub Issues?](https://arxiv.org/abs/2310.06770)
