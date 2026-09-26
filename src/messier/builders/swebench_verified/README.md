# SWE-bench Verified Description

[SWE-bench Verified](https://huggingface.co/datasets/princeton-nlp/SWE-bench_Verified) is a human-validated subset of 500 SWE-bench tasks. Each task provides a GitHub issue and a Python repository checkout at a fixed commit. The agent inspects and edits the repository, runs tests, and produces a patch.

The SWE-bench test harness applies the patch and runs task-specific tests. A task succeeds when the tests exposing the issue pass and previously passing tests continue to pass. The harness produces one binary result, which is used directly as the trial result. Individual test results are retained when provided by the source. Human completion-time estimates from BRIDGE are also attached to every task.

| | SWE-bench Verified |
| --- | --- |
| Task and environment | A GitHub issue and a repository checkout at a fixed commit |
| Environment state | Repository files and installed project dependencies |
| Action space | Shell commands for inspecting and editing files, running tests, and producing a patch |
| Verifier | A script checks that the issue is resolved without breaking previously passing tests |
| Scoring rule | The binary output from the verifier is used directly |
| Gold answer | The patch written by the developer who resolved the original issue |

The following example contains fields from one imported task and result. The issue, test results, and patches are shortened here.

```python
Task(
    benchmark="swebench_verified",
    task_id="astropy__astropy-12907",
    environment_id="astropy/astropy",
    environment_description=(
        "A checkout of astropy/astropy. "
        "Astronomy and astrophysics core library."
    ),
    environment_metadata={
        "action_space": {"types": ["shell"]},
        "environment_state": {
            "type": "filesystem",
            "access": "read_write",
            "ref": "github://astropy/astropy@d16bfe0...",
        },
    },
    task_description=(
        "Modeling's separability_matrix does not compute separability "
        "correctly for nested CompoundModels ..."
    ),
    task_metadata={
        "test_patch": (
            "diff --git a/astropy/modeling/tests/test_separable.py ..."
        ),
    },
    gold_answer="diff --git a/astropy/modeling/separable.py ...",
    task_date="2022-03-03",
    verifiers=[
        {
            "type": "script",
            "mode": "final",
            "name": "swebench_harness",
        }
    ],
    human={
        "minutes_median": 30.0,
        "minutes_low": 15.0,
        "minutes_high": 60.0,
    },
    difficulty_label="15 min - 1 hour",
    scoring_rule="direct",
    source_platform="huggingface",
)

Record(
    benchmark="swebench_verified",
    task_id="astropy__astropy-12907",
    agent_id="qwen-3-coder-32b+openhands",
    agent_model="qwen-3-coder-32b",
    agent_scaffold="openhands",
    model_date="2025-07-23",
    trial=0,
    result=1,
    metadata={
        "raw_source": "coderforge",
        "trajectory_id": "astropy__astropy-12907_run1",
        "num_steps": 38,
        "unit_results": {
            "astropy/modeling/tests/test_separable.py::test_cstack": 1,
            "astropy/modeling/tests/test_separable.py::test_arith_oper": 1,
            "...": 1,
        },
    },
    source_platform="huggingface",
)
```

### SWE-bench Verified (`astropy__astropy-12907`) example

**Agent** = **Model:** `qwen-3-coder-32b` + **Scaffold:** `openhands`

| | Interaction |
| --- | --- |
| 👤 **User** | Modeling's `separability_matrix` does not compute separability correctly for nested `CompoundModels`. |
| 🤖 **Agent** | Inspects `astropy/modeling/separable.py` and reproduces the incorrect matrix. |
| 🤖 **Agent** | Changes `_cstack` to copy the nested model's separability matrix instead of replacing it with ones. |
| 🤖 **Agent** | Runs `python -m pytest astropy/modeling/tests/test_separable.py -v`. |
| 🌐 **Env** | Returns `11 passed in 0.26s`. |
| 🟥 **Trial result** | `1`, because the submitted patch passed the final benchmark harness. |

## References

- Original tasks: [princeton-nlp/SWE-bench_Verified](https://huggingface.co/datasets/princeton-nlp/SWE-bench_Verified)
- Imported results: [Agent Psychometrics](https://github.com/dariakryvosheieva/agent-psychometrics)
- Human completion times: [BRIDGE](https://github.com/McGill-NLP/BRIDGE)
- Additional trajectories: [CoderForge](https://huggingface.co/datasets/togethercomputer/CoderForge-Preview-32B-SWE-Bench-Verified-Evaluation-trajectories)
- Additional results: [LiveSWE-agent](https://huggingface.co/livesweagent)
- Additional results: [Devin SWE-bench output](https://huggingface.co/datasets/OpenHandsCommunity/Devin-SWE-bench-output)
