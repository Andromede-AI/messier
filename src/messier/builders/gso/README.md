# GSO Description

[GSO](https://github.com/gso-bench/gso) evaluates coding agents on software performance optimization tasks. Each task provides a repository and a performance test, and asks the agent to improve runtime without changing the program's behavior. We include 102 GSO tasks with 1,530 pass/fail results released through [Agent Psychometrics](https://github.com/dariakryvosheieva/agent-psychometrics). All imported runs pair a model with the OpenHands scaffold.

| | GSO |
| --- | --- |
| Task and environment | A repository at a fixed revision and a performance test for a target API |
| Environment state | Files and directories containing the repository |
| Action space | Shell commands and file edits |
| Verifier | Python test scripts check the final state for correct behavior and runtime improvement |
| Scoring rule | Threshold. Success requires correct behavior and at least 95% of the runtime improvement achieved by the expert developer's patch (i.e. the gold answer) |
| Gold answer | The expert developer's optimization patch (used as a reference in the Verifier) |

The following example shows selected fields from one imported task and result. The task description, performance test, and patch are shortened here.

```python
Task(
    benchmark="gso",
    task_id="numpy__numpy-248c60e",
    environment_id="numpy/numpy",
    task_description="Optimize the runtime of the supplied numpy.char.isdecimal performance test. ...",
    environment_metadata={
        "action_space": {"types": ["shell"]},
        "environment_state": {
            "type": "filesystem",
            "access": "read_write",
            "ref": "github://numpy/numpy@248c60e^",
        },
    },
    task_metadata={
        "api": "numpy.char.isdecimal",
        "prob_script": "import numpy as np\n...",
    },
    gold_answer="diff --git a/numpy/_core/code_generators/generate_umath.py ...",
    verifiers=[
        {
            "type": "script",
            "mode": "final",
            "metric": "fraction_of_expert_runtime_improvement",
            "threshold": 0.95,
        }
    ],
    scoring_rule="threshold",
    source_platform="huggingface",
)

Record(
    benchmark="gso",
    task_id="numpy__numpy-248c60e",
    agent_id="claude-opus-4+openhands",
    agent_model="claude-opus-4",
    agent_scaffold="openhands",
    trial=0,
    result=1,
    source_platform="github",
    data_provider="agent_psychometrics",
)
```

### GSO (`numpy__numpy-248c60e`) example

**Agent** = **Model:** `claude-opus-4` + **Scaffold:** `openhands`

| | Interaction |
| --- | --- |
| 👤 **User** | Optimize the supplied `numpy.char.isdecimal` workload while preserving its behavior. |
| 🤖 **Agent** | Runs `cd /workspace/numpy__numpy && pwd && ls -la` to inspect the repository. |
| 🌐 **Env** | Returns the repository path and directory contents. |
| 🤖 **Agent** | Creates and runs `/workspace/test_opt.py` to measure the original implementation. |
| 🌐 **Env** | Reports an average runtime of `0.0975` seconds. |
| 🤖 **Agent** | Edits `numpy/_core/src/multiarray/multiarraymodule.c`, rebuilds NumPy, and reruns the tests. |
| 🌐 **Env** | Reports an average runtime of `0.0078` seconds and passing correctness checks. |
| 🟥 **Trial result** | `1`, because the patch passes the correctness checks and reaches the required runtime-improvement threshold. |

## References

- Imported results: [Agent Psychometrics](https://github.com/dariakryvosheieva/agent-psychometrics)
- Original benchmark: [GSO](https://github.com/gso-bench/gso)
