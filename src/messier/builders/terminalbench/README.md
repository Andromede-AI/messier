# TerminalBench Description

[TerminalBench](https://github.com/laude-institute/terminal-bench) evaluates agents on tasks performed through a command-line interface. Each task provides an instruction and a Linux environment containing the files, software, and services needed to complete it. Tasks include software engineering, system administration, security, scientific computing, and machine learning.

The agent works through shell commands. After the trial, a task-specific test script examines the resulting files, program behavior, or service state and returns a binary result. MESSIER imports task definitions and results from [Agent Psychometrics](https://github.com/dariakryvosheieva/agent-psychometrics), together with additional public results and trajectories from [TerminalBench Trajectories](https://huggingface.co/datasets/yoonholee/terminalbench-trajectories).

| | TerminalBench |
| --- | --- |
| Task and environment | An instruction and a Linux environment containing the files, software, and services needed to complete it |
| Environment state | Files, directories, programs, and services that the agent can inspect and modify |
| Action space | Shell commands executed in the Linux environment |
| Verifier | A test script checks the final files, program behavior, or service state |
| Scoring rule | The binary output from the verifier is used directly |
| Gold answer | The benchmark's reference solution |

The following example shows selected fields from one imported task and result. The task description, test script, and reference solution are shortened here.

```python
Task(
    benchmark="terminalbench",
    task_id="break-filter-js-from-html",
    environment_id="break-filter-js-from-html",
    environment_description=(
        "A Linux environment containing files, software, and services that "
        "can be inspected and modified through shell commands."
    ),
    task_description=(
        "Create /app/out.html so that JavaScript still runs after the file "
        "is processed by /app/filter.py. ..."
    ),
    environment_metadata={
        "action_space": {
            "types": ["shell"],
            "description": "Shell commands executed in the Linux environment.",
        },
        "environment_state": {
            "type": "filesystem",
            "access": "read_write",
            "ref": "https://github.com/laude-institute/terminal-bench/tree/.../break-filter-js-from-html",
        },
    },
    task_metadata={
        "category": "security",
        "parser_name": "pytest",
    },
    gold_answer="#! /bin/bash\n...",
    verifiers=[
        {
            "type": "script",
            "mode": "final",
            "name": "terminalbench_harness",
            "script_code": "### tests/filter.py\n...",
        }
    ],
    scoring_rule="direct",
    source_platform="github",
    data_provider="agent_psychometrics",
)

Record(
    benchmark="terminalbench",
    task_id="break-filter-js-from-html",
    agent_id="claude-haiku-4-5+claude-code",
    agent_model="claude-haiku-4-5",
    agent_scaffold="claude-code",
    model_date="2025-10-15",
    trial=1,
    result=1,
    metadata={
        "raw_source": "yoonholee",
        "raw_model": "claude-haiku-4-5-20251001@anthropic",
    },
    source_platform="huggingface",
)
```

### TerminalBench (`break-filter-js-from-html`) example

**Agent** = **Model:** `claude-haiku-4-5` + **Scaffold:** `claude-code`

| | Interaction |
| --- | --- |
| 👤 **User** | Asks the agent to create `/app/out.html` so that it still triggers a JavaScript alert after `/app/filter.py` removes JavaScript. |
| 🤖 **Agent** | Reads `/app/filter.py` and `/app/test_outputs.py`, then writes `/app/out.html`. |
| 🌐 **Env** | Returns the contents of the created file. |
| 🤖 **Agent** | Runs the filter and the supplied test script against the file. |
| 🌐 **Env** | Reports that the alert was triggered successfully after filtering. |
| 🟥 **Trial result** | `1`, because the final test script passes. |

## References

- Imported task definitions and results: [Agent Psychometrics](https://github.com/dariakryvosheieva/agent-psychometrics)
- Imported results and trajectories: [TerminalBench Trajectories](https://huggingface.co/datasets/yoonholee/terminalbench-trajectories)
- Original benchmark: [TerminalBench](https://github.com/laude-institute/terminal-bench)
- Original paper: [Terminal-Bench: Benchmarking Agents on Hard, Realistic Tasks in Command Line Interfaces](https://arxiv.org/abs/2601.11868)
