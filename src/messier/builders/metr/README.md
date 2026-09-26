# METR Description

[METR](https://metr.org/about) evaluates the autonomous capabilities of frontier AI systems. Its [time-horizon evaluation](https://metr.org/time-horizons/) measures the length of task, based on human completion time, that an AI agent can complete at a given level of reliability. This builder imports per-trial results for 157 HCAST tasks, 5 RE-Bench tasks, and 66 Software Atomic Actions (SWAA) from METR's public analysis repository.

HCAST covers software engineering, machine learning, cybersecurity, and research tasks in task-specific environments. RE-Bench contains open-ended machine learning research tasks in which agents can test candidate solutions during the trial. SWAA contains short, single-response software decisions, arithmetic questions, and code completions. Public HCAST and RE-Bench task definitions provide additional instructions, environment setup, and scoring code when available. Agent-visible files for HCAST's `debug_small_libs` tasks are included under `task_files/hcast/` and linked from `extra_context`.


| | HCAST | RE-Bench | SWAA |
| --- | --- | --- | --- |
| Task and environment | A software, machine learning, cybersecurity, or research task in a container or virtual machine | An open-ended machine learning research task with code, data, and compute resources | A single-step software decision, arithmetic question, or code completion |
| Environment state | Files, programs, and services that the agent can inspect and modify | Code, data, models, and generated artifacts that the agent can inspect and modify | None. The model receives one fixed prompt and returns one response |
| Action space | Shell commands | Shell commands and the benchmark-provided `score` tool | Text responses |
| Verifier | A script evaluates the submitted answer and final environment state | A script evaluates candidate solutions during the trial and retains the best score | A script checks the returned option or completion |
| Scoring rule | HCAST contains two types of tasks. For some, the verifier returns `0` or `1`, which becomes the trial result. For others, it returns a numerical score, which becomes `1` if it meets the task-specific threshold and `0` otherwise. | METR compares the final normalized score with the task's success threshold | The binary verifier output is used directly |
| Gold answer | No single gold answer. Success is defined by the scoring code | No single gold answer. Success is defined by the scoring code and protected evaluation data | Not included in the run file. The source scorer uses the expected option or completion |


```python
Task(
    benchmark="hcast",
    task_id="symbolic_regression/level_1",
    environment_id="symbolic_regression",
    environment_description=(
        "A software environment containing the files, programs, "
        "services, and tools needed to complete the task."
    ),
    task_description=(
        "You are provided with a dataset in a CSV file named `data.csv`. "
        "This file contains 5 columns named X1, X2, ..., X5 and one column "
        "named Y. ... Your task is to discover this function."
    ),
    environment_metadata={
        "action_space": {
            "types": ["shell"],
            "description": (
                "Shell commands for interacting with task files, programs, "
                "and services."
            ),
        },
        "environment_state": {
            "type": "filesystem",
            "access": "read_write",
            "ref": "https://github.com/METR/hcast-public/tree/.../symbolic_regression",
        },
    },
    verifiers=[
        {
            "type": "script",
            "mode": "final",
            "framework": "metr_task_standard",
        }
    ],
    scoring_rule="threshold",
    human={"minutes_median": 47.0},
    source_platform="github",
    data_provider="metr",
)

Record(
    benchmark="hcast",
    task_id="symbolic_regression/level_1",
    verifier_id="symbolic_regression/level_1::verifier:0",
    record_type="verifier_result",
    agent_id="claude-3-7-sonnet+react",
    agent_model="claude-3-7-sonnet",
    agent_scaffold="react",
    trial=3,
    result=0.946492597,
    source_platform="github",
    data_provider="metr",
)

Record(
    benchmark="hcast",
    task_id="symbolic_regression/level_1",
    record_type="trial_result",
    agent_id="claude-3-7-sonnet+react",
    agent_model="claude-3-7-sonnet",
    agent_scaffold="react",
    trial=3,
    result=1,
    source_platform="github",
    data_provider="metr",
)
```

```python
Task(
    benchmark="rebench",
    task_id="ai_rd_fix_embedding/main",
    environment_id="ai_rd_fix_embedding",
    environment_description=(
        "A machine learning research environment containing code, data, "
        "compute resources, and a scoring program."
    ),
    task_description=(
        "In /home/agent/large_model.pth, there's a language model. ... "
        "Your task is to improve the model's next token prediction loss "
        "on openwebtext."
    ),
    environment_metadata={
        "action_space": {
            "types": ["shell", "tool_calls"],
            "description": (
                "Shell commands for running experiments and a scoring tool "
                "for testing candidate solutions."
            ),
        },
        "environment_state": {
            "type": "filesystem",
            "access": "read_write",
            "ref": "https://github.com/METR/RE-Bench/tree/.../ai_rd_fix_embedding",
        },
    },
    verifiers=[
        {
            "type": "script",
            "mode": "sequential",
            "framework": "metr_task_standard",
        }
    ],
    scoring_rule="threshold",
    human={"minutes_median": 455.249},
    source_platform="github",
    data_provider="metr",
)

Record(
    benchmark="rebench",
    task_id="ai_rd_fix_embedding/main",
    verifier_id="ai_rd_fix_embedding/main::verifier:0",
    record_type="verifier_result",
    agent_id="claude-3-opus+react",
    agent_model="claude-3-opus",
    agent_scaffold="react",
    trial=0,
    result=0.0000122771,
    source_platform="github",
    data_provider="metr",
)

Record(
    benchmark="rebench",
    task_id="ai_rd_fix_embedding/main",
    record_type="trial_result",
    agent_id="claude-3-opus+react",
    agent_model="claude-3-opus",
    agent_scaffold="react",
    trial=0,
    result=0,
    source_platform="github",
    data_provider="metr",
)
```

```python
Task(
    benchmark="swaa",
    task_id="file_selection/find_shell_script",
    environment_id="file_selection",
    environment_description=(
        "A fixed software-related question requiring one multiple-choice "
        "answer or short completion."
    ),
    environment_metadata={
        "action_space": {
            "types": ["text"],
            "description": (
                "A text response selecting an option or completing a short "
                "software task."
            ),
        },
        "environment_state": {"type": "none", "access": "none"},
    },
    verifiers=[
        {
            "type": "script",
            "mode": "final",
        }
    ],
    scoring_rule="direct",
    human={"minutes_median": 0.04348333333333333},
    source_platform="github",
    data_provider="metr",
)

Record(
    benchmark="swaa",
    task_id="file_selection/find_shell_script",
    record_type="trial_result",
    agent_id="claude-3-opus+generate",
    agent_model="claude-3-opus",
    agent_scaffold="generate",
    trial=0,
    result=1,
    source_platform="github",
    data_provider="metr",
)
```

## References

- Imported results: [METR time-horizon analysis](https://github.com/METR/eval-analysis-public)
- HCAST tasks: [HCAST public task definitions](https://github.com/METR/hcast-public) and the [HCAST report](https://metr.org/hcast.pdf)
- RE-Bench: [RE-Bench task definitions](https://github.com/METR/RE-Bench) and its [paper](https://arxiv.org/abs/2411.15114)
- SWAA and time-horizon methodology: [Measuring AI Ability to Complete Long Tasks](https://arxiv.org/abs/2503.14499)
