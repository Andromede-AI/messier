# TheAgentCompany Description

[TheAgentCompany](https://github.com/TheAgentCompany/TheAgentCompany) evaluates agents on long-horizon workplace tasks in a software company environment. Its environment contains internal services for source code, project management, file sharing, and team communication. Tasks cover software development, project management, human resources, finance, administration, data science, quality assurance, research, machine learning, and business management.

Agents interact with the company services through a browser and shell. Evaluator scripts inspect files, service state, or recorded actions through one or more checkpoints. A trial succeeds when the task evaluator reports full credit. MESSIER preserves this binary trial result and stores available checkpoint results as values between `0` and `1`.

| | TheAgentCompany |
| --- | --- |
| Task and environment | A workplace instruction and company services containing files, accounts, messages, and project data |
| Environment state | Files, projects, messages, accounts, and other company data that the agent can inspect and modify |
| Action space | Browser interactions and shell commands |
| Verifier | Task-specific scripts inspect files, company service state, or recorded actions |
| Scoring rule | A trial succeeds when the task evaluator reports full credit |
| Gold answer | Checkpoint descriptions and expected output files used by the evaluator |


```python
Task(
    benchmark="theagentcompany",
    task_id="hr-mass-survey",
    environment_id="hr",
    environment_description=(
        "A company workspace containing internal services for source control, "
        "project management, file sharing, and team communication, together "
        "with the files and data used by the task."
    ),
    task_description=(
        "Conduct a comprehensive survey to collect information on employees' "
        "year-end vacation plans. ..."
    ),
    environment_metadata={
        "action_space": {
            "types": ["shell", "ui"],
        },
        "environment_state": {
            "type": "filesystem",
            "access": "read_write",
            "ref": "github://TheAgentCompany/TheAgentCompany@.../hr-mass-survey",
        },
        "dependencies": "- owncloud\n- rocketchat",
    },
    gold_answer={
        "rubric": [
            {
                "step": 1,
                "points": 5,
                "description": "Check the survey columns, rows, and total budget.",
            },
            {
                "step": 2,
                "points": 2,
                "description": "Confirm that all 17 employees were contacted.",
            },
        ],
        "files": {"solution.xlsx": None},
    },
    verifiers=[
        {"type": "script", "mode": "final", "name": "checkpoint", "points": 5},
        {"type": "script", "mode": "final", "name": "checkpoint", "points": 2},
    ],
    scoring_rule="all_pass",
    source_platform="github",
)

Record(
    benchmark="theagentcompany",
    task_id="hr-mass-survey",
    agent_id="gemini-1-5-pro+openhands",
    agent_model="gemini-1-5-pro",
    agent_scaffold="openhands",
    trial=0,
    result=0,
    metadata={
        "source_result": 1,
        "source_total": 7,
        "source_fraction": 0.14285714285714285,
    },
    source_platform="github",
)

Record(
    benchmark="theagentcompany",
    task_id="hr-mass-survey",
    verifier_id="hr-mass-survey::checkpoint_1",
    record_type="verifier_result",
    agent_id="gemini-1-5-pro+openhands",
    agent_model="gemini-1-5-pro",
    agent_scaffold="openhands",
    trial=0,
    result=0.5,
    metadata={
        "source_result": 1,
        "source_total": 2,
        "source_fraction": 0.5,
    },
    source_platform="github",
)
```

## References

- Imported tasks: [TheAgentCompany](https://github.com/TheAgentCompany/TheAgentCompany)
- Imported results: [TheAgentCompany experiments](https://github.com/TheAgentCompany/experiments)
- Original paper: [TheAgentCompany: Benchmarking LLM Agents on Consequential Real World Tasks](https://arxiv.org/abs/2412.14161)
