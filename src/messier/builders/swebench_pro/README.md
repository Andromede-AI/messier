# SWE-bench Pro Description

[SWE-bench Pro](https://huggingface.co/datasets/ScaleAI/SWE-bench_Pro) evaluates whether agents can resolve software issues in large repositories. It contains 731 tasks from 11 repositories written in Python, JavaScript, TypeScript, and Go. Each task provides an issue description and a repository checkout at a fixed commit. The agent inspects and edits the repository, runs tests, and produces a patch.

The SWE-bench Pro test harness applies the patch and runs task-specific tests. A task succeeds when the tests exposing the issue pass and previously passing tests continue to pass. The harness produces one binary result, which is used directly as the trial result. Individual test results are also retained when provided by the source.

| | SWE-bench Pro |
| --- | --- |
| Task and environment | A software issue and a repository checkout at a fixed commit |
| Environment state | Repository files and installed project dependencies |
| Action space | Shell commands for inspecting and editing files, running tests, and producing a patch |
| Verifier | A script checks that the issue is resolved without breaking previously passing tests |
| Scoring rule | The binary output from the verifier is used directly |
| Gold answer | The patch written by the developer who resolved the original issue |

The following example contains fields from one imported task and result. The issue, requirements, test results, and patches are shortened here.

```python
Task(
    benchmark="swebench_pro",
    task_id="NodeBB__NodeBB-04998908ba6721d64eba79ae3b65a351dcfbc5b5",
    environment_id="NodeBB/NodeBB",
    environment_description=(
        "A checkout of NodeBB/NodeBB. "
        "Node.js based forum software built for the modern web."
    ),
    environment_metadata={
        "action_space": {"types": ["shell"]},
        "environment_state": {
            "type": "filesystem",
            "access": "read_write",
            "ref": "github://NodeBB/NodeBB@1e137b0...",
        },
        "repo_language": "js",
    },
    task_description=(
        "Email Validation Status Not Handled Correctly in ACP and "
        "Confirmation Logic ..."
    ),
    task_metadata={
        "requirements": (
            "The loadUserInfo(callerUid, uids) function should include logic "
            "to retrieve and attach email:pending and email:expired flags ..."
        ),
        "test_patch": "diff --git a/test/database/keys.js ...",
    },
    gold_answer="diff --git a/public/language/en-GB/admin/manage/users.json ...",
    task_date="2025-08-26",
    verifiers=[
        {
            "type": "script",
            "mode": "final",
            "name": "swebench_harness",
        }
    ],
    scoring_rule="direct",
    source_platform="huggingface",
)

Record(
    benchmark="swebench_pro",
    task_id="NodeBB__NodeBB-04998908ba6721d64eba79ae3b65a351dcfbc5b5",
    agent_id="claude-sonnet-4-5+ii-agent",
    agent_model="claude-sonnet-4-5",
    agent_scaffold="ii-agent",
    model_date="2025-09-29",
    trial=0,
    result=1,
    metadata={
        "raw_source": "ii_agent_claude_sonnet_4_5",
        "unit_results": {
            "Key methods should get a key without error": 1,
            "Key methods should return null if a key does not exist": 1,
            "...": 1,
        },
    },
    source_platform="huggingface",
)
```

### SWE-bench Pro (`NodeBB__NodeBB-04998908ba6721d64eba79ae3b65a351dcfbc5b5`) example

**Agent** = **Model:** `claude-sonnet-4-5` + **Scaffold:** `ii-agent`

| | Interaction |
| --- | --- |
| 👤 **User** | Email Validation Status Not Handled Correctly in ACP and Confirmation Logic |
| 🤖 **Agent** | Inspects the repository and its database and user modules. |
| 🌐 **Env** | Returns the files under `/app/src/database/` and `/app/src/user/`. |
| 🤖 **Agent** | Adds `db.mget()` to the Redis, MongoDB, and PostgreSQL adapters and updates the email-validation logic. |
| 🤖 **Agent** | Runs `npm test -- test/email-validation-fix.js test/user/emails.js`. |
| 🌐 **Env** | Returns `34 passing (6s)`. |
| 🟥 **Trial result** | `1`, because the submitted patch passed the final benchmark harness. |

## References

- Original tasks: [ScaleAI/SWE-bench_Pro](https://huggingface.co/datasets/ScaleAI/SWE-bench_Pro)
- Imported results: [Agent Psychometrics](https://github.com/dariakryvosheieva/agent-psychometrics)
- Additional trajectories: [Claude Sonnet 4.5 with ii-agent](https://huggingface.co/datasets/Intelligent-Internet/swebench-pro-claude-sonnet-4.5-ii-agent-trajectories)
- Additional trajectories: [GPT-5 Codex with ii-agent](https://huggingface.co/datasets/Intelligent-Internet/swebench-pro-gpt-5-codex-ii-agent-trajectories)
- Additional results: [Claude Sonnet 4.5 with LiveSWE-agent](https://huggingface.co/datasets/livesweagent/claude-sonnet-4-5_swebench_pro_traj)
