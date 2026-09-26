# Toolathlon Description

[Toolathlon](https://github.com/hkust-nlp/Toolathlon) evaluates language agents on 108 workflows across 32 software applications and 604 tools. Tasks often coordinate several applications, begin with initialized application data, and require many tool calls before a task-specific script checks the final state. Its scaffold is a modified agent loop built on the [OpenAI Agents SDK](https://github.com/openai/openai-agents-python). MESSIER imports 7,116 attempts from 22 model configurations in the [Toolathlon trajectory release](https://huggingface.co/datasets/hkust-nlp/Toolathlon-Trajectories).

Initial workspace files are included under `task_files/toolathlon/` and linked from each task's `extra_context` with their original workspace paths.

| | Toolathlon |
| --- | --- |
| Task and environment | A workflow spanning one or more software applications, often with initialized application data or files |
| Environment state | Application data and files that the available tools can read and change |
| Action space | Application tools, local utilities, text responses, and terminal commands where provided |
| Verifier | A script checks the final application and workspace state |
| Scoring rule | The binary output from the verifier is used directly |
| Gold answer | Expected final state encoded by the evaluation script |

```python
Task(
    benchmark="toolathon",
    task_id="subway-planning",
    environment_id="subway-planning",
    environment_description=(
        "Software applications containing accounts and data, together with a "
        "shared workspace for reading and writing files."
    ),
    task_description=(
        "I am staying at the Singapore Mobility Gallery and would like to "
        "travel to Changi Airport MRT station. ..."
    ),
    environment_metadata={
        "action_space": {
            "types": ["tool_calls", "text"],
            "mcp_servers": [
                "google_map",
                "filesystem",
                "playwright_with_chunk",
                "fetch",
            ],
        },
        "environment_state": {
            "type": "in_memory",
            "access": "read_write",
        },
    },
    task_metadata={
        "initialization": {
            "workspace": "tasks/finalpool/subway-planning/initial_workspace"
        },
        "extra_context": {
            "system_prompt": "Accessible workspace directory: /workspace/dumps/workspace ...",
            "input_files": [],
        },
    },
    gold_answer=None,
    verifiers=[{"type": "script", "mode": "final"}],
    scoring_rule="direct",
    source_platform="huggingface",
)

Record(
    benchmark="toolathon",
    task_id="subway-planning",
    agent_id="o3+toolathon",
    agent_model="o3",
    agent_scaffold="toolathon",
    trial=1,
    result=1,
    source_platform="huggingface",
)
```

### Subway planning (`subway-planning`) example

**Agent** = **Model:** `o3` + **Scaffold:** `toolathon`

| | Interaction |
| --- | --- |
| 👤 **User** | Requests an MRT-only route from Singapore Mobility Gallery to Changi Airport and asks for every station to be written to `routine.txt`. |
| 🤖 **Agent** | Calls `filesystem-write_file` with the ordered stations from Little India through Expo to Changi Airport. |
| 🌐 **Env** | Returns `Successfully wrote to /workspace/dumps/workspace/routine.txt`. |
| 🤖 **Agent** | Returns the route and confirms that the station list was saved. |
| 🟥 **Trial result** | `1`, because the task's evaluation script accepts the final file. |

## References

- Imported results: [Toolathlon Trajectories](https://huggingface.co/datasets/hkust-nlp/Toolathlon-Trajectories)
- Original benchmark: [Toolathlon](https://github.com/hkust-nlp/Toolathlon), its [paper](https://arxiv.org/abs/2510.25726), and its [website](https://toolathlon.xyz)
