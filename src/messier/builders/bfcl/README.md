# BFCL Description

The Berkeley Function-Calling Leaderboard (BFCL) evaluates whether language models select functions and produce valid calls from provided function definitions. MESSIER includes BFCL Live and BFCL Multi-Turn from the `2025-12-16` snapshot of [`HuanzhiMao/BFCL-Result`](https://github.com/HuanzhiMao/BFCL-Result), with the multi-turn environment linked to [`ShishirPatil/gorilla`](https://github.com/ShishirPatil/gorilla).

In BFCL, a user turn is one predefined instruction. BFCL Live contains one such turn, with no later instruction or changing environment state. BFCL Multi-Turn presents several turns in sequence. After each one, the agent calls functions, sees their responses, and continues from the environment state produced by earlier calls.

| | BFCL Live | BFCL Multi-Turn |
| --- | --- | --- |
| Task and environment | Initial instruction and a set of available functions to choose from | A conversation over several user turns and a set of stateful classes |
| Environment state | Not stateful. Nothing is changing. | Stored data, such as files or API records, which function calls can change. |
| Action space | One or more function calls | Function calls and text replies |
| Verifier | Compares the produced calls with the expected calls | Runs the agent's calls and expected calls, then compares their responses and final environment state |
| Gold answer | The function call expected for the request, including its name and arguments. Some tasks expect multiple calls. | The function calls expected for each user turn |


```python
Task(
    benchmark="bfcl-live",
    task_id="live_multiple_0-0-0",
    environment_id="live_multiple",
    task_description="update my latte to a large size with coconut milk ...",
    environment_metadata={
        "action_space": {
            "types": ["tool_calls"],
            "mode": "per_task_schemas",
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "ChaDri.change_drink",
                        "parameters": {"...": "..."},
                    },
                }
            ],
        },
        "environment_state": {"type": "none", "access": "none"},
    },
    gold_answer="expected function calls",
    verifiers=[{"type": "exact_match", "mode": "final"}],
    source_platform="github",
)

Record(
    benchmark="bfcl-live",
    task_id="live_multiple_266-127-1",
    agent_id="bitagent-bounty-8b+bfcl-prompted",
    agent_model="bitagent-bounty-8b",
    agent_scaffold="bfcl-prompted",
    trial=0,
    result=1,
    source_platform="github",
)
```

### BFCL Live (`live_simple_0-0-0`) example:

**Agent** = **Model:** `gpt-4-1-nano` + **Scaffold:** `bfcl-fc`

| | Interaction |
| --- | --- |
| 👤 **User** | Can you retrieve the details for the user with ID 7890, who has black as their special request? |
| 🌐 **Env** | Makes `get_user_info(user_id, special)` available. |
| 🤖 **Agent** | Calls `get_user_info(user_id=7890, special="black")`. |
| 🤖 **Agent** | Calls `get_user_info(user_id=7890, special="none")`. |
| 🟥 **Trial result** | `0`, because the gold answer contains only the first call. |

### BFCL Multi-Turn (`multi_turn_miss_param_0`) example:

**Agent** = **Model:** `gpt-4-1-nano` + **Scaffold:** `bfcl-fc`

| | Interaction |
| --- | --- |
| 👤 **User** | Move `final_report.pdf` within the `document` directory to a `temp` directory. Make sure to create the directory. |
| 🤖 **Agent** | Calls `pwd()`. |
| 🌐 **Env** | Returns `{"current_working_directory": "/workspace"}`. |
| 🤖 **Agent** | Calls `mkdir(dir_name="document/temp")`. |
| 🌐 **Env** | Returns `{"error": "mkdir: cannot create directory 'document/temp': Invalid character"}`. |
| 👤 **User** | Perform a detailed search using grep to identify sections in the file pertaining to `budget analysis`. |
| 🤖 **Agent** | Calls `grep(file_name="final_report.pdf", pattern="budget analysis")`. |
| 🌐 **Env** | Returns `{"error": "grep: final_report.pdf: No such file or directory"}`. |
| 🟥 **Trial result** | `0`, because the failed calls leave the filesystem in the wrong final state. |

## References

- Imported results: [`HuanzhiMao/BFCL-Result`](https://github.com/HuanzhiMao/BFCL-Result)
- Original benchmark: [Berkeley Function-Calling Leaderboard](https://github.com/ShishirPatil/gorilla/tree/main/berkeley-function-call-leaderboard)
