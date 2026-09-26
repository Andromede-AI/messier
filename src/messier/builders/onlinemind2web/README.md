# Online-Mind2Web Description

[Online-Mind2Web](https://github.com/OSU-NLP-Group/Online-Mind2Web) evaluates web agents on 300 tasks performed on live public websites. Tasks include finding information, changing website settings, and completing multi-step actions. This builder joins the task definitions with human-evaluated results for ten agents from the public leaderboard. Human reviewers evaluate each completed trajectory from its task, action history, and screenshots. This builder imports their final success or failure judgments.

| | Online-Mind2Web |
| --- | --- |
| Task and environment | A user request performed on a live public website |
| Environment state | Website pages and session state that the agent can inspect and change |
| Action space | Browser actions such as clicking, typing, scrolling, and navigating, followed by a text response |
| Verifier | A human reviews the task, action history, and screenshots after the trial |
| Scoring rule | The binary output from the human verifier is used directly |
| Gold answer | No fixed answer. The human reviewer decides whether the requested web action was completed |

The following example contains fields from one imported task and result.

```python
Task(
    benchmark="onlinemind2web",
    task_id="b7258ee05d75e6c50673a59914db412e",
    environment_id="gamestop.com",
    environment_description=(
        "A live website at gamestop.com that the agent navigates and modifies "
        "through a browser."
    ),
    environment_metadata={
        "action_space": {"types": ["ui", "text"]},
        "environment_state": {
            "type": "live_web",
            "access": "read_write",
            "ref": "https://www.gamestop.com/",
        },
    },
    task_description=(
        "Find the store location and hours of the closest Gamestop to zip code "
        "90028 and set it as the home store on Gamestop."
    ),
    task_date="2025-04-02",
    verifiers=[
        {
            "type": "human_label",
            "mode": "final",
            "scale": "0=failure, 1=success",
        }
    ],
    human={"steps": 8},
    difficulty_label="medium",
    scoring_rule="direct",
    source_platform="huggingface",
)

Record(
    benchmark="onlinemind2web",
    task_id="b7258ee05d75e6c50673a59914db412e",
    agent_id="undisclosed+openai-operator",
    agent_model="undisclosed",
    agent_scaffold="openai-operator",
    model_date="2025-01-23",
    trial=0,
    result=1,
    metadata={"human_label_raw": "1"},
    source_platform="huggingface",
)
```

## References

- Imported tasks: [Online-Mind2Web dataset](https://huggingface.co/datasets/osunlp/Online-Mind2Web)
- Imported results: [Online-Mind2Web leaderboard](https://huggingface.co/spaces/osunlp/Online_Mind2Web_Leaderboard)
- Original benchmark: [Online-Mind2Web](https://github.com/OSU-NLP-Group/Online-Mind2Web) and its [paper](https://arxiv.org/abs/2504.01382)
