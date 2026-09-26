# MLE-bench Description

[MLE-bench](https://github.com/openai/mle-bench) evaluates agents on machine learning engineering through 75 Kaggle competitions. Each task requires an agent to inspect competition data, develop and train a model, and produce a prediction file. This builder joins 82 task definitions from the MLE-bench repository with 494 trial results for 13 models on 38 competitions released by [BRIDGE](https://github.com/McGill-NLP/BRIDGE).

The BRIDGE results provide three binary values for each trial: whether the submission is valid, whether it scores above the competition median, and whether it earns any medal. We use the source-provided `any_medal` value directly as the trial result and retain `valid_submission` and `above_median` in metadata.

| | MLE-bench |
| --- | --- |
| Task and environment | A Kaggle competition with training data, test data, and instructions for producing a prediction file |
| Environment state | Competition files, code, trained models, and generated submissions that the agent can inspect and modify |
| Action space | Shell commands for inspecting data, training models, and producing a submission |
| Verifier | A script compares the submitted predictions with held-out answers using the competition's evaluation metric |
| Scoring rule | The source-provided `any_medal` value is used directly as the trial result |
| Gold answer | The held-out answers used by the competition's grading script |

The following example contains fields from one imported task and result. Long text, scripts, and metadata are shortened for readability.

```python
Task(
    benchmark="mlebench",
    task_id="aerial-cactus-identification",
    environment_id="aerial-cactus-identification",
    environment_description=(
        "A Kaggle competition environment for Aerial Cactus Identification, "
        "containing training and test data, common machine learning libraries, "
        "and a grading server."
    ),
    task_description=(
        "Create an algorithm that identifies a specific type of cactus "
        "in aerial images. ..."
    ),
    environment_metadata={
        "action_space": {
            "types": ["shell"],
            "description": (
                "Shell commands for inspecting the data, training models, and "
                "producing a submission file."
            ),
        },
        "environment_state": {
            "type": "filesystem",
            "access": "read_write",
            "ref": "kaggle://competitions/aerial-cactus-identification",
        },
        "url": "https://www.kaggle.com/competitions/aerial-cactus-identification",
    },
    task_metadata={
        "name": "Aerial Cactus Identification",
        "competition_type": "code",
        "awards_medals": False,
        "grader_name": "auc-roc",
    },
    verifiers=[
        {
            "type": "script",
            "mode": "final",
            "name": "auc-roc",
            "success_condition": "any_medal",
            "script_code": "...",
        }
    ],
    scoring_rule="direct",
    source_platform="github",
)

Record(
    benchmark="mlebench",
    task_id="aerial-cactus-identification",
    agent_id="gemini-2-5-flash",
    agent_model="gemini-2-5-flash",
    agent_scaffold=None,
    trial=0,
    result=0,
    metadata={
        "source_results": {
            "valid_submission": 1,
            "above_median": 0,
            "any_medal": 0,
        },
        "input_tokens": 168420.0,
        "num_turns": 13.0,
        "output_tokens": 6956.0,
        "reasoning_tokens": 2317.0,
        "total_tokens": 177693.0,
    },
    source_platform="github",
    data_provider="bridge",
)
```

The released BRIDGE results do not include step-by-step agent trajectories, so no interaction example is shown.

## References

- Imported results: [BRIDGE](https://github.com/McGill-NLP/BRIDGE)
- Original benchmark: [MLE-bench](https://github.com/openai/mle-bench) and its [paper](https://arxiv.org/abs/2410.07095)
