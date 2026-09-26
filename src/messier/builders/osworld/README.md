# OSWorld Description

[OSWorld](https://github.com/xlang-ai/OSWorld) evaluates agents on computer tasks performed in an Ubuntu desktop. Its 361 tasks cover Chrome, GIMP, LibreOffice, Thunderbird, VLC, VS Code, operating-system settings, and workflows spanning several applications.

Each task starts from a specified virtual-machine snapshot. The agent controls the desktop through mouse and keyboard actions and can issue shell commands by opening a terminal. After the agent finishes, one or more task-specific scripts inspect the final application state or output files and return one combined reward.

Files used to initialize tasks are included under `task_files/osworld/` and linked from `extra_context` with their original paths in the virtual machine.

| | OSWorld |
| --- | --- |
| Task and environment | An instruction completed in an Ubuntu desktop containing one or more applications |
| Environment state | Application settings, files, open documents, and website state initialized from a virtual-machine snapshot |
| Action space | Desktop UI actions and shell commands entered through the terminal |
| Verifier | Task-specific scripts inspect the final desktop state or compare produced files with expected outputs, returning a numerical reward from `0` to `1` |
| Scoring rule | Threshold. A reward of `1.0` produces a trial result of `1`; lower rewards produce `0` |
| Gold answer | The expected application state, file contents, URL, or reference file defined by the task |

The source may combine several evaluator functions into one reward. We preserve that reward as a verifier result and separately store the binary trial result.

For agent frameworks, `agent_model` identifies the primary model and `agent_scaffold` identifies the framework. Additional grounding or execution models are stored in record metadata when the source identifies them.

The following example contains fields from one imported task and its two result records.

```python
Task(
    benchmark="osworld",
    task_id="8f080098-ddb1-424c-b438-4e96e5e4786e",
    environment_id="vlc",
    environment_description=(
        "An Ubuntu desktop with VLC, media files, playback controls, conversion "
        "tools, filters, and application settings."
    ),
    environment_metadata={
        "action_space": {"types": ["ui", "shell"]},
        "environment_state": {
            "type": "filesystem",
            "access": "read_write",
            "ref": "osworld://vlc/base_setup",
        },
    },
    task_description=(
        "Could you convert the song from this music video as an MP3 file? ... "
        "Please save the file just on the desktop and title the file "
        "\"Baby Justin Bieber.mp3.\""
    ),
    task_metadata={
        "extra_context": {
            "input_files": [
                {
                    "path": "/home/user/Desktop/Baby Justin Bieber.mp4",
                    "dataset_path": "task_files/osworld/vlc/.../Baby Justin Bieber.mp4",
                }
            ]
        }
    },
    gold_answer={
        "type": "cloud_file",
        "dest": "baby_gold.mp3",
    },
    task_date="2024-04-11",
    verifiers=[
        {
            "type": "script",
            "mode": "final",
            "name": "compare_audios",
            "scale": "0.0-1.0",
            "threshold": 1.0,
        }
    ],
    scoring_rule="threshold",
    source_platform="github",
)

Record(
    benchmark="osworld",
    task_id="8f080098-ddb1-424c-b438-4e96e5e4786e",
    record_type="verifier_result",
    verifier_id="8f080098-ddb1-424c-b438-4e96e5e4786e::verifier:0",
    agent_id="o3+osworld-default-100steps",
    agent_model="o3",
    agent_scaffold="osworld-default-100steps",
    trial=0,
    result=0.9868665069144599,
    source_platform="huggingface",
)

Record(
    benchmark="osworld",
    task_id="8f080098-ddb1-424c-b438-4e96e5e4786e",
    record_type="trial_result",
    agent_id="o3+osworld-default-100steps",
    agent_model="o3",
    agent_scaffold="osworld-default-100steps",
    trial=0,
    result=0,
    source_platform="huggingface",
)
```

## References

- Imported tasks: [OSWorld task configurations](https://github.com/xlang-ai/OSWorld/tree/main/evaluation_examples)
- Imported results: [OSWorld verified trajectories](https://huggingface.co/datasets/xlangai/ubuntu_osworld_verified_trajs)
- Original benchmark: [OSWorld](https://os-world.github.io/) and its [paper](https://arxiv.org/abs/2404.07972)
