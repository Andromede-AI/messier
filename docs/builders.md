# Adding a benchmark

Each builder translates one pinned benchmark into task, verifier, and result rows, extracting as much information from the source as possible. If something is unsupported by the data models it usually goes under metadata. Trajectories are streamed separately whenever the source provides them.

| Object | Plain meaning |
| --- | --- |
| `Task` | What the agent is asked to do |
| `VerifierDefinition` | One grading procedure |
| `Record` | One verifier result, trial result, or source-reported summary |
| `Trajectory` | Step-by-step interaction between the agent and the environment during the task |

Below you will find an example for adding a benchmark. All data models and important fields can be found in [`models.py`](../src/messier/models.py), while more information is available in the existing builders' READMEs under [`builders/`](../src/messier/builders). Typically, there are several things to check:

1. **Environment**
   - What state type is it? Choose one of `none` (no state beyond the task inputs), `filesystem` (files and directories, such as a software repository), `in_memory` (program objects, such as a tool-managed database), or `live_web` (an external website or web service). Only one environment state type is stored per task for now.
   - What kind of access does the agent have to that state (mutable or immutable)? Choose one of `none`, `read_only`, or `read_write`. Reading search results is read-only. Editing files or changing a reservation through a tool is read-write.
   - What is the agent's action space? Choose one or more of `text`, `tool_calls`, `shell`, and `ui`. Multiple action-space types are supported for the same task.
2. **Task**
   - Where can we fetch the original instruction given to the solver agent? Store it in `task_description`.
   - Is there any additional information intended for the agent, such as a system prompt, starter code, or files it is told to inspect? Store it under `task_metadata.extra_context`.
   - For input files, store both `path`, where the agent saw the file, and `dataset_path`, where the file is preserved in the release, under `task_metadata.extra_context.input_files`.
   - Does the source provide a gold answer or rubric? If so, store it in `gold_answer`. How is success defined?
3. **Verifier**
   - Do we have one verifier or more? Create separate verifier definitions only when their results are reported separately (like multiple LLM judges or a combination of an LLM judge with a script). A counterexample: a script that runs 50 internal tests but returns one pass/fail value is one verifier, since the result does not directly expose the individual tests or the intricacies of how they were combined.
   - What type is each verifier? Choose from `script`, `exact_match`, `llm_judge`, and `human_label`. A task can have multiple verifiers, including verifiers of different types.
   - Does the verifier run on the final output or state, or does it check intermediate interactions? Use `final` or `sequential`, respectively.
   - Does the source provide more verifier information? Store its description in `VerifierDefinition.description` and its verifier-specific reference answer in `VerifierDefinition.gold_answer`.
4. **Trial result**
   - How are verifier results converted into a trial result? Use `direct` when one binary verifier result is used unchanged, `all_pass` when every binary verifier must return `1`, or `threshold` when a numerical result or multiple verifiers must meet some threshold.
   - What is the binary result after applying that scoring rule to the verifier results? Store it in `Record.result` on the `trial_result`.
   - Store separately reported verifier results and the final trial result as `Record` rows with the same task, agent, and trial index.
   - Are there repeated attempts? Give each one a separate `trial` index so we retain maximal info. Use `result=None` for non-scorable runs. A scored agent failure remains `0`.
5. **Agent**
   - What model was used? Normalize its name using the shared mappings in [`agents.py`](../src/messier/builders/agents.py). If its not in that list then it has to be added together with its date of release.
   - Was a reasoning setting reported? Store it in `model_reasoning_effort`. The available values are `enabled` (for models without a scale, e.g. that have *reasoning or thinking* only), `low`, `medium`, `high`, `xhigh`, and `max`.
   - Was the solver agent using a scaffold? If so, identify it independently of the model and use its normalized name. Leave `agent_scaffold` null when the source does not identify one.
   - Build `agent_id` from the normalized model, reasoning effort, and scaffold.
6. **Trajectory**
   - Does the source provide the step-by-step interaction? If so, store it as a `Trajectory` with the same task, agent, and trial index as the corresponding records.
7. **Source**
   - Where are the data hosted? Store `github` or `huggingface` in `source_platform`.
   - Did an intermediary collection, such as BRIDGE or Agent Psychometrics, provide the data? If so, store it in `data_provider`.
   - Is every repository commit or dataset revision pinned so the source can be fetched again?


```python
from ...models import (
    ActionSpaceType,
    BuildResult,
    SourcePlatform,
    EnvironmentStateAccess,
    EnvironmentStateType,
    Record,
    RecordType,
    ScoringRule,
    Task,
    Trajectory,
    VerifierDefinition,
    VerifierMode,
    VerifierType,
)


def build() -> BuildResult:
    benchmark = "example"
    task_id = "task-1"
    verifier_id = "task-1::verifier:0"

    task = Task(
        benchmark=benchmark,
        task_id=task_id,
        environment_id="example-environment",
        environment_description="A short description of the system used by the task.",
        environment_metadata={
            "action_space": {"types": [ActionSpaceType.TEXT]},
            "environment_state": {
                "type": EnvironmentStateType.NONE,
                "access": EnvironmentStateAccess.NONE,
            },
        },
        task_description="The instruction shown to the agent.",
        verifiers=[{"type": VerifierType.EXACT_MATCH, "mode": VerifierMode.FINAL}],
        scoring_rule=ScoringRule.DIRECT,
        source_platform=SourcePlatform.HUGGINGFACE,
    )

    verifier = VerifierDefinition(
        benchmark=benchmark,
        task_id=task_id,
        verifier_id=verifier_id,
        description="Checks whether the answer matches the reference answer.",
        verifier={"type": VerifierType.EXACT_MATCH, "mode": VerifierMode.FINAL},
        source_platform=SourcePlatform.HUGGINGFACE,
    )

    trial_result = Record(
        benchmark=benchmark,
        task_id=task_id,
        agent_id="model+scaffold",
        agent_model="model",
        agent_scaffold="scaffold",
        trial=0,
        result=1,
        source_platform=SourcePlatform.HUGGINGFACE,
    )

    return [task], [verifier], [trial_result]


def trajectories():
    yield Trajectory(
        benchmark="example",
        task_id="task-1",
        agent_id="model+scaffold",
        agent_model="model",
        agent_scaffold="scaffold",
        trial=0,
        content='[{"role": "user", "content": "The instruction"}]',
        source_platform=SourcePlatform.HUGGINGFACE,
    )
```

The example uses direct scoring, so the one verifier result and the trial result are the same value and are stored once. If a source reports several verifier results, store each one as a separate record with the same `trial` index:

```python
Record(
    benchmark=benchmark,
    task_id=task_id,
    verifier_id=verifier_id,
    record_type=RecordType.VERIFIER_RESULT,
    agent_id="model+scaffold",
    agent_model="model",
    agent_scaffold="scaffold",
    trial=0,
    result=1,
    source_platform=SourcePlatform.HUGGINGFACE,
)
```

The final trial result is another `Record` with the same task, agent, and trial index. Its value must follow one of the supported scoring rules: `direct`, `all_pass`, or `threshold`. Use `result=None` for non-scorable runs, such as those affected by environment or infrastructure failures. A scored agent failure remains `0`.

It can happen that individual results are unavailable and a summary such as pass@k is reported. LiveCodeBench is currently the only such case. If so, store the reported value, normalized to 0--1, in `result`, set `record_type` to `source_summary`, and leave `trial` as `None`.

## Common patterns

- **One binary verifier.** [`hle/build.py`](../src/messier/builders/hle/build.py) imports one final LLM-judge result, which becomes the trial result directly.
- **Multiple verifier results.** [`harbor_runs/build.py`](../src/messier/builders/harbor_runs/build.py) imports each HarveyAI-Lab rubric result and uses all-pass scoring.
- **Numerical threshold.** [`gdpval/build.py`](../src/messier/builders/gdpval/build.py) represents a numerical judge score with a task threshold.
- **Final or sequential evaluation.** [`bfcl/build.py`](../src/messier/builders/bfcl/build.py) marks BFCL Live evaluation as final and BFCL multi-turn evaluation as sequential.
- **Trajectories.** [`harbor_runs/build.py`](../src/messier/builders/harbor_runs/build.py) streams the source-provided JSON trajectory for each available trial.
- **Source summary only.** [`livecodebench/build.py`](../src/messier/builders/livecodebench/build.py) stores reported pass@1 values as source summaries.

## Register and test

1. Create `src/messier/builders/<name>/build.py` and re-export `build` from its `__init__.py`. Export `trajectories` when the source provides interactions.
2. Register the builder in [`build.py`](../src/messier/build.py).
3. Add its group, default scoring rule, and release date to [`benchmarks.py`](../src/messier/builders/benchmarks.py).
4. Add its frozen raw-data fetch to [`fetch.py`](../src/messier/fetch.py).
5. Pin every upstream repository or dataset revision.
6. Run the builder and inspect its three returned lists and any streamed trajectories.

Use the shared mappings for model names, scaffolds, agent IDs, and dates instead of adding source-specific string rules. This keeps the same entity comparable across benchmarks.
