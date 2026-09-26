# Harbor Runs Description

MESSIER includes evaluations collected through [Harbor](https://harborframework.com/) on six benchmarks: DABStep, HarveyAI-Lab, QCircuitBench, ReplicationBench, ScienceAgentBench, and MedAgentBench.

| Benchmark | Task and environment | Environment state | Verifier | Scoring rule | Gold answer |
| --- | --- | --- | --- | --- | --- |
| DABStep | A data-analysis question with payment data and documentation | Files and directories | A Python script compares the submitted answer with the expected answer | The binary output from the verifier is used directly | The expected numerical or categorical answer |
| HarveyAI-Lab | A legal task with matter documents and one or more requested deliverables | Files and directories | Claude Sonnet 4.6 evaluates 23-194 task-specific rubric items and returns one binary result for each | All-pass. Every verifier result must be 1 | None. The deliverables are evaluated against the rubric items |
| QCircuitBench | A quantum-circuit design environment with Qiskit, a writable Python workspace, and an oracle or target circuit | Files and directories | A Python script simulates the submitted circuit and evaluates its validity, behavior, and gate count on a scale from 0 to 1 | Threshold. A trial succeeds when the source score is 1 | The target circuit behavior |
| ReplicationBench | An astrophysics replication task with a paper, scientific data, and supporting software | Files and directories | A Python script compares the submitted result with the expected result using a task-defined tolerance | The binary output from the verifier is used directly | The expected numerical or structured result |
| ScienceAgentBench | A scientific computing task with research data and a requested Python program | Files and directories | A Python script runs the program and evaluates its generated output, using an LLM judge for visual outputs | The binary output from the verifier is used directly | The oracle program and reference output held by the official evaluator |
| MedAgentBench | A clinical task in a simulated electronic health record system | Patient records maintained by a local FHIR server | A Python script compares the submitted answer and resulting clinical state with the reference solution | The binary output from the verifier is used directly | The expected answer and, when applicable, clinical state |

The shared DABStep data and scorer are included under `task_files/dabstep/`, and each task contains its expected answer. Original HarveyAI-Lab input files are included under `task_files/harveyai-lab/`. ReplicationBench includes the paper and task context available inside each environment, together with the source dataset information. Agent-visible files are linked from each task's `extra_context`. ScienceAgentBench instructions and results are retained, while its input files, reference outputs, oracle programs, evaluator scripts, and trajectories are not redistributed. The complete environment remains available through the [official benchmark](https://github.com/OSU-NLP-Group/ScienceAgentBench#benchmark-access).

The following examples show selected fields from the same tasks and results used in the interaction examples below. Long task descriptions are shortened.

### DABStep example

```python
Task(
    benchmark="dabstep",
    task_id="63",
    environment_id="63",
    task_description="What are the possible values for the field account_type? ...",
    environment_metadata={
        "action_space": {"types": ["shell"]},
        "environment_state": {
            "type": "filesystem",
            "access": "read_write",
            "ref": "task_files/dabstep/context",
        },
    },
    task_metadata={
        "extra_context": {
            "input_files": [
                {
                    "path": "/app/data/payments.csv",
                    "dataset_path": "task_files/dabstep/context/payments.csv",
                },
                ...
            ]
        }
    },
    gold_answer="R,D,H,F,S,O",
    verifiers=[
        {
            "type": "script",
            "mode": "final",
            "name": "dabstep_scorer",
            "script_code": "...",
            "comparison": "fuzzy",
        }
    ],
    difficulty_label="easy",
    scoring_rule="direct",
    source_platform="huggingface",
)

Record(
    benchmark="dabstep",
    task_id="63",
    agent_id="gpt-4o+openhands",
    agent_model="gpt-4o",
    agent_scaffold="openhands",
    trial=0,
    result=1,
    source_platform="huggingface",
)
```

### HarveyAI-Lab example

This task has 92 rubric verifiers, shortened below to two.

```python
Task(
    benchmark="harveyai-lab",
    task_id="antitrust-competition-analyze-antitrust-hsr-strategy",
    task_description="Review the attached deal documents and prepare an antitrust risk assessment and HSR filing strategy ...",
    environment_metadata={
        "action_space": {"types": ["shell"]},
        "environment_state": {
            "type": "filesystem",
            "access": "read_write",
            "ref": "task_files/harveyai-lab/antitrust-competition-analyze-antitrust-hsr-strategy",
        },
    },
    task_metadata={
        "harbor_dataset_slug": "harveyai/lab",
        "harbor_task_name": "harveyai/antitrust-competition-analyze-antitrust-hsr-strategy",
        "extra_context": {
            "input_files": [
                {
                    "path": "/workspace/documents/bulk-co2-market-analysis.xlsx",
                    "dataset_path": "task_files/harveyai-lab/antitrust-competition-analyze-antitrust-hsr-strategy/bulk-co2-market-analysis.xlsx",
                },
                ...
            ]
        },
    },
    gold_answer=None,
    verifiers=[
        {
            "type": "llm_judge",
            "mode": "final",
            "role": "rubric_item",
            "judge_model": "anthropic/claude-sonnet-4-6",
            "name": "c-001",
            "weight": 1.0,
        },
        {
            "type": "llm_judge",
            "mode": "final",
            "role": "rubric_item",
            "judge_model": "anthropic/claude-sonnet-4-6",
            "name": "c-002",
            "weight": 1.0,
        },
        ...
    ],
    scoring_rule="all_pass",
    source_platform="huggingface",
)

Record(
    benchmark="harveyai-lab",
    task_id="antitrust-competition-analyze-antitrust-hsr-strategy",
    verifier_id="antitrust-competition-analyze-antitrust-hsr-strategy::c-001",
    record_type="verifier_result",
    agent_id="claude-haiku-4-5+openhands",
    agent_model="claude-haiku-4-5",
    agent_scaffold="openhands",
    trial=0,
    result=1,
    source_platform="huggingface",
)

Record(
    benchmark="harveyai-lab",
    task_id="antitrust-competition-analyze-antitrust-hsr-strategy",
    record_type="trial_result",
    agent_id="claude-haiku-4-5+openhands",
    agent_model="claude-haiku-4-5",
    agent_scaffold="openhands",
    trial=0,
    result=0,
    source_platform="huggingface",
)
```

### QCircuitBench example

```python
Task(
    benchmark="qcircuitbench",
    task_id="deutsch_jozsa-n8",
    environment_id="deutsch_jozsa-n8",
    task_description="Design an eight-qubit Deutsch-Jozsa circuit ...",
    environment_metadata={
        "action_space": {"types": ["shell"]},
        "environment_state": {
            "type": "filesystem",
            "access": "read_write",
            "ref": "sha256:16c4ad6ac83891990437ab087c374b1f1c7c1da74255cf2007d1579d30b95254",
        },
    },
    task_metadata={
        "paper_id": "hubble_trails",
        "extra_context": {
            "input_files": [
                {
                    "path": "/app/resources/paper_masked.json",
                    "dataset_path": "task_files/replicationbench/hubble_trails__satellite_fractions/paper_masked.json",
                },
                {
                    "path": "/app/resources/dataset_info.json",
                    "dataset_path": "task_files/replicationbench/hubble_trails__satellite_fractions/dataset_info.json",
                },
            ],
            "dataset": {
                "kind": "huggingface",
                "hf_name": ["StevenDillmann/hubble_trails"],
                "hf_revision": ["97ffe47377cdf595b2a26fd08440ae37201efe4a"],
            },
        },
    },
    gold_answer=[0.027, 0.032, 0.017],
    verifiers=[
        {
            "type": "script",
            "mode": "final",
            "comparison": "tolerance",
            "tolerance": [0.002, 0.002, 0.002],
        }
    ],
    scoring_rule="threshold",
    source_platform="huggingface",
)

Record(
    benchmark="qcircuitbench",
    task_id="deutsch_jozsa-n8",
    agent_id="claude-haiku-4-5+openhands",
    agent_model="claude-haiku-4-5",
    agent_scaffold="openhands",
    trial=0,
    result=1,
    source_platform="huggingface",
)
```

### ReplicationBench example

```python
Task(
    benchmark="replicationbench",
    task_id="hubble_trails__satellite_fractions",
    environment_id="hubble_trails__satellite_fractions",
    task_description="Compute three satellite-trail fractions from the supplied Hubble observations ...",
    environment_metadata={
        "action_space": {"types": ["shell"]},
        "environment_state": {
            "type": "filesystem",
            "access": "read_write",
            "ref": "sha256:2854a5641f212e666b2322809c1ec37dae2f6197528aa5a9b047b835fe737e9a",
        },
    },
    gold_answer=None,
    verifiers=[{"type": "script", "mode": "final"}],
    scoring_rule="direct",
    source_platform="huggingface",
)

Record(
    benchmark="replicationbench",
    task_id="hubble_trails__satellite_fractions",
    agent_id="gpt-5-nano+openhands",
    agent_model="gpt-5-nano",
    agent_scaffold="openhands",
    trial=0,
    result=1,
    source_platform="huggingface",
)
```

### ScienceAgentBench example

```python
Task(
    benchmark="scienceagentbench",
    task_id="sab_100",
    environment_id="sab_100",
    task_description="Visualize the spatial data by total counts, gene counts, and clusters ...",
    environment_metadata={
        "action_space": {"types": ["shell"]},
        "environment_state": {
            "type": "filesystem",
            "access": "read_write",
            "ref": "sha256:efa3ed01037b832735e9c42a5077d0b8cba828e0c27dbc8a9947cbc1564532ec",
        },
    },
    task_metadata={
        "category": "scientific_computing",
        "tags": ["scienceagentbench", "Bioinformatics", "scientific_computing"],
    },
    gold_answer=None,
    verifiers=[
        {
            "type": "script",
            "mode": "final",
            "uses_llm_judge": True,
        }
    ],
    difficulty_label="medium",
    scoring_rule="direct",
    source_platform="huggingface",
)

Record(
    benchmark="scienceagentbench",
    task_id="sab_100",
    agent_id="gpt-5-nano+openhands",
    agent_model="gpt-5-nano",
    agent_scaffold="openhands",
    trial=0,
    result=1,
    source_platform="huggingface",
)
```

### MedAgentBench example

```python
Task(
    benchmark="medagentbench",
    task_id="task1_19",
    environment_id="task1_19",
    task_description="What is the MRN of the patient named Tim Ramos, born on 1959-04-28? ...",
    environment_metadata={
        "action_space": {"types": ["shell", "tool_calls"]},
        "environment_state": {
            "type": "filesystem",
            "access": "read_write",
            "ref": "sha256:b8b5c5cb521042e0c7dc83fbb9c771c135f1061dffab035c9052c53958f9377d",
        },
    },
    gold_answer=None,
    verifiers=[{"type": "script", "mode": "final"}],
    scoring_rule="direct",
    source_platform="huggingface",
)

Record(
    benchmark="medagentbench",
    task_id="task1_19",
    agent_id="claude-haiku-4-5+openhands",
    agent_model="claude-haiku-4-5",
    agent_scaffold="openhands",
    trial=0,
    result=1,
    source_platform="huggingface",
)
```

## Interaction examples

These are real source trials. Long prompts, files, and command outputs are shortened.

### DABStep (`63`) example

**Agent** = **Model:** `gpt-4o` + **Scaffold:** `openhands`

| | Interaction |
| --- | --- |
| 👤 **User** | What are the possible values for the field `account_type`? |
| 🤖 **Agent** | Opens the supplied data and its documentation, then writes the answer to `/app/answer.txt`. |
| 🌐 **Env** | Returns the data files and their contents. |
| 🤖 **Agent** | Answers `R, D, H, F, S, O`. |
| 🟥 **Trial result** | `1`, because the submitted answer matches the expected answer. |

### HarveyAI-Lab (`antitrust-competition-analyze-antitrust-hsr-strategy`) example

**Agent** = **Model:** `claude-haiku-4-5` + **Scaffold:** `openhands`

| | Interaction |
| --- | --- |
| 👤 **User** | Review the attached deal documents and prepare an antitrust risk assessment and HSR filing strategy. |
| 🤖 **Agent** | Lists the supplied Word documents and spreadsheets, then reads their contents. |
| 🌐 **Env** | Returns the deal rationale, merger agreement, market-share data, competitor analysis, and supporting documents. |
| 🤖 **Agent** | Creates `antitrust-risk-assessment-memo.docx` and `hsr-filing-strategy-memo.docx` in `/workspace/output/`. |
| 🟥 **Trial result** | `0`, because at least one of the 92 verifier results is `0`. |

### QCircuitBench (`deutsch_jozsa-n8`) example

**Agent** = **Model:** `claude-haiku-4-5` + **Scaffold:** `openhands`

| | Interaction |
| --- | --- |
| 👤 **User** | Implement the eight-qubit Deutsch-Jozsa algorithm as an OpenQASM circuit and Python post-processing function. |
| 🤖 **Agent** | Creates `solution.py` containing an OpenQASM circuit that uses the provided `Oracle` gate and Python code that interprets the simulation result. |
| 🟥 **Trial result** | `1`, because the verifier loads the circuit and post-processing code and all 100 executions across 10 test cases succeed. |

### ReplicationBench (`hubble_trails__satellite_fractions`) example

**Agent** = **Model:** `gpt-5-nano` + **Scaffold:** `openhands`

| | Interaction |
| --- | --- |
| 👤 **User** | Compute three satellite-trail fractions from the supplied Hubble observations. |
| 🤖 **Agent** | Inspects the CSV file, computes the fractions, and writes them to `/app/result.json`. |
| 🌐 **Env** | Returns `[0.02680464544050538, 0.03179236352161565, 0.016964540478846202]`. |
| 🟥 **Trial result** | `1`, because all three submitted values are within the accepted tolerances. |

### ScienceAgentBench (`sab_100`) example

**Agent** = **Model:** `gpt-5-nano` + **Scaffold:** `openhands`

| | Interaction |
| --- | --- |
| 👤 **User** | Write a Python program that visualizes spatial data by total counts, gene counts, and clusters in one image. |
| 🤖 **Agent** | Creates `/testbed/spatial_2.py`, which loads the supplied data and writes `pred_results/spatial_2_pred.png`. |
| 🌐 **Env** | Confirms that the program was created. |
| 🟥 **Trial result** | `1`, because the generated program runs successfully and its output passes the script evaluation. |

### MedAgentBench (`task1_19`) example

**Agent** = **Model:** `claude-haiku-4-5` + **Scaffold:** `openhands`

| | Interaction |
| --- | --- |
| 👤 **User** | Find the medical record number for Tim Ramos, born on April 28, 1959. |
| 🤖 **Agent** | Queries the local FHIR server for the patient's name and date of birth. |
| 🌐 **Env** | Returns the matching patient with medical record number `S6192632`. |
| 🤖 **Agent** | Writes `S6192632` to `/workspace/answer.json`. |
| 🟥 **Trial result** | `1`, because the submitted medical record number matches the official reference solution. |

## References

- Evaluation framework: [Harbor](https://harborframework.com/)
- Original benchmarks: [DABStep](https://huggingface.co/datasets/adyen/DABstep), [HarveyAI-Lab](https://github.com/harveyai/harvey-labs), [QCircuitBench](https://github.com/EstelYang/QCircuitBench), [ReplicationBench](https://github.com/Christine8888/replicationbench-release), [ScienceAgentBench](https://github.com/OSU-NLP-Group/ScienceAgentBench), and [MedAgentBench](https://github.com/stanfordmlgroup/MedAgentBench)
