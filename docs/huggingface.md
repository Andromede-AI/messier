---
pretty_name: MESSIER
license: other
configs:
- config_name: records
  default: true
  data_files:
  - split: full
    path: records.jsonl
- config_name: tasks
  data_files:
  - split: full
    path: tasks.jsonl
- config_name: verifiers
  data_files:
  - split: full
    path: verifiers.jsonl
- config_name: trajectories
  data_files:
  - split: full
    path: trajectories.jsonl
- config_name: classifications
  data_files:
  - split: full
    path: classifications.jsonl
---

<p align="center">
  <img src="https://raw.githubusercontent.com/Andromede-AI/messier/main/docs/assets/logo.png" width="220" alt="MESSIER logo">
</p>

<h1 align="center">MESSIER</h1>

<h3 align="center"><strong>A High-Resolution Corpus for Cross-Benchmark Agent Evaluation</strong></h3>

<p align="center">
  <a href="https://arxiv.org/abs/2607.25891"><img src="https://img.shields.io/badge/Paper-B31B1B?style=for-the-badge&logo=arxiv&logoColor=white" height="24" alt="Paper"></a>&nbsp;&nbsp;&nbsp;&nbsp;
  <a href="https://andromede-ai.github.io/messier/"><img src="https://img.shields.io/badge/Page-4285F4?style=for-the-badge&logo=googlechrome&logoColor=white" height="24" alt="Page"></a>&nbsp;&nbsp;&nbsp;&nbsp;
  <a href="https://huggingface.co/datasets/Andromede-AI/messier"><img src="https://img.shields.io/badge/Dataset-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black" height="24" alt="Dataset"></a>&nbsp;&nbsp;&nbsp;&nbsp;
  <a href="https://github.com/Andromede-AI/messier/blob/main/docs/data.md"><img src="https://img.shields.io/badge/Docs-1684C2?style=for-the-badge&logo=readthedocs&logoColor=white" height="24" alt="Documentation"></a>&nbsp;&nbsp;&nbsp;&nbsp;
  <a href="https://github.com/Andromede-AI/messier"><img src="https://img.shields.io/badge/Code-111111?style=for-the-badge&logo=github&logoColor=white" height="24" alt="Code"></a>
</p>

MESSIER is a corpus for the storage and comparison of agent evaluations. It contains model, scaffold, task, verifier and scoring information required to understand how each published result was produced. This repository contains the data loaders and builders, maintenance tools, and analyses presented in the paper. More generally, MESSIER is an effort to enumerate the elements of agent evaluation and to align their definitions and representations across different benchmarks.

As agent systems and evaluations evolve, keeping track of their progress is increasingly important. MESSIER can help answer questions about these systems, whether about their capabilities and safety or about how evaluations should be designed. It supports error analysis and studies of how performance changes over time, with task difficulty, across models and scaffolds, under different types of verifiers and more. Each task also has SOC and NAICS classifications for fine-grained analysis across occupations and industries, while full trajectories, where available, enable deeper analysis of agent behavior. We invite collaborators to contribute new benchmarks, models, runs and experiments (see [contributing](https://github.com/Andromede-AI/messier/blob/main/CONTRIBUTING.md)).

<table align="center">
  <tr>
    <td align="center"><strong>31</strong><br>Benchmarks</td>
    <td align="center"><strong>725</strong><br>Agents</td>
    <td align="center"><strong>11,999</strong><br>Tasks</td>
  </tr>
  <tr>
    <td align="center"><strong>72,000</strong><br>Verifiers</td>
    <td align="center"><strong>902,338</strong><br>Records</td>
    <td align="center"><strong>118,089</strong><br>Trajectories</td>
  </tr>
</table>


## News
- **[26/10/2026]** 🌕 We will present MESSIER at the EMNLP 2026 Main Conference.
- **[09/10/2026]** 🌔 We will present *Predicting Task Difficulty Without Rollouts* as a poster at the COLM 2026 Workshop on Agent Behavior (WAB).
- **[26/09/2026]** 🌓 We release the first public version of MESSIER.

- **[13/09/2026]** 🌒 Updated the data model to include `extra_context`, under which task files are saved. We added original task files from HarveyAI-Lab, GDPval, DABStep, OSWorld, and Toolathlon so the materials available to an agent can be accessed alongside each task.
- **[05/09/2026]** 🌑 Updated MathArena to a newer pinned source revision and moved APEX to its complete three-shard release, refreshing the imported tasks, model responses, scores, and trajectories.
- **[27/08/2026]** 🌘 Added Toolathlon with 108 tasks across 32 applications and 604 tools, together with 7,116 results from 22 model configurations.
- **[12/08/2026]** 🌗 Expanded the data model to distinguish multiple action-space types, environment state and access, final and sequential verification and model reasoning effort. SOC and NAICS classifications were also revised.
- **[01–02/08/2026]** 🌖 We presented MESSIER as a poster at the Berkeley RDI Summit.
- **[28/07/2026]** 🌕 Released and published the first version of MESSIER.

## Documentation

- [Data format and terminology](https://github.com/Andromede-AI/messier/blob/main/docs/data.md)
- [Adding a benchmark](https://github.com/Andromede-AI/messier/blob/main/docs/builders.md)
- [Publishing guide](https://github.com/Andromede-AI/messier/blob/main/docs/publishing.md)

## Install

Clone the repository and install with [uv](https://docs.astral.sh/uv/getting-started/installation/):

```bash
git clone https://github.com/Andromede-AI/messier.git
cd messier

# Install the data loaders
uv sync

# To include builders, analyses, and validation tools, use instead:
# uv sync --group dev

# Start Python
uv run python
```

## Load the data

**Download the files.** Run this example from the repository root. It downloads all five tables from [Hugging Face](https://huggingface.co/datasets/Andromede-AI/messier/tree/main) into the folders our loaders and analyses read:

```python
from huggingface_hub import hf_hub_download

for table in ("tasks", "records", "verifiers", "trajectories", "classifications"):
    hf_hub_download(
        "Andromede-AI/messier",
        f"{table}.jsonl",
        repo_type="dataset",
        local_dir="data/artifacts" if table == "classifications" else "data/processed",
    )
```

**Load in Python.** Our loaders fetch tasks, records, and verifiers as pandas DataFrames. They yield trajectories one at a time:

```python
import messier

# Add source="local" to read local files.
tasks = messier.tasks()
records = messier.records()
verifiers = messier.verifiers()
trajectories = messier.trajectories()
```

Or stream with Hugging Face's [datasets library](https://huggingface.co/docs/datasets):

```python
import json
from datasets import load_dataset

tasks = load_dataset(
    "text",
    data_files="hf://datasets/Andromede-AI/messier/tasks.jsonl",
    split="train",
    streaming=True,
).map(lambda row: json.loads(row["text"]), remove_columns=["text"])
```

This example streams rows from Hugging Face. Change the filename to stream records, verifiers, trajectories, or classifications. Note that the analysis scripts read from local files.

## Quick start

Below we show several use cases and quick starts for using MESSIER.


Combine all five tables into one complete dataset containing the full information.

```python
from collections import defaultdict
import pandas as pd
from messier.io import data_path, load_jsonl

tables = {}
for name in ("tasks", "records", "verifiers", "trajectories", "classifications"):
    rows_by_task = defaultdict(list)
    for row in load_jsonl(data_path(f"{name}.jsonl", source="local")):
        rows_by_task[(row["benchmark"], row["task_id"])].append(row)

    tables[name] = pd.Series(rows_by_task, dtype=object)

dataset = pd.DataFrame(tables).rename_axis(["benchmark", "task_id"]).reset_index()
print(dataset.head())
```

Compare agents within each benchmark using trials with a score.

```python
import messier

records = messier.records(source="local")
rates = (
    messier.trial_records(records)
    .groupby(["benchmark", "agent_id"])["result"]
    .agg(success_rate="mean", scored_trials="count")
)
print(rates.head())
```


Track each benchmark's best observed results by model release quarter.

```bash
uv run --group dev python analysis/frontier_progress/run.py
# Results: analysis/outputs/frontier_progress/
```

The script downloads and verifies the frozen raw sources, rebuilds the corpus, and runs validation tests.

```bash
bash scripts/update.sh
```

## License

MESSIER is licensed under [MIT](https://github.com/Andromede-AI/messier/blob/main/LICENSE).

**Disclaimer.** MESSIER incorporates material from original benchmarks and result providers, whose ownership and intellectual property we respect. Third-party material remains subject to its original terms. We have made our best effort to identify and cite these sources in the benchmark READMEs. We welcome requests from the original authors to correct an attribution, modify included material, or remove it when needed.

## Citation
```bibtex
@article{krsteski2026messier,
  title={Messier: A High-Resolution Corpus for Cross-Benchmark Agent Evaluation},
  author={Krsteski, Stefan and Meyer, Charlotte and Allegre, Guillaume and O'Halloran, Tony and Sallinen, Alexandre},
  journal={arXiv preprint arXiv:2607.25891},
  year={2026}
}
```
