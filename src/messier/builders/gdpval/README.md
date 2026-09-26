# GDPval Description

[GDPval](https://huggingface.co/datasets/openai/gdpval) evaluates models on workplace tasks spanning 44 predominantly digital occupations across nine major sectors of the U.S. economy. Tasks ask models to produce practical deliverables such as documents, spreadsheets, and presentations. MESSIER joins the 220 public tasks released by [BRIDGE](https://github.com/McGill-NLP/BRIDGE).

BRIDGE runs each task through Inspect AI and uses Gemini 3 Pro to grade the resulting deliverable from 1 to 5 against a task-specific rubric. Grades of 4 or 5 are successful. The released results contain this final pass/fail value and related metadata, but not the generated deliverable or trajectory. Original task inputs are included under `task_files/gdpval/` and linked from `extra_context`.

| | GDPval |
| --- | --- |
| Task and environment | A task instruction and optional reference files, where the goal is to produce one or more deliverable files |
| Environment state | Files and directories |
| Action space | Bash shell and Python interpreter |
| Verifier | LLM Judge (Gemini 3 Pro) evaluates the final deliverables and assigns a grade from 1 to 5 |
| Scoring rule | Threshold. A grade of at least 4 is a success and produces a trial result of 1 |
| Gold answer | A rubric provided to the verifier, such as whether the document uses the required format and contains the requested information |

The following example contains fields from one imported task and result. The task description is shortened here.

```python
Task(
    benchmark="gdpval",
    task_id="1e5a1d7f-12c1-48c6-afd9-82257b3f2409",
    environment_id="property-real-estate-and-community-association-managers",
    task_description="You are the Vice President of Operations for a property management company. ...",
    environment_metadata={
        "action_space": {"types": ["shell"]},
        "environment_state": {
            "type": "filesystem",
            "access": "read_write",
            "ref": "task_files/gdpval/1e5a1d7f-12c1-48c6-afd9-82257b3f2409",
        },
    },
    task_metadata={
        "sector": "Real Estate and Rental and Leasing",
        "occupation": "Property, Real Estate, and Community Association Managers",
        "extra_context": {
            "input_files": [
                {
                    "path": "PM Duties.pdf",
                    "dataset_path": "task_files/gdpval/.../PM Duties.pdf",
                }
            ],
        },
    },
    gold_answer=[
        {
            "step": 0,
            "expected": "Deliverable is a Word document",
            "target_score": 2,
        },
        ...
    ],
    verifiers=[{"type": "llm_judge", "mode": "final", "threshold": 4}],
    scoring_rule="threshold",
    source_platform="huggingface",
)

Record(
    benchmark="gdpval",
    task_id="1e5a1d7f-12c1-48c6-afd9-82257b3f2409",
    agent_id="gemini-2-5-flash+inspect_ai",
    agent_model="gemini-2-5-flash",
    agent_scaffold="inspect_ai",
    trial=0,
    result=0,
    metadata={
        "num_turns": 1.0,
        "input_tokens": 470.0,
        "output_tokens": 131.0,
    },
    source_platform="github",
    data_provider="bridge",
)
```

## References

- Imported results: [BRIDGE](https://github.com/McGill-NLP/BRIDGE)
- Original benchmark: [GDPval](https://huggingface.co/datasets/openai/gdpval)
