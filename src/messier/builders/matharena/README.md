# MathArena Description

[MathArena](https://matharena.ai/) evaluates language models on mathematics competition problems. We import 352 short-answer and proof problems from 13 competitions, together with up to four responses for each available model-problem pair. The source conversations are retained as trajectories.

Short-answer problems have a binary result based on the extracted final answer. For proof problems, each judge grades the submitted proof against a problem-specific marking scheme, such as `3/7` and `4/7`, and MathArena averages the normalized grades across available judges. We retain this average as the verifier result, ready for downstream usage. Note that we assume that a proof is successful only when all available judges award full credit.

| | Short answer | Proof |
| --- | --- | --- |
| Task and environment | A competition problem requesting a final answer | A competition problem requesting a written proof |
| Environment state | None. The model receives no feedback after the problem is presented | None. The model receives no feedback after the problem is presented |
| Action space | Text responses | Text responses |
| Verifier | Extracts the final answer and compares it with the gold | One or two human or LLM judges grade the proof against a marking scheme |
| Scoring rule | The binary output from the verifier is used directly | Threshold introduced by MESSIER. A proof counts as successful only when every judge gives it full marks |
| Gold answer | The expected final answer | A problem-specific marking scheme used by the judges |

The following examples show one short-answer task and its result, followed by the verifier and trial results stored for one proof response. Long text and metadata are shortened for readability.

```python
Task(
    benchmark="matharena",
    task_id="aime_2025.1",
    environment_id="aime_2025",
    environment_description=(
        "American Invitational Mathematics Examination (AIME) 2025 - "
        "a mathematics problem requiring a short numerical answer."
    ),
    task_description=(
        "Find the sum of all integer bases b>9 for which 17_b is a "
        "divisor of 97_b."
    ),
    environment_metadata={
        "action_space": {
            "types": ["text"],
            "description": (
                "Text responses containing a short answer or written "
                "mathematical proof."
            ),
            "format": "exact_match",
        },
        "environment_state": {"type": "none", "access": "none"},
        "competition": "aime_2025",
        "competition_date": "2025-02-06",
    },
    gold_answer="70",
    task_metadata={"problem_idx": "1", "kind": "binary"},
    task_date="2025-02-06",
    verifiers=[{"type": "exact_match", "mode": "final"}],
    scoring_rule="direct",
    source_platform="huggingface",
)

Record(
    benchmark="matharena",
    task_id="aime_2025.1",
    record_type="trial_result",
    agent_id="gpt-5-nano (high)",
    agent_model="gpt-5-nano",
    model_reasoning_effort="high",
    agent_scaffold=None,
    trial=1,
    result=1,
    metadata={
        "raw_model": "GPT-5-nano (high)",
        "model_config": "openai/gpt-5-nano",
    },
    source_platform="huggingface",
)

Record(
    benchmark="matharena",
    task_id="imo_2025.4",
    verifier_id="imo_2025.4::verifier:0",
    record_type="verifier_result",
    agent_id="o3 (high)",
    agent_model="o3",
    model_reasoning_effort="high",
    agent_scaffold=None,
    trial=3,
    result=0.5,
    source_platform="huggingface",
)

Record(
    benchmark="matharena",
    task_id="imo_2025.4",
    record_type="trial_result",
    agent_id="o3 (high)",
    agent_model="o3",
    model_reasoning_effort="high",
    agent_scaffold=None,
    trial=3,
    result=0,
    source_platform="huggingface",
)
```

### AIME 2025 (`aime_2025.1`) example

**Agent** = **Model:** `gpt-5-nano` + **Reasoning effort:** `high` + **Scaffold:** `None`

| | Interaction |
| --- | --- |
| 👤 **User** | Find the sum of all integer bases `b>9` for which `17_b` is a divisor of `97_b`. Put the final answer within `\\boxed{}`. |
| 🤖 **Agent** | `\\boxed{70}` |
| 🟥 **Trial result** | `1`, because the extracted answer `70` equals the expected answer `70`. |

### USAMO 2025 (`usamo_2025.1`) proof example

**Agent** = **Model:** `qwq-32b` + **Scaffold:** `None`

The proof response is shortened to three exact excerpts.

| | Interaction |
| --- | --- |
| 👤 **User** | Let `k` and `d` be positive integers. Prove that there exists a positive integer `N` such that, for every odd integer `n>N`, all digits of `n^k` in base `2n` are greater than `d`. |
| 🤖 **Agent** | "Any number `n^k` can be expressed in base-`2n` as `n^k = sum a_i(2n)^i`." |
| 🤖 **Agent** | "The least significant digit `a_0` is `n^k mod 2n`. Since `n` is odd, `n^k mod 2n = n`." |
| 🤖 **Agent** | "Taking `N` as the maximum of these `N_i` values ensures all digits `a_i` are greater than `d` for `n>N`." |
| **Judge 1** | `5/7` |
| **Judge 2** | `5/7` |
| **Verifier result** | `5/7`, the average normalized grade. |
| 🟥 **Trial result** | `0`, because our normalization requires full credit for success. |

## References

- Imported results: [MathArena model outputs](https://huggingface.co/collections/MathArena/matharena-outputs)
- Original benchmark: [MathArena](https://github.com/eth-sri/matharena) and its [paper](https://arxiv.org/abs/2505.23281)
- Source license: The imported MathArena datasets are released under [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/).

### Stanford Math Tournament 2025 attribution

These 53 MathArena task statements originate from the [official SMT 2025 tests](https://www.stanfordmathtournament.org/past-tests/SMT/2025). Stanford Math Tournament permits use of officially released problems with year, round, and problem-number attribution in its [reuse FAQ](https://www.stanfordmathtournament.org/competitions/smt-2026). The table supplies that attribution for every included task.

The MESSIER task IDs and MathArena `problem_idx` values are internal indices, not the original SMT problem numbers. In particular, the Team tasks omit original questions 1 and 3; task `smt_2025.40` is Team problem 12.

| MESSIER task ID | Attribution | Original problem |
| --- | --- | --- |
| `smt_2025.1` | Stanford Math Tournament 2025 Calculus 1 | [PDF page 1](https://www.stanfordmathtournament.org/pdfs/smt2025/calculus-problems.pdf#page=1) |
| `smt_2025.2` | Stanford Math Tournament 2025 Calculus 2 | [PDF page 1](https://www.stanfordmathtournament.org/pdfs/smt2025/calculus-problems.pdf#page=1) |
| `smt_2025.3` | Stanford Math Tournament 2025 Calculus 3 | [PDF page 1](https://www.stanfordmathtournament.org/pdfs/smt2025/calculus-problems.pdf#page=1) |
| `smt_2025.4` | Stanford Math Tournament 2025 Calculus 4 | [PDF page 1](https://www.stanfordmathtournament.org/pdfs/smt2025/calculus-problems.pdf#page=1) |
| `smt_2025.5` | Stanford Math Tournament 2025 Calculus 5 | [PDF page 1](https://www.stanfordmathtournament.org/pdfs/smt2025/calculus-problems.pdf#page=1) |
| `smt_2025.6` | Stanford Math Tournament 2025 Calculus 6 | [PDF page 1](https://www.stanfordmathtournament.org/pdfs/smt2025/calculus-problems.pdf#page=1) |
| `smt_2025.7` | Stanford Math Tournament 2025 Calculus 7 | [PDF page 1](https://www.stanfordmathtournament.org/pdfs/smt2025/calculus-problems.pdf#page=1) |
| `smt_2025.8` | Stanford Math Tournament 2025 Calculus 8 | [PDF page 2](https://www.stanfordmathtournament.org/pdfs/smt2025/calculus-problems.pdf#page=2) |
| `smt_2025.9` | Stanford Math Tournament 2025 Calculus 9 | [PDF page 2](https://www.stanfordmathtournament.org/pdfs/smt2025/calculus-problems.pdf#page=2) |
| `smt_2025.10` | Stanford Math Tournament 2025 Calculus 10 | [PDF page 2](https://www.stanfordmathtournament.org/pdfs/smt2025/calculus-problems.pdf#page=2) |
| `smt_2025.11` | Stanford Math Tournament 2025 Discrete 1 | [PDF page 1](https://www.stanfordmathtournament.org/pdfs/smt2025/discrete-problems.pdf#page=1) |
| `smt_2025.12` | Stanford Math Tournament 2025 Discrete 2 | [PDF page 1](https://www.stanfordmathtournament.org/pdfs/smt2025/discrete-problems.pdf#page=1) |
| `smt_2025.13` | Stanford Math Tournament 2025 Discrete 3 | [PDF page 1](https://www.stanfordmathtournament.org/pdfs/smt2025/discrete-problems.pdf#page=1) |
| `smt_2025.14` | Stanford Math Tournament 2025 Discrete 4 | [PDF page 1](https://www.stanfordmathtournament.org/pdfs/smt2025/discrete-problems.pdf#page=1) |
| `smt_2025.15` | Stanford Math Tournament 2025 Discrete 5 | [PDF page 1](https://www.stanfordmathtournament.org/pdfs/smt2025/discrete-problems.pdf#page=1) |
| `smt_2025.16` | Stanford Math Tournament 2025 Discrete 6 | [PDF page 1](https://www.stanfordmathtournament.org/pdfs/smt2025/discrete-problems.pdf#page=1) |
| `smt_2025.17` | Stanford Math Tournament 2025 Discrete 7 | [PDF page 1](https://www.stanfordmathtournament.org/pdfs/smt2025/discrete-problems.pdf#page=1) |
| `smt_2025.18` | Stanford Math Tournament 2025 Discrete 8 | [PDF page 1](https://www.stanfordmathtournament.org/pdfs/smt2025/discrete-problems.pdf#page=1) |
| `smt_2025.19` | Stanford Math Tournament 2025 Discrete 9 | [PDF page 1](https://www.stanfordmathtournament.org/pdfs/smt2025/discrete-problems.pdf#page=1) |
| `smt_2025.20` | Stanford Math Tournament 2025 Discrete 10 | [PDF page 2](https://www.stanfordmathtournament.org/pdfs/smt2025/discrete-problems.pdf#page=2) |
| `smt_2025.21` | Stanford Math Tournament 2025 Geometry 1 | [PDF page 1](https://www.stanfordmathtournament.org/pdfs/smt2025/geometry-problems.pdf#page=1) |
| `smt_2025.22` | Stanford Math Tournament 2025 Geometry 2 | [PDF page 1](https://www.stanfordmathtournament.org/pdfs/smt2025/geometry-problems.pdf#page=1) |
| `smt_2025.23` | Stanford Math Tournament 2025 Geometry 3 | [PDF page 1](https://www.stanfordmathtournament.org/pdfs/smt2025/geometry-problems.pdf#page=1) |
| `smt_2025.24` | Stanford Math Tournament 2025 Geometry 4 | [PDF page 1](https://www.stanfordmathtournament.org/pdfs/smt2025/geometry-problems.pdf#page=1) |
| `smt_2025.25` | Stanford Math Tournament 2025 Geometry 5 | [PDF page 1](https://www.stanfordmathtournament.org/pdfs/smt2025/geometry-problems.pdf#page=1) |
| `smt_2025.26` | Stanford Math Tournament 2025 Geometry 6 | [PDF page 1](https://www.stanfordmathtournament.org/pdfs/smt2025/geometry-problems.pdf#page=1) |
| `smt_2025.27` | Stanford Math Tournament 2025 Geometry 7 | [PDF page 1](https://www.stanfordmathtournament.org/pdfs/smt2025/geometry-problems.pdf#page=1) |
| `smt_2025.28` | Stanford Math Tournament 2025 Geometry 8 | [PDF page 1](https://www.stanfordmathtournament.org/pdfs/smt2025/geometry-problems.pdf#page=1) |
| `smt_2025.29` | Stanford Math Tournament 2025 Geometry 9 | [PDF page 1](https://www.stanfordmathtournament.org/pdfs/smt2025/geometry-problems.pdf#page=1) |
| `smt_2025.30` | Stanford Math Tournament 2025 Geometry 10 | [PDF page 2](https://www.stanfordmathtournament.org/pdfs/smt2025/geometry-problems.pdf#page=2) |
| `smt_2025.31` | Stanford Math Tournament 2025 Team 2 | [PDF page 1](https://www.stanfordmathtournament.org/pdfs/smt2025/team-problems.pdf#page=1) |
| `smt_2025.32` | Stanford Math Tournament 2025 Team 4 | [PDF page 1](https://www.stanfordmathtournament.org/pdfs/smt2025/team-problems.pdf#page=1) |
| `smt_2025.33` | Stanford Math Tournament 2025 Team 5 | [PDF page 1](https://www.stanfordmathtournament.org/pdfs/smt2025/team-problems.pdf#page=1) |
| `smt_2025.34` | Stanford Math Tournament 2025 Team 6 | [PDF page 1](https://www.stanfordmathtournament.org/pdfs/smt2025/team-problems.pdf#page=1) |
| `smt_2025.35` | Stanford Math Tournament 2025 Team 7 | [PDF page 1](https://www.stanfordmathtournament.org/pdfs/smt2025/team-problems.pdf#page=1) |
| `smt_2025.36` | Stanford Math Tournament 2025 Team 8 | [PDF page 1](https://www.stanfordmathtournament.org/pdfs/smt2025/team-problems.pdf#page=1) |
| `smt_2025.37` | Stanford Math Tournament 2025 Team 9 | [PDF page 2](https://www.stanfordmathtournament.org/pdfs/smt2025/team-problems.pdf#page=2) |
| `smt_2025.38` | Stanford Math Tournament 2025 Team 10 | [PDF page 2](https://www.stanfordmathtournament.org/pdfs/smt2025/team-problems.pdf#page=2) |
| `smt_2025.39` | Stanford Math Tournament 2025 Team 11 | [PDF page 2](https://www.stanfordmathtournament.org/pdfs/smt2025/team-problems.pdf#page=2) |
| `smt_2025.40` | Stanford Math Tournament 2025 Team 12 | [PDF page 2](https://www.stanfordmathtournament.org/pdfs/smt2025/team-problems.pdf#page=2) |
| `smt_2025.41` | Stanford Math Tournament 2025 Team 13 | [PDF page 2](https://www.stanfordmathtournament.org/pdfs/smt2025/team-problems.pdf#page=2) |
| `smt_2025.42` | Stanford Math Tournament 2025 Team 14 | [PDF page 2](https://www.stanfordmathtournament.org/pdfs/smt2025/team-problems.pdf#page=2) |
| `smt_2025.43` | Stanford Math Tournament 2025 Team 15 | [PDF page 2](https://www.stanfordmathtournament.org/pdfs/smt2025/team-problems.pdf#page=2) |
| `smt_2025.44` | Stanford Math Tournament 2025 Algebra 1 | [PDF page 1](https://www.stanfordmathtournament.org/pdfs/smt2025/algebra-problems.pdf#page=1) |
| `smt_2025.45` | Stanford Math Tournament 2025 Algebra 2 | [PDF page 1](https://www.stanfordmathtournament.org/pdfs/smt2025/algebra-problems.pdf#page=1) |
| `smt_2025.46` | Stanford Math Tournament 2025 Algebra 3 | [PDF page 1](https://www.stanfordmathtournament.org/pdfs/smt2025/algebra-problems.pdf#page=1) |
| `smt_2025.47` | Stanford Math Tournament 2025 Algebra 4 | [PDF page 1](https://www.stanfordmathtournament.org/pdfs/smt2025/algebra-problems.pdf#page=1) |
| `smt_2025.48` | Stanford Math Tournament 2025 Algebra 5 | [PDF page 1](https://www.stanfordmathtournament.org/pdfs/smt2025/algebra-problems.pdf#page=1) |
| `smt_2025.49` | Stanford Math Tournament 2025 Algebra 6 | [PDF page 1](https://www.stanfordmathtournament.org/pdfs/smt2025/algebra-problems.pdf#page=1) |
| `smt_2025.50` | Stanford Math Tournament 2025 Algebra 7 | [PDF page 1](https://www.stanfordmathtournament.org/pdfs/smt2025/algebra-problems.pdf#page=1) |
| `smt_2025.51` | Stanford Math Tournament 2025 Algebra 8 | [PDF page 1](https://www.stanfordmathtournament.org/pdfs/smt2025/algebra-problems.pdf#page=1) |
| `smt_2025.52` | Stanford Math Tournament 2025 Algebra 9 | [PDF page 1](https://www.stanfordmathtournament.org/pdfs/smt2025/algebra-problems.pdf#page=1) |
| `smt_2025.53` | Stanford Math Tournament 2025 Algebra 10 | [PDF page 1](https://www.stanfordmathtournament.org/pdfs/smt2025/algebra-problems.pdf#page=1) |
