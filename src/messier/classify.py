"""
Assign one occupation and industry label to each task.

GDPval provides these labels directly. For every other benchmark, three
models independently select a 2018 SOC occupation group and a 2022 NAICS
sector. Agreement between at least two models determines each label. A fourth
model resolves labels without a majority. Responses are cached using the task,
prompt, and model so rerunning the script calls a model only for new or changed
inputs. The final classifications are written to ``classifications.jsonl``.
Run the dataset builder before this module so it can read the current tasks,
then run the builder again to attach the resulting labels to ``tasks.jsonl``.
"""

import asyncio
import hashlib
import json
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
import litellm
from dotenv import load_dotenv
from .io import ARTIFACTS, CLASSIFICATIONS_PATH, TASKS_PATH
from .taxonomy import ADJUDICATOR_PROMPT, VOTER_PROMPT, Classification

load_dotenv(Path(__file__).resolve().parents[2] / ".env")


@dataclass(frozen=True)
class ClassificationRequest:
    benchmark: str
    task_id: str
    message: str
    content_hash: str

    @property
    def key(self) -> tuple[str, str]:
        return self.benchmark, self.task_id


# gdpval provides exact occupation and industry labels for each task
SOURCE_LABELED_BENCHMARKS = {"gdpval"}

VOTERS = {
    "haiku": ("anthropic/claude-haiku-4-5", ARTIFACTS / "classifications_haiku.jsonl"),
    "sonnet": ("anthropic/claude-sonnet-4-6", ARTIFACTS / "classifications_sonnet.jsonl"),
    "gpt5_mini": ("openai/gpt-5-mini", ARTIFACTS / "classifications_gpt5mini.jsonl"),
}

ADJUDICATOR_MODEL = "openai/gpt-5"
ADJUDICATION_CACHE = ARTIFACTS / "_adjudications.jsonl"
CONCURRENCY = 16
ADJUDICATION_CONCURRENCY = 8
MAX_RETRIES = 3


async def run() -> None:
    tasks = load_rows(TASKS_PATH)
    tasks_to_classify = []
    for task in tasks.values():
        if task["benchmark"] in SOURCE_LABELED_BENCHMARKS:
            continue

        tasks_to_classify.append(task)

    for name, (model, cache_path) in VOTERS.items():
        requests = [voter_request(task, model) for task in tasks_to_classify]
        await run_batch(
            requests=requests,
            cache_path=cache_path,
            model=model,
            prompt=VOTER_PROMPT,
            concurrency=CONCURRENCY,
            label=f"voter:{name}",
        )

    await merge(tasks)


async def merge(tasks: dict[tuple[str, str], dict]) -> None:
    voter_caches = {
        name: load_rows(cache_path)
        for name, (_, cache_path) in VOTERS.items()
    }
    keys = sorted(
        key
        for key, task in tasks.items()
        if task["benchmark"] not in SOURCE_LABELED_BENCHMARKS
    )

    print(f"\n[merge] {len(keys)} classifications")
    proposed = {}
    adjudication_requests = []

    for key in keys:
        task = tasks[key]
        picks = []
        for name, (model, _) in VOTERS.items():
            request = voter_request(task, model)
            pick = voter_caches[name].get(key)
            if pick and pick.get("content_hash") == request.content_hash:
                picks.append(pick)
            else:
                picks.append(None)

        if sum(pick is not None for pick in picks) < 2:
            raise RuntimeError(f"{key[0]}/{key[1]} has fewer than two current classifications")

        soc_code = majority([pick.get("soc_code") if pick else None for pick in picks])
        naics_code = majority([pick.get("naics_code") if pick else None for pick in picks])
        split_axes = []
        if soc_code is None:
            split_axes.append("SOC")
        if naics_code is None:
            split_axes.append("NAICS")

        proposed[key] = {
            "soc_code": soc_code,
            "naics_code": naics_code,
            "votes": {
                name: {
                    "soc_code": pick["soc_code"],
                    "naics_code": pick["naics_code"],
                    "rationale": pick["rationale"],
                    "confidence": pick["confidence"],
                    "model": pick["model"],
                }
                if pick
                else None
                for name, pick in zip(VOTERS, picks)
            },
        }

        if split_axes:
            lines = []
            for (_, (model, _)), pick in zip(VOTERS.items(), picks):
                if pick is None:
                    lines.append(f"  {model}: no classification")
                    continue

                lines.append(
                    f"  {model}: SOC={pick.get('soc_code')}   "
                    f"NAICS={pick.get('naics_code')}\n"
                    f"      rationale: {pick.get('rationale', '')}"
                )

            message = (
                f"TASK:\n{json.dumps(task_payload(task), indent=2, default=str)}\n\n"
                f"THREE CLASSIFIERS' PICKS (they split on {' and '.join(split_axes)}):\n"
                + "\n".join(lines)
                + "\n\nPick the final SOC and NAICS. Override the votes if all three picks are wrong."
            )
            adjudication_requests.append(
                ClassificationRequest(
                    benchmark=key[0],
                    task_id=key[1],
                    message=message,
                    content_hash=content_hash(
                        {
                            "model": ADJUDICATOR_MODEL,
                            "system": ADJUDICATOR_PROMPT,
                            "task": message,
                        }
                    ),
                )
            )

    # adjudicate disagreements and retain only current cache entries
    adjudication_cache = await run_batch(
        requests=adjudication_requests,
        cache_path=ADJUDICATION_CACHE,
        model=ADJUDICATOR_MODEL,
        prompt=ADJUDICATOR_PROMPT,
        concurrency=ADJUDICATION_CONCURRENCY,
        label="adjudicate",
    )
    adjudications = {
        request.key: adjudication_cache[request.key]
        for request in adjudication_requests
        if adjudication_cache.get(request.key, {}).get("content_hash")
        == request.content_hash
    }

    # write one final classification for every task
    rows = []
    for key in keys:
        classification = proposed[key]
        adjudication = adjudications.get(key)
        soc_code = classification["soc_code"]
        naics_code = classification["naics_code"]

        if soc_code is None and adjudication:
            soc_code = adjudication["soc_code"]
        if naics_code is None and adjudication:
            naics_code = adjudication["naics_code"]

        rows.append(
            {
                "benchmark": key[0],
                "task_id": key[1],
                "soc_code": soc_code,
                "naics_code": naics_code,
                "soc_source": "majority" if classification["soc_code"] else "adjudicated",
                "naics_source": "majority" if classification["naics_code"] else "adjudicated",
                "votes": classification["votes"],
                "adjudicator_model": adjudication.get("model") if adjudication else None,
                "adjudicator_rationale": adjudication.get("rationale") if adjudication else None,
                "adjudicator_confidence": adjudication.get("confidence") if adjudication else None,
            }
        )

    for key, task in sorted(tasks.items()):
        if key in proposed:
            continue

        rows.append(
            {
                "benchmark": key[0],
                "task_id": key[1],
                "soc_code": task.get("soc_code"),
                "naics_code": task.get("naics_code"),
                "soc_source": "source_metadata",
                "naics_source": "source_metadata",
                "votes": None,
                "adjudicator_model": None,
                "adjudicator_rationale": None,
                "adjudicator_confidence": None,
            }
        )

    missing = [
        (row["benchmark"], row["task_id"])
        for row in rows
        if row["soc_code"] is None or row["naics_code"] is None
    ]
    if missing:
        raise ValueError(f"{len(missing)} classifications remain unresolved")

    CLASSIFICATIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with CLASSIFICATIONS_PATH.open("w") as output:
        for row in rows:
            output.write(json.dumps(row) + "\n")

    source_counts = Counter(
        (row["soc_source"], row["naics_source"])
        for row in rows
    )

    print(f"\n[done] wrote {len(rows)} -> {CLASSIFICATIONS_PATH} (missing=0)")
    print("  source distribution (soc / naics):")
    for (soc_source, naics_source), count in source_counts.most_common():
        print(f"    {soc_source:18s} / {naics_source:18s} {count:5d}")


# helpers

def content_hash(value: str | dict) -> str:
    serialized = value if isinstance(value, str) else json.dumps(value, sort_keys=True, default=str)
    return hashlib.sha256(serialized.encode()).hexdigest()[:16]


def load_rows(path: Path) -> dict[tuple[str, str], dict]:
    if not path.exists():
        return {}

    with path.open() as file:
        return {
            (row["benchmark"], row["task_id"]): row
            for row in (json.loads(line) for line in file if line.strip())
        }


def task_payload(task: dict) -> dict:
    environment = task.get("environment_metadata") or {}
    action_space = environment.get("action_space") or {}
    environment_state = environment.get("environment_state") or {}

    return {
        "benchmark": task.get("benchmark"),
        "task_id": task.get("task_id"),
        "task_description": task.get("task_description") or "",
        "environment_description": task.get("environment_description"),
        "action_space": {
            "types": action_space.get("types"),
            "description": action_space.get("description"),
        },
        "environment_state": {
            "type": environment_state.get("type"),
            "access": environment_state.get("access"),
        },
    }


def voter_request(task: dict, model: str) -> ClassificationRequest:
    payload = task_payload(task)
    return ClassificationRequest(
        benchmark=task["benchmark"],
        task_id=task["task_id"],
        message=json.dumps(payload, indent=2, default=str),
        content_hash=content_hash(
            {
                "model": model,
                "system": VOTER_PROMPT,
                "task": payload,
            }
        ),
    )


def majority(values: list[str | None]) -> str | None:
    counts = Counter(value for value in values if value is not None)
    if not counts:
        return None

    value, count = counts.most_common(1)[0]
    return value if count >= 2 else None


async def call_model(model: str, prompt: str, message: str) -> Classification | None:
    for attempt in range(MAX_RETRIES):
        try:
            response = await litellm.acompletion(
                model=model,
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": message},
                ],
                response_format=Classification,
                max_tokens=2000,
            )
            return Classification.model_validate_json(response.choices[0].message.content)
        except Exception as error:
            if attempt + 1 == MAX_RETRIES:
                print(
                    f"[llm failed] {type(error).__name__}: {str(error)[:200]}",
                    file=sys.stderr,
                    flush=True,
                )
                return None
            await asyncio.sleep(2**attempt)


async def run_batch(
    requests: list[ClassificationRequest],
    cache_path: Path,
    model: str,
    prompt: str,
    concurrency: int,
    label: str,
) -> dict[tuple[str, str], dict]:

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache = load_rows(cache_path)
    pending = [
        request
        for request in requests
        if cache.get(request.key, {}).get("content_hash") != request.content_hash
    ]
    cache_hits = len(requests) - len(pending)
    print(
        f"[{label}] {len(pending)} / {len(requests)} "
        f"(cache hits={cache_hits}, model={model}, concurrency={concurrency})"
    )
    if not pending:
        return cache

    semaphore = asyncio.Semaphore(concurrency)

    async def classify(request: ClassificationRequest):
        async with semaphore:
            classification = await call_model(model, prompt, request.message)
        return request, classification

    jobs = [asyncio.create_task(classify(request)) for request in pending]
    with cache_path.open("a") as output:
        for completed, job in enumerate(asyncio.as_completed(jobs), start=1):
            request, classification = await job
            if classification is None:
                print(f"    [failed] {request.benchmark}/{request.task_id}", flush=True)
                continue

            row = {
                "benchmark": request.benchmark,
                "task_id": request.task_id,
                "content_hash": request.content_hash,
                "soc_code": classification.soc_code,
                "naics_code": classification.naics_code,
                "rationale": classification.rationale,
                "confidence": classification.confidence,
                "model": model,
            }
            output.write(json.dumps(row) + "\n")
            output.flush()

            if completed % 50 == 0:
                print(
                    f"    [{completed}/{len(pending)}] "
                    f"{request.benchmark}/{request.task_id} -> "
                    f"{classification.soc_code} / {classification.naics_code}",
                    flush=True,
                )

    return load_rows(cache_path)


if __name__ == "__main__":
    asyncio.run(run())
