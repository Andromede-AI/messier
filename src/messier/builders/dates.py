"""
Resolve benchmark, task, and model release dates. Relies on the centralized mappings in `agents.py` and `benchmarks.py`.
"""

import csv
import json
import re
from datetime import date, datetime
from functools import lru_cache
from typing import Any
import yaml
from ..io import RAW
from .agents import DATE_OVERRIDES, EXTRA_MODEL_DATES
from .benchmarks import BENCHMARK_RELEASE_DATES

# match the date format used in CyBench task headers
HTB_DATE_RE = re.compile(
    r"(\d{1,2})\s*(?:\$\^\{[a-z]+\}\$|[a-z]{0,2})\s+"
    r"(January|February|March|April|May|June|July|August|September|October|November|December)"
    r"\s+(\d{4})",
    re.IGNORECASE,
)
MONTHS = {
    month: index + 1
    for index, month in enumerate(
        [
            "january",
            "february",
            "march",
            "april",
            "may",
            "june",
            "july",
            "august",
            "september",
            "october",
            "november",
            "december",
        ]
    )
}


def to_iso_date(value: Any) -> str | None:
    """
    Return a date in ``YYYY-MM-DD`` form.

    :param value: String, date, datetime, or eight-digit ``YYYYMMDD`` value.
    :return: Normalized date, or ``None`` when the value cannot be parsed.
    """
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()

    try:
        return datetime.fromisoformat(str(value).strip()).date().isoformat()
    except ValueError:
        return None


def _normalize_key(name: str) -> str:
    name = re.sub(
        r"\s+\((?:reasoning|max|xhigh|high|medium|low)\)$",
        "",
        name,
        flags=re.IGNORECASE,
    )
    for ch in "./ -":
        name = name.replace(ch, "_")
    return "_".join(filter(None, name.lower().split("_")))


@lru_cache(maxsize=1)
def _load_date_lookup() -> dict[str, str]:
    # retain the first date found across sources
    dates_by_model: dict[str, str] = {}

    def add(key: str, raw_date: Any) -> None:
        # normalize keys before combining sources
        date_value = to_iso_date(raw_date)
        if date_value:
            dates_by_model.setdefault(_normalize_key(key), date_value)

    epoch = RAW / "epoch" / "notable_ai_models.csv"
    if epoch.exists():
        with epoch.open() as f:
            for row in csv.DictReader(f):
                name = (row.get("Model") or "").strip()
                pub = (row.get("Publication date") or "").strip()
                if name and pub:
                    add(name, pub)

    metr = RAW / "metr" / "release_dates.yaml"
    if metr.exists():
        release_dates = yaml.safe_load(metr.read_text()) or {}
        model_dates = release_dates.get("date", {})
        for model_name, model_date in model_dates.items():
            add(model_name, model_date)

    bridge = RAW / "bridge" / "model_run_mapping.json"
    if bridge.exists():
        model_runs = json.loads(bridge.read_text())
        for model_name, info in model_runs.items():
            release_date = info.get("release_date")
            if not release_date:
                continue

            add(model_name, release_date)
            if info.get("alias"):
                add(info["alias"], release_date)
            for run_id in info.get("run_ids") or []:
                add(run_id, release_date)

    tb = RAW / "agent_psychometrics" / "terminalbench" / "model_release_dates.json"
    if tb.exists():
        model_dates = json.loads(tb.read_text())
        for model_name, model_date in model_dates.items():
            add(model_name, model_date)

    for name in ("swebench_pro", "gso"):
        path = RAW / "agent_psychometrics" / name / "agent_dates.json"
        if path.exists():
            model_dates = json.loads(path.read_text())
            for model_name, model_date in model_dates.items():
                add(model_name, model_date)

    for model_name, model_date in EXTRA_MODEL_DATES.items():
        add(model_name, model_date)

    # date overrides take precedence when an upstream source disagrees
    for model_name, model_date in DATE_OVERRIDES.items():
        date_value = to_iso_date(model_date)
        if date_value:
            dates_by_model[_normalize_key(model_name)] = date_value

    return dates_by_model


DATE_PREFIX_RE = re.compile(r"^(\d{4})-?(\d{2})-?(\d{2})[-_]")
DATE_SUFFIX_RE = re.compile(r"[-_](\d{4})-?(\d{2})-?(\d{2})$")


def _parse_date_prefix(name: str) -> str | None:
    match = DATE_PREFIX_RE.match(name)
    if not match:
        match = DATE_SUFFIX_RE.search(name)
    if not match:
        return None

    return f"{match.group(1)}-{match.group(2)}-{match.group(3)}"


def resolve_model_date(*candidates: str | None) -> str | None:
    """
    Find the release date of the first recognized model name.

    :param candidates: Possible names for the same model.
    :return: Release date, or ``None`` when no name is recognized.
    """
    if candidates and candidates[0] in {"multiple", "undisclosed"}:
        return None

    date_lookup = _load_date_lookup()
    for candidate in candidates:
        if not candidate:
            continue

        normalized = _normalize_key(candidate)
        if normalized in date_lookup:
            return date_lookup[normalized]

    for candidate in candidates:
        if not candidate:
            continue
        parsed_date = _parse_date_prefix(candidate)
        if parsed_date:
            return parsed_date
    return None


def resolve_task_date(benchmark: str, per_task_date: Any = None) -> str | None:
    """
    Find the date associated with a task.

    :param benchmark: Benchmark identifier.
    :param per_task_date: Date supplied for the individual task.
    :return: Task date when available, otherwise the benchmark release date.
    """
    specific = to_iso_date(per_task_date)
    if specific:
        return specific
    return BENCHMARK_RELEASE_DATES.get(benchmark)


def parse_htb_readme_date(readme_text: str | None) -> str | None:
    if not readme_text:
        return None

    match = HTB_DATE_RE.search(readme_text)
    if not match:
        return None

    day = int(match.group(1))
    month = match.group(2).lower()
    year = int(match.group(3))
    return f"{year:04d}-{MONTHS[month]:02d}-{day:02d}"
