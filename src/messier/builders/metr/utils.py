"""
Read METR task manifests, prompts, and scoring code.
"""

import json
import re
from pathlib import Path
from markdown_it import MarkdownIt
import yaml


def parse_family_readme(text: str) -> dict:
    """
    Extract task instructions and descriptions from a METR README.

    :param text: README contents.
    :return: Shared and task-specific text found in the document.
    """
    parser = MarkdownIt("commonmark")

    def sections_at_level(source: str, level: int) -> dict:
        lines = source.splitlines(keepends=True)
        tokens = parser.parse(source)
        headings = [
            (token, tokens[index + 1].content.strip())
            for index, token in enumerate(tokens)
            if token.type == "heading_open" and token.level == 0
            and int(token.tag[1:]) <= level
        ]
        sections = {}
        for index, (heading, title) in enumerate(headings):
            if heading.tag != f"h{level}":
                continue

            end = headings[index + 1][0].map[0] if index + 1 < len(headings) else len(lines)
            # keep the first section when a README repeats a heading
            sections.setdefault(title, "".join(lines[heading.map[1]:end]).strip())

        return sections

    sections = sections_at_level(text, 2)

    family_summary = (
        re.sub(
            r"^Version:[^\n]*\n+",
            "",
            sections.get("Family Summary", ""),
            flags=re.MULTILINE,
        ).strip()
        or None
    )
    summaries = sections_at_level(sections.get("Task Specific Summaries", ""), 3)
    instructions = sections_at_level(sections.get("Full Task Instructions", ""), 3)

    variants = {}
    for variant in set(summaries) | set(instructions):
        summary = summaries.get(variant, "")
        summary_lines = [
            line
            for line in summary.split("\n")
            if not line.lstrip().startswith("-")
        ]
        summary = "\n".join(summary_lines).strip() or None

        instruction = instructions.get(variant, "")
        instruction = next(
            (token.content for token in parser.parse(instruction) if token.type == "fence"),
            instruction,
        ).strip() or None

        variants[variant] = {
            "summary": summary,
            "instructions": instruction,
        }

    return {
        "family_summary": family_summary,
        "variants": variants
    }


def load_readmes(root: Path) -> dict:
    readmes = {}
    if not root.exists():
        return readmes

    for readme_path in root.rglob("README.md"):
        rel = readme_path.relative_to(root)
        # read only family-level descriptions
        if len(rel.parts) != 2:
            continue

        family = rel.parts[0]
        readmes[family] = parse_family_readme(readme_path.read_text())

    return readmes


def load_family_env(family_dir: Path) -> dict | None:
    """
    Read the files that describe a METR environment.

    :param family_dir: Directory containing one METR task family.
    :return: Available setup data, or ``None`` when none is provided.
    """
    build_steps_path = family_dir / "build_steps.json"
    reqs_path = family_dir / "requirements.txt"
    if not build_steps_path.exists() and not reqs_path.exists():
        return None

    environment = {}
    if build_steps_path.exists():
        environment["build_steps"] = json.loads(build_steps_path.read_text())

    if reqs_path.exists():
        environment["requirements"] = reqs_path.read_text()

    return environment or None


def load_manifests(root: Path) -> dict:
    """
    Read every task manifest below a METR source directory.

    :param root: Directory containing METR task families.
    :return: Task metadata indexed by family and task name.
    """
    manifests = {}
    if not root.exists():
        return manifests

    for manifest_path in root.rglob("manifest.yaml"):
        manifest = yaml.safe_load(manifest_path.read_text())
        if not isinstance(manifest, dict) or "tasks" not in manifest:
            continue

        family = manifest_path.parent.name
        family_meta = manifest.get("meta") or {}
        scoring_path = manifest_path.parent / f"{family}.py"
        script_code = scoring_path.read_text() if scoring_path.exists() else None
        family_env = load_family_env(manifest_path.parent)

        for task_name, task_info in (manifest.get("tasks") or {}).items():
            task_metadata = (task_info or {}).get("meta") or {}
            manifests[(family, task_name)] = {
                "description": task_metadata.get("task_description")
                or family_meta.get("task_description"),
                "family_name": family_meta.get("name", family),
                "family_expertise": family_meta.get("expertise"),
                "scoring": task_metadata.get("scoring") or {},
                "script_code": script_code,
                "resources": (task_info or {}).get("resources"),
                "version": manifest.get("version"),
                "family_env": family_env,
            }

    return manifests
