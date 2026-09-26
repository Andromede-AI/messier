import re

ENVIRONMENT_DESCRIPTION = (
    "A company workspace containing internal services for source control, project management, "
    "file sharing, and team communication, together with the files and data used by the task."
)

# files stored in dedicated task fields
HANDLED_FILES = {
    "task.md",
    "checkpoints.md",
    "evaluator.py",
    "dependencies.yml",
    "Dockerfile",
    "Makefile",
}

# filenames that identify expected outputs
GOLD_FILE_GLOBS = ("golden_*", "*_golden.*", "solution.*", "expected_*")

# text formats stored inline
TEXT_EXTS = {
    ".py",
    ".md",
    ".yml",
    ".yaml",
    ".sh",
    ".sql",
    ".txt",
    ".csv",
    ".json",
    ".toml",
    ".cfg",
    ".ini",
    ".html",
    ".css",
    ".js",
    ".ts",
    ".go",
    ".java",
    ".cpp",
    ".c",
    ".h",
    ".hpp",
    ".rb",
    ".pl",
    ".rmd",
}
MAX_INLINE_BYTES = 256_000

# model and scaffold overrides for nonstandard submission names
SUBMISSION_OVERRIDES: dict[str, tuple[str, str | None]] = {
    "20250510_OWL-RolePlay-gpt-4o-o3-mini": ("owl-roleplay-gpt-4o-o3-mini", "multiple"),
    "20250614_OpenHands-Versa-claude-3.7-sonnet": ("OpenHands-Versa", "claude-3.7-sonnet"),
    "20250614_OpenHands-Versa-claude-sonnet-4": ("OpenHands-Versa", "claude-sonnet-4"),
    "20250729_OpenHands-0.28.1-claude37-gpt4o": ("openhands-claude37-gpt4o", "multiple"),
    "20250729_OpenHands-0.28.1-claude37-deepseekv3": ("openhands-claude37-deepseekv3", "multiple"),
    "20251013_MUSE-gemini-2.5-flash": ("MUSE", "gemini-2.5-flash"),
    "20251110_TTE-MatrixAgent-Deepseek-V3.2": ("TTE-MatrixAgent", "Deepseek-V3.2"),
}

OPENHANDS_RE = re.compile(r"^\d{8}_OpenHands-\d+(?:\.\d+)+-(.+)$")
CHECKPOINT_RE = re.compile(
    r"^##\s+(?:(?:Checkpoin(?:t|g)\s+(\d+|Final))|(?:Final\s+Checkpoint))"
    r"(?:\s*:[^(]+)?\s*\((\d+)\s*(?:pts?|points?)\)\s*$",
    re.IGNORECASE | re.MULTILINE,
)
RESULT_FILE_RE = re.compile(r"^eval_(.+?)(?:-image)?\.json$")
SUB_DATE_RE = re.compile(r"^(\d{8})_")
