import re

# use the source revisions whose task text matches the reported results
PRIMARY_REVISION  = "44d0dce62fd3"
FALLBACK_REVISION = "24f7af818c0b"

TASK_ID_SUFFIX_RE = re.compile(r"_\d{6}$")

# scaffold, model, and fallback release date from the source paper
AGENTS: dict[str, tuple[str | None, str | None, str | None]] = {
    "Operator":                    ("openai-operator",        None,                "2025-01-23"),
    "Agent-E":                     ("agent-e",                "gpt-4o",            None),
    "Browser_Use":                 ("browser-use",            "gpt-4o",            None),
    "Claude_Computer_Use_3.5":     ("claude-computer-use",  "claude-3-5-sonnet", None),
    "Claude_Computer_Use_3.7":     ("claude-computer-use",  "claude-3-7-sonnet", None),
    "SeeAct":                      ("see-act",                "gpt-4o",            None),
    "ACT-1-20250703":              (None,                     "act-1-20250703",    "2025-07-03"),
    "ACT-1-20250814":              (None,                     "act-1-20250814",    "2025-08-14"),
    "Google_Computer_Use_09-2025": ("google-computer-use",    None,                "2025-09-01"),
    "Navigator":                   ("navigator",              None,                None),
}

ACTION_SPACE_DESCRIPTION = (
    "Live website operated through a browser. Page state is observed via the DOM, "
    "rendered screenshots, and the accessibility tree. Actions include click, type, "
    "scroll, hover, key-press, navigate, tab/window switch, file upload, and JavaScript "
    "execution, dispatched against DOM elements, pixel coordinates, or accessibility-tree "
    "node references. The agent may also return a final text response."
)
