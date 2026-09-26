import re

# source identifiers from Agent Psychometrics begin with a date
AP_SUBJECT_RE = re.compile(r"^(?P<date>\d{8})_(?P<rest>.+)$")

# the CoderForge source exposes test output for each trajectory
CODERFORGE_REPO = "togethercomputer/CoderForge-Preview-32B-SWE-Bench-Verified-Evaluation-trajectories"
CODERFORGE_FILES = tuple(f"trajectory/train-{index:05d}-of-00032.parquet" for index in range(32))
CODERFORGE_MODEL = "qwen3-coder-32b"
CODERFORGE_SCAFFOLD = "openhands"

# source test output may use pytest or unittest syntax
PYTEST_RESULT_RE   = re.compile(r"^(PASSED|FAILED|ERROR|SKIPPED)\s+(\S+)", re.MULTILINE)
UNITTEST_RESULT_RE = re.compile(r"^(test_\S+) \(([^)]+)\)[^\n]*?\.\.\.\s*(ok|FAIL|ERROR|skipped)", re.MULTILINE)

LIVESWEAGENT_TRAJ_SOURCES = [
    ("livesweagent_claude_sonnet_4_5", "livesweagent/claude-sonnet-4-5_swebench_verified_traj", "claude-sonnet-4-5", "livesweagent", "eval_result.json"),
    ("livesweagent_gemini_3_pro",      "livesweagent/gemini_3_pro_swebench_verified_traj",          "gemini-3-pro",        "livesweagent", "result.json"),
    ("livesweagent_gpt_5",             "livesweagent/gpt-5_swebench_verified_traj",                  "gpt-5",               "livesweagent", "eval_result.json"),
    ("livesweagent_gpt_5_mini",        "livesweagent/gpt-5-mini_swebench_verified_traj",             "gpt-5-mini",          "livesweagent", "eval_result.json"),
]

# the first model-family token separates scaffold and model names
LLM_PREFIX = re.compile(
    r"^(claude|opus|sonnet|haiku|gpt|gemini|llama|qwen|kimi|o\d|mistral|grok|"
    r"deepseek|phi|doubao|swellama|codex|flamingo|lingma|glm|nova|devstral)",
    re.IGNORECASE,
)
VERSION_TOKEN = re.compile(
    r"^(\d\S*|v\d\S*|k\d+|"
    r"opus|sonnet|haiku|mini|nano|flash|pro|turbo|preview|instruct|chat|"
    r"coder|seed|code|base|max|fast|exp|experimental|small|medium|large|"
    r"reasoning|non-reasoning|thinking|think|premier)$",
    re.IGNORECASE,
)

# identifiers that cannot be split reliably
REST_OVERRIDES: dict[str, tuple[str | None, str]] = {
    "qodo_command":                    ("qodo-command", "claude-4"),
    "lingma-agent_lingma-swe-gpt-72b": ("lingma-agent", "lingma-swe-gpt-72b"),
    "lingma-agent_lingma-swe-gpt-7b":  ("lingma-agent", "lingma-swe-gpt-7b"),
    "navie-2-gpt4o-sonnet":            ("navie-2",      "multiple"),
    "SWE-Fixer_Qwen2.5-7b-retriever_Qwen2.5-72b-editor_20241128": ("SWE-Fixer", "multiple"),
    "SWE-Fixer_Qwen2.5-7b-retriever_Qwen2.5-72b-editor": ("SWE-Fixer", "multiple"),
    "Skywork-SWE-32B":         (None,      "Skywork-SWE-32B"),
    "Skywork-SWE-32B+TTS_Bo8": ("tts_bo8", "Skywork-SWE-32B"),
    "frogboss-32b":            (None,      "frogboss-32b"),
    "frogmini-14b":            (None,      "frogmini-14b"),
    "amazon.nova-premier-v1.0": ("amazon", "nova-premier-v1.0"),
}

# expand BRIDGE difficulty centers into ranges
BUCKET_BANDS = {
    3.872983346207417: (1.0,    3.87, 15.0),
    30.0:              (15.0,  30.0,  60.0),
    120.0:             (60.0, 120.0, 240.0),
    480.0:             (240.0, 480.0, 960.0),
}
