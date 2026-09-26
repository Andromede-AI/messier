import re

# source task IDs use "instance_<owner>__<repo>-<sha>-v<version>"
INSTANCE_SUFFIX_RE = re.compile(r"-v[\da-f]+$|-vnan$")

# source model IDs use "Claude 4 Sonnet - 10132025"
AP_SUBJECT_RE = re.compile(r"^(?P<model>.+?)\s*-\s*\d{8}$")

# sources with individual test results and a binary trial result
PER_TEST_TRAJ_SOURCES = [
    # (local_dir, hf_repo, model, scaffold)
    ("ii_agent_claude_sonnet_4_5",
     "Intelligent-Internet/swebench-pro-claude-sonnet-4.5-ii-agent-trajectories",
     "claude-sonnet-4-5", "ii-agent"),
    ("ii_agent_gpt_5_codex",
     "Intelligent-Internet/swebench-pro-gpt-5-codex-ii-agent-trajectories",
     "gpt-5-codex", "ii-agent"),
]

# sources with only a binary trial result
BINARY_TRAJ_SOURCES = [
    ("livesweagent_claude_sonnet_4_5",
     "livesweagent/claude-sonnet-4-5_swebench_pro_traj",
     "claude-sonnet-4-5", "livesweagent"),
]
