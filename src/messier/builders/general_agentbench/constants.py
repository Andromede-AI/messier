from ...models import ActionSpaceType, EnvironmentStateAccess, EnvironmentStateType, VerifierMode, VerifierType

# threshold used by the source to convert MCPBench's mean result to success
MCP_BINARY_THRESHOLD = 5.0
MCP_SCORE_TOTAL = 10.0 # maximum source verifier result

# numerical verifier results reported by MCPBench
MCP_DIMENSIONS = (
    "task_fulfillment",
    "grounding",
    "tool_appropriateness",
    "parameter_accuracy",
    "dependency_awareness",
    "parallelism_and_efficiency",
)


# map source files to benchmarks and optional domain filters
BENCHMARK_SPECS = [
    ("mathhay",  "mathhay",    None),
    ("search",   "browsecomp", "browsecomp"),
    ("search",   "webvoyager", "webvoyager"),
    ("mcpbench", "mcpbench",   None),
]


SPECS = {
    "mathhay": {
        "desc_key":      "question",
        "gold_key":      "golden_answer",
        "action_types":  [ActionSpaceType.TOOL_CALLS, ActionSpaceType.TEXT],
        "state_type":    EnvironmentStateType.NONE,
        "state_access":  EnvironmentStateAccess.NONE,
        "verifier":      VerifierType.EXACT_MATCH,
        "verifier_mode": VerifierMode.FINAL,
        "env_desc":      (
            "Long-context mathematical reasoning over relevant and distractor documents "
            "provided in the prompt. The agent computes a numerical answer to a multi-step "
            "question."
        ),
        "action_desc": (
            "Calls to the answer-submission tool and text responses."
        ),
    },
    "browsecomp": {
        "desc_key":      "question",
        "gold_key":      "ground_truth",
        "action_types":  [ActionSpaceType.TOOL_CALLS, ActionSpaceType.TEXT],
        "state_type":    EnvironmentStateType.LIVE_WEB,
        "state_access":  EnvironmentStateAccess.READ_ONLY,
        "verifier":      VerifierType.LLM_JUDGE,
        "verifier_mode": VerifierMode.FINAL,
        "env_desc":      (
            "Web-search questions drawn from BrowseComp. The agent uses a shared tool "
            "interface to find obscure facts and returns a text answer."
        ),
        "action_desc": (
            "Tool calls through a shared interface, primarily live web search, and text responses."
        ),
    },
    "webvoyager": {
        "desc_key":      "question",
        "gold_key":      "ground_truth",
        "action_types":  [ActionSpaceType.TOOL_CALLS, ActionSpaceType.TEXT],
        "state_type":    EnvironmentStateType.LIVE_WEB,
        "state_access":  EnvironmentStateAccess.READ_ONLY,
        "verifier":      VerifierType.LLM_JUDGE,
        "verifier_mode": VerifierMode.FINAL,
        "env_desc":      (
            "Public web content accessed through live search for shopping, travel, academic, "
            "and other online information requests."
        ),
        "action_desc": "Live web-search calls and text responses.",
    },
    "mcpbench": {
        "desc_key":      "task_description",
        "gold_key":      None,
        "action_types":  [ActionSpaceType.TOOL_CALLS, ActionSpaceType.TEXT],
        "state_type":    EnvironmentStateType.IN_MEMORY,
        "state_access":  EnvironmentStateAccess.READ_WRITE,
        "verifier":      VerifierType.LLM_JUDGE,
        "verifier_mode": VerifierMode.SEQUENTIAL,
        "env_desc":      (
            "Tool-use tasks completed through one or more Model Context Protocol servers. "
            "Six numerical verifier results measure task fulfillment, grounding, tool use, "
            "parameter accuracy, dependency awareness, and efficiency."
        ),
        "action_desc": (
            "Parameterized calls to tools exposed by the task's MCP servers and text responses."
        ),
    },
}
