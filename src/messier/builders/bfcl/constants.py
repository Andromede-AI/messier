from ..benchmarks import UPSTREAM_COMMITS

CATEGORIES = {
    "live":       "bfcl-live",
    "multi_turn": "bfcl-multi-turn",
}

# exclude "Should NOT call a function" subtests because their semantics are inverted
DROP_SUBTESTS = {"live_irrelevance", "live_relevance"}

ENV_DESCRIPTIONS = {
    "live_simple":             "Single-turn function calling with one expected call.",
    "live_multiple":           "Single-turn function calling with multiple expected calls in sequence.",
    "live_parallel":           "Single-turn function calling with multiple expected calls that may be issued together.",
    "live_parallel_multiple":  "Single-turn function calling combining sequential and parallel calls.",
    "multi_turn_base":         "Stateful function calling across multiple user turns.",
    "multi_turn_miss_func":    "Stateful function calling where a needed function is unavailable.",
    "multi_turn_miss_param":   "Stateful function calling where a needed parameter is missing.",
    "multi_turn_long_context": "Stateful function calling across a long interaction history.",
    "multi_turn_composite":    "Stateful function calling combining unavailable functions, missing parameters, and long interaction histories.",
}

LIVE_ACTION_DESCRIPTION = (
    "Single-turn function-calling: agent receives the user request and a list of "
    "OpenAI compatible tool schemas. The agent emits one or more tool calls graded by structural AST "
    "match against the expected call sequence."
)
MULTI_TURN_ACTION_DESCRIPTION = (
    "Multi-turn function-calling: agent invokes methods on stateful in-memory class "
    "registries (TwitterAPI, GorillaFileSystem, ...) whose state is mutated turn by turn. "
    "graded by structural match on the per-turn method-call trace."
)
MULTI_TURN_CLASS_REPO_REF = (
    "github://ShishirPatil/gorilla/berkeley-function-call-leaderboard/"
    f"bfcl_eval/multi_turn_eval/func_source_code@{UPSTREAM_COMMITS['bfcl_gorilla']}"
)
