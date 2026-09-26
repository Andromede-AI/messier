BENCHMARK = "tau2-bench"

# use verified tasks where available and the Sierra source for banking knowledge
VERIFIED_DOMAINS = ("airline", "retail", "telecom")
SIERRA_DOMAINS   = ("banking_knowledge",)
DOMAINS          = VERIFIED_DOMAINS + SIERRA_DOMAINS

ENV_DESCRIPTIONS = {
    "airline": (
        "An airline customer-service system containing users, flights, reservations, "
        "payments, baggage, and travel insurance. The agent assists a customer while "
        "working with the booking database."
    ),
    "retail": (
        "A retail customer-service system containing user accounts, products, orders, "
        "payments, returns, exchanges, and refunds. The agent assists a customer while "
        "working with the store database."
    ),
    "telecom": (
        "A telecom customer-service system containing customer accounts, phone lines, "
        "plans, devices, bills, and technical-support information. The agent assists a "
        "customer while working with the service database."
    ),
    "banking_knowledge": (
        "A bank customer-service system containing policy and product documents, "
        "task-specific customer data, and services for applications, referrals, transactions, "
        "and transfers to a human agent."
    ),
}

ACTION_DESCRIPTIONS = {
    "airline": (
        "Tool calls for reading and updating the airline system, together with text responses "
        "to the customer."
    ),
    "retail": (
        "Tool calls for reading and updating the retail system, together with text responses "
        "to the customer."
    ),
    "telecom": (
        "Tool calls for reading and updating the telecom system, together with text responses "
        "to the customer."
    ),
    "banking_knowledge": (
        "Tool calls and text responses for searching bank documents and carrying out customer-service "
        "actions. Some run configurations also provide shell access for document search."
    ),
}

# public Sierra trajectory submissions
SIERRA_S3_BUCKET = "sierra-tau-bench-public"
SIERRA_S3_REGION = "us-west-2"
SIERRA_S3_BASE = f"https://{SIERRA_S3_BUCKET}.s3.{SIERRA_S3_REGION}.amazonaws.com/submissions"

# submissions whose trajectory files are present in the source archive
SUBMISSIONS = [
    "claude-opus-4-5_sierra_2026-02-26",
    "claude-sonnet-4-5_sierra_2026-02-26",
    "gemini-3-pro_sierra_2026-03-02",
    "gemini-3-flash_sierra_2026-03-02",
    "glm-5-think_sierra_2026-03-02",
    "gpt-5-2_sierra_2026-02-26",
    "qwen3.5-397b-a17b-think_sierra_2026-03-02",
]

# these trials use task definitions that differ from the other submissions
EXCLUDED_TASKS = {
    "gemini-3-flash_sierra_2026-03-02": {
        "airline.2",
        "airline.5",
        "airline.7",
        "airline.9",
        "airline.14",
        "airline.23",
        "airline.27",
        "airline.29",
        "airline.33",
        "airline.35",
        "airline.37",
        "airline.38",
        "airline.44",
    },
}
