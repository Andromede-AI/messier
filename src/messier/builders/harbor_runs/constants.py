# map Harbor dataset slugs to MESSIER benchmark names
SLUG_TO_NAME = {
    "adyen/dabstep":                       "dabstep",
    "harveyai/lab":                        "harveyai-lab",
    "qcircuitbench/qcircuitbench":         "qcircuitbench",
    "replicationbench/replicationbench":   "replicationbench",
    "scienceagentbench/scienceagentbench": "scienceagentbench",
    "stanford/medagentbench":              "medagentbench",
}

# environment description stored for each benchmark
ENV_DESCRIPTIONS = {
    "dabstep":           "Payment transaction data and documentation describing its fields, fees, merchants, and acquirers.",
    "harveyai-lab":      "Legal matters involving contracts, regulations, fact patterns, and supporting documents.",
    "qcircuitbench":     "A quantum-circuit design environment with Qiskit, a writable Python workspace, and an oracle or target circuit.",
    "replicationbench":  "An astrophysics research environment containing a paper, scientific datasets, software dependencies, and a writable workspace for reproducing a reported result.",
    "scienceagentbench": "A scientific computing environment containing research datasets, Python libraries, and a writable workspace for producing and evaluating a standalone program.",
    "medagentbench":     "Clinical tasks performed in a simulated electronic health record system using the FHIR standard.",
}

# verifier type stored for each benchmark
VERIFIER_TYPES = {
    "dabstep":           "script",
    "harveyai-lab":      "llm_judge",
    "qcircuitbench":     "script",
    "replicationbench":  "script",
    "scienceagentbench": "script",
    "medagentbench":     "script",
}

# judge model recorded for HarveyAI-Lab
HARVEYAI_JUDGE = "anthropic/claude-sonnet-4-6"

DABSTEP_CONTEXT_FILES = [
    "payments.csv",
    "fees.json",
    "manual.md",
    "merchant_data.json",
    "merchant_category_codes.csv",
    "payments-readme.md",
    "acquirer_countries.csv",
]
