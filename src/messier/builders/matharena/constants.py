BENCHMARK = "matharena"
MAX_ATTEMPTS_PER_CELL = 4

# model-name suffixes that identify scaffolds
SCAFFOLD_SUFFIXES = {
    "(Selfcheck Agent)":  "selfcheck",
    "(best-of-32)":       "best-of-32",
    "(agent)":            "matharena-agent",
}

# competition name, date, and result type
COMPETITIONS = [
    ("aime_2025",      "2025-02-06", "binary"),
    ("aime_2026",      "2026-02-05", "binary"),
    ("hmmt_feb_2025",  "2025-02-15", "binary"),
    ("hmmt_feb_2026",  "2026-02-14", "binary"),
    ("hmmt_nov_2025",  "2025-11-08", "binary"),
    ("brumo_2025",     "2025-04-12", "binary"),
    ("smt_2025",       "2025-02-15", "binary"),
    ("cmimc_2025",     "2025-03-22", "binary"),
    ("usamo_2025",     "2025-03-19", "proof"),
    ("usamo_2026",     "2026-03-25", "proof"),
    ("imo_2025",       "2025-07-15", "proof"),
    ("imc_2025",       "2025-07-31", "proof"),
    ("apex_shortlist", "2025-09-01", "binary"),
]

HUMAN_GRADED_PROOFS = {"usamo_2025", "imo_2025", "imc_2025"}

# local completion-time estimates
# TIME_BANDS = {
#     "aime":     [(0.5, 3, 5)] * 5 + [(2, 12, 18)] * 5 + [(5, 25, 40)] * 5,
#     "hmmt":     [(0.5, 3, 5)] * 3 + [(2, 10, 18)] * 4 + [(5, 22, 35)] * 5,
#     "olympiad": [(5, 30, 60), (15, 90, 150), (30, 150, 240)] * 2,
#     "apex":     [(5, 25, 60)] * 30,
# }

ENV_DESCRIPTIONS = {
    "aime_2025":      "American Invitational Mathematics Examination (AIME) 2025 - a mathematics problem requiring a short numerical answer.",
    "aime_2026":      "American Invitational Mathematics Examination (AIME) 2026 - a mathematics problem requiring a short numerical answer.",
    "hmmt_feb_2025":  "Harvard-MIT Mathematics Tournament (HMMT), February 2025 - a mathematics problem requiring a short answer.",
    "hmmt_feb_2026":  "Harvard-MIT Mathematics Tournament (HMMT), February 2026 - a mathematics problem requiring a short answer.",
    "hmmt_nov_2025":  "Harvard-MIT Mathematics Tournament (HMMT), November 2025 - a mathematics problem requiring a short answer.",
    "brumo_2025":     "Brown University Mathematics Olympiad (BRUMO) 2025 - a mathematics problem requiring a short answer.",
    "smt_2025":       "Stanford Math Tournament (SMT) 2025 - a mathematics problem requiring a short answer.",
    "cmimc_2025":     "Carnegie Mellon Informatics and Mathematics Competition (CMIMC) 2025 - a mathematics problem requiring a short answer.",
    "usamo_2025":     "USA Mathematical Olympiad (USAMO) 2025 - a mathematics problem requiring a written proof.",
    "usamo_2026":     "USA Mathematical Olympiad (USAMO) 2026 - a mathematics problem requiring a written proof.",
    "imo_2025":       "International Mathematical Olympiad (IMO) 2025 - a mathematics problem requiring a written proof.",
    "imc_2025":       "International Mathematics Competition for University Students (IMC) 2025 - a mathematics problem requiring a written proof.",
    "apex_shortlist": "MathArena APEX shortlist - a competition-style mathematics problem requiring a short answer.",
}

ACTION_SPACE_DESCRIPTION = "Text responses containing a short answer or written mathematical proof."

# pinned source for each competition
MATHARENA_COMMITS: dict[str, str] = {
    "aime_2025":      "16e93d3bb9b0",
    "aime_2026":      "f916cfc3eebc",
    "hmmt_feb_2025":  "e3fb6c74f24e",
    "hmmt_feb_2026":  "4523bd2724d5",
    "hmmt_nov_2025":  "1cdfff8872c0",
    "brumo_2025":     "3b234efea64d",
    "smt_2025":       "bf5c9e10a65c",
    "cmimc_2025":     "2d400e898d07",
    "usamo_2025":     "32b67b37603d",
    "usamo_2026":     "e45c83f64b0a",
    "imo_2025":       "f4ae9a23b328",
    "imc_2025":       "a68faf848caa",
    "apex_shortlist": "c7c045234706",
}
