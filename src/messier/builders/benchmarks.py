"""
Benchmark groups, scoring rules, release dates, and source metadata.
"""

# NOTE: benchmark groups are qualitative so their boundaries are not definitive
BENCHMARK_GROUPS = {
    "swebench":          "programming",
    "swebench_verified": "programming",
    "swebench_pro":      "programming",
    "livecodebench":     "programming",
    "gso":               "programming",
    "qcircuitbench":     "programming",
    "terminalbench":     "programming",
    "cybench":           "programming",
    "hcast":             "research",
    "swaa":              "research",
    "rebench":           "research",
    "mlebench":          "research",
    "scienceagentbench": "research",
    "replicationbench":  "research",
    "hle":               "research",
    "matharena":         "research",
    "mathhay":           "research",
    "theagentcompany":   "enterprise",
    "harveyai-lab":      "enterprise",
    "gdpval":            "enterprise",
    "medagentbench":     "enterprise",
    "tau2-bench":        "enterprise",
    "mcpbench":          "enterprise",
    "dabstep":           "enterprise",
    "osworld":           "gui",
    "onlinemind2web":    "gui",
    "browsecomp":        "gui",
    "webvoyager":        "gui",
    "bfcl-live":         "function_calling",
    "bfcl-multi-turn":   "function_calling",
    "toolathlon":        "enterprise",
}


# default used when the source does not provide a task-specific rule
DEFAULT_SCORING_RULES = {
    "swebench_verified": "direct",
    "swebench_pro":      "direct",
    "swebench":          "direct",
    "terminalbench":     "direct",
    "gso":               "threshold",
    "livecodebench":     "direct",
    "tau2-bench":        "all_pass",
    "bfcl-multi-turn":   "direct",
    "swaa":              "direct",
    "rebench":           "threshold",
    "mlebench":          "direct",
    "cybench":           "direct",
    "gdpval":            "threshold",
    "onlinemind2web":    "direct",
    "bfcl-live":         "direct",
    "hle":               "direct",
    "mathhay":           "direct",
    "browsecomp":        "direct",
    "webvoyager":        "direct",
    "mcpbench":          "threshold",
    "osworld":           "threshold",
    "dabstep":           "direct",
    "harveyai-lab":      "all_pass",
    "qcircuitbench":     "threshold",
    "replicationbench":  "direct",
    "scienceagentbench": "direct",
    "medagentbench":     "direct",
    "toolathlon":        "direct",
}


# benchmark release dates used when a task has no date
BENCHMARK_RELEASE_DATES = {
    "swebench_verified": "2024-08-13", # announced by OpenAI
    "swebench_pro":      "2025-08-26", # created on Hugging Face
    "terminalbench":     "2025-11-07", # tbench.ai 2.0 launch
    "gso":               "2025-05-29", # arXiv 2505.23671 v1
    "hcast":             "2025-03-18", # arXiv 2503.14499 v1 (with SWAA)
    "rebench":           "2024-11-22", # arXiv 2411.15114 v1
    "swaa":              "2025-03-18", # bundled with HCAST
    "mlebench":          "2024-10-09", # arXiv 2410.07095 v1
    "cybench":           "2024-08-15", # arXiv 2408.08926 v1
    "gdpval":            "2025-09-25", # announced by OpenAI
    "theagentcompany":   "2024-12-18", # arXiv 2412.14161 v1
    "onlinemind2web":    "2025-04-02", # arXiv 2504.01382 v1
    "bfcl-live":         "2024-08-19", # released with BFCL V2 Live
    "bfcl-multi-turn":   "2024-09-19", # released with BFCL V3 Multi-Turn
    "tau2-bench":        "2025-06-09", # arXiv 2506.07982 v1
    "matharena":         "2025-02-06", # earliest competition (AIME 2025)
    "livecodebench":     "2024-04-01", # released with LCB v1
    "hle":               "2025-04-03", # cais/hle freeze date (2,500 questions)
    "mathhay":           "2024-10-07", # arXiv 2410.04698 v1
    "browsecomp":        "2025-04-09", # released by OpenAI
    "webvoyager":        "2024-01-25", # arXiv 2401.13919 v1
    "mcpbench":          "2025-08-28", # accenture MCP-Bench arXiv 2508.20453 release (cx-cmu evals came later)
    "osworld":           "2024-04-11", # arXiv 2404.07972 v1 (OSWorld original)
    "swebench":          "2023-10-10", # original SWE-bench arXiv (per-task created_at preferred)
    "dabstep":           "2025-02-04", # adyen and Hugging Face release
    "harveyai-lab":      "2026-05-06", # harvey Legal Agent Benchmark (LAB) release
    "qcircuitbench":     "2024-10-10", # arXiv 2410.07961 v1
    "replicationbench":  "2025-10-28", # arXiv 2510.24591 v1
    "scienceagentbench": "2024-10-07", # arXiv 2410.05080 v1
    "medagentbench":     "2025-01-24", # arXiv 2501.14654 v1
    "toolathlon":        "2025-10-29", # arXiv 2510.25726 v1
}


# repository descriptions preserved as provided by GitHub
REPO_DESCRIPTIONS = {
    "NodeBB/NodeBB":               "Node.js based forum software built for the modern web",
    "abetlen/llama-cpp-python":    "Python bindings for llama.cpp",
    "ansible/ansible":             "Ansible is a radically simple IT automation platform that makes your applications and systems easier to deploy and maintain. Automate everything from code deployment to network configuration to cloud management, in a language that approaches plain English, using SSH, with no agents to install on remote systems. https://docs.ansible.com.",
    "astropy/astropy":             "Astronomy and astrophysics core library",
    "django/django":               "The Web framework for perfectionists with deadlines.",
    "element-hq/element-web":      "A glossy Matrix collaboration client for the web.",
    "flipt-io/flipt":              "Enterprise-ready, Git native feature management solution",
    "future-architect/vuls":       "Agent-less vulnerability scanner for Linux, FreeBSD, Container, WordPress, Programming language libraries, Network devices",
    "gravitational/teleport":      "The easiest, and most secure way to access and protect all of your infrastructure.",
    "huggingface/datasets":        "🤗 The largest hub of ready-to-use datasets for AI models with fast, easy-to-use and efficient data manipulation tools",
    "huggingface/tokenizers":      "💥 Fast State-of-the-Art Tokenizers optimized for Research and Production",
    "huggingface/transformers":    "🤗 Transformers: the model-definition framework for state-of-the-art machine learning models in text, vision, audio, and multimodal models, for both inference and training.",
    "internetarchive/openlibrary": "One webpage for every book ever published!",
    "matplotlib/matplotlib":       "matplotlib: plotting with Python",
    "mwaskom/seaborn":             "Statistical data visualization in Python",
    "navidrome/navidrome":         "🎧 Your Personal Streaming Service ",
    "numpy/numpy":                 "The fundamental package for scientific computing with Python.",
    "pallets/flask":               "The Python micro framework for building web applications.",
    "pandas-dev/pandas":           "Flexible and powerful data analysis / manipulation library for Python, providing labeled data structures similar to R data.frame objects, statistical functions, and much more",
    "protonmail/webclients":       "Monorepo hosting the proton web clients",
    "psf/requests":                "A simple, yet elegant, HTTP library.",
    "pydantic/pydantic":           "Data validation using Python type hints",
    "pydata/xarray":               "N-D labeled arrays and datasets in Python",
    "pylint-dev/pylint":           "It's not just a linter that annoys you!",
    "pytest-dev/pytest":           "The pytest framework makes it easy to write small tests, yet scales to support complex functional testing",
    "python-pillow/Pillow":        "Python Imaging Library (fork)",
    "qutebrowser/qutebrowser":     "A keyboard-driven, vim-like browser based on Python and Qt.",
    "scikit-learn/scikit-learn":   "scikit-learn: machine learning in Python",
    "sphinx-doc/sphinx":           "The Sphinx documentation generator",
    "sympy/sympy":                 "A computer algebra system written in pure Python",
    "tornadoweb/tornado":          "Tornado is a Python web framework and asynchronous networking library, originally developed at FriendFeed.",
    "tutao/tutanota":              "Tuta is an email service with a strong focus on security and privacy that lets you encrypt emails, contacts and calendar entries on all your devices.",
    "uploadcare/pillow-simd":      "The friendly PIL fork",
}


# pinned revisions used by environment state references
UPSTREAM_COMMITS: dict[str, str] = {
    "theagentcompany_main": "98b68ef82a47690c316f42fddb05baafaab56851",
    "terminalbench":        "1a6ffa9674b571da0ed040c470cb40c4d85f9b9b",
    "cybench":              "88d6893231fe7ae75d109250f9a1dde310c60008",
    "metr_hcast":           "376f08a6c887d7a6dd3995425290470eda92fcae",
    "metr_rebench":         "93b98062e55f6945d4a7e213a3226dd419896170",
    "bfcl_gorilla":         "6ea57973c7a6097fd7c5915698c54c17c5b1b6c8",
    "gso":                  "c2e4f1a58427cccd15e0e542f136bd204fb19284",
    "swebench_verified":    "c104f840cc67f8b6eec6f759ebc8b2693d585d4a",
    "swebench_pro":         "7ab5114912baf22bb098818e604c02fe7ad2c11f",
    "hle":                  "5a81a4c7271a2a2a312b9a690f0c2fde837e4c29",
    "gdpval":               "11e7900cdcac61bc4daf59e65feb238acda98fbf",
    "dabstep":              "6cf1f3e2869d9fdbae1e9e12392e38dfb437f24f",
    "replicationbench":      "a87284ced5652645d81b82dcace2f41a7a02d3ce",
    "osworld":              "ed1529b70e06227abca94e9ab34de4b7cb15dba1",
    "osworld_file_cache":   "711e0811642364e7aa8f10a8918367d0b626d578",
    "livecodebench":        "0fe84c3912ea0c4d4a78037083943e8f0c4dd505",
    "toolathlon":           "6194034105bc27fa438447172be0e7b4e35396e4",
    "toolathlon_tasks":     "5b19250a0aa3db65babb6dfad39c322dc369240b",
}
