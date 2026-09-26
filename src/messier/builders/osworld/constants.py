import re

BENCHMARK = "osworld"

# trajectory dumps from the xlangai/ubuntu_osworld_verified_trajs HF release.
HF_TRAJS_REPO = "xlangai/ubuntu_osworld_verified_trajs"
HF_TRAJS_BASE = f"https://huggingface.co/datasets/{HF_TRAJS_REPO}/resolve/main"

# task configurations come from the upstream evaluation_examples directory
GITHUB_TASKS_REPO = "xlang-ai/OSWorld"
GITHUB_RAW_BASE = f"https://raw.githubusercontent.com/{GITHUB_TASKS_REPO}"
HF_FILE_CACHE_REPO = "xlangai/ubuntu_osworld_file_cache"

DOMAINS = (
    "chrome",
    "gimp",
    "libreoffice_calc",
    "libreoffice_impress",
    "libreoffice_writer",
    "multi_apps",
    "os",
    "thunderbird",
    "vlc",
    "vs_code",
)

DOMAIN_ENV_DESCRIPTIONS = {
    "chrome": (
        "An Ubuntu desktop with Chrome, including live websites, browser tabs, bookmarks, "
        "profile settings, and forms."
    ),
    "gimp": (
        "An Ubuntu desktop with GIMP and image files that can be edited, transformed, "
        "and exported."
    ),
    "libreoffice_calc": (
        "An Ubuntu desktop with LibreOffice Calc and spreadsheets containing formulas, "
        "formatted data, filters, and charts."
    ),
    "libreoffice_impress": (
        "An Ubuntu desktop with LibreOffice Impress and presentations containing slides, "
        "text, images, layouts, and transitions."
    ),
    "libreoffice_writer": (
        "An Ubuntu desktop with LibreOffice Writer and documents containing formatted text, "
        "tables, headers, and footers."
    ),
    "multi_apps": (
        "An Ubuntu desktop with tasks spanning multiple applications, including office "
        "software, the web browser, files, media applications, and the terminal."
    ),
    "os": (
        "An Ubuntu desktop with system settings, processes, installed packages, network "
        "configuration, display settings, and a terminal."
    ),
    "thunderbird": (
        "An Ubuntu desktop with Thunderbird, configured email accounts, messages, folders, "
        "filters, and preferences."
    ),
    "vlc": (
        "An Ubuntu desktop with VLC, media files, playback controls, conversion tools, "
        "filters, and application settings."
    ),
    "vs_code": (
        "An Ubuntu desktop with VS Code, project files, extensions, editor settings, "
        "and build and run controls."
    ),
}

ACTION_SPACE_DESCRIPTION = (
    "Mouse and keyboard control through PyAutoGUI or a computer-use interface, including "
    "clicking, typing, scrolling, keyboard shortcuts, and window management. The agent "
    "can also open a terminal and issue shell commands through the desktop."
)


# strip step-budget and duplicate-run suffixes when grouping archives
ZIP_SUFFIX_RE = re.compile(r"([_-]\d+_?steps?)(_\d+)?(-\d+)?$")

# map archives only when the underlying model and scaffold can be identified
ZIP_AGENT_MAP: dict[str, tuple[str, str]] = {
    "claude-3-7-sonnet-20250219":            ("claude-3-7-sonnet", "osworld-default"),
    "claude-4-sonnet-20250514":              ("claude-sonnet-4", "osworld-default"),
    "claude-sonnet-4-5-20250929":            ("claude-sonnet-4-5", "osworld-default"),
    "results-vlaa-gui":                      ("claude-opus-4-6", "vlaa-gui"),
    "results_opus4p5_single_agent":          ("claude-opus-4-5", "single-agent"),
    "results_hippo_agent":                   ("claude-opus-4-5", "hippo-agent"),
    "doubao-1-5-thinking-vision-pro-250428": ("doubao-1-5-thinking-vision-pro", "osworld-default"),
    "kimi-k25":                              ("kimi-k2-5", "osworld-default"),
    "kimi-k26":                              ("kimi-k2-6", "osworld-default"),
    "kimi-vl-a3b":                           ("kimi-vl-a3b", "osworld-default"),
    "o3":                                    ("o3", "osworld-default"),
    "o3_gta1":                               ("o3", "gta1"),
    "results_gemini_50_steps_aws":           ("gemini-2-5-pro", "osworld-default"),
    "results_klick_openapa_full_newkey_bg2": ("gemini-3-1-pro", "klick-openapa"),
    "uipath_gpt_5":                          ("gpt-5", "uipath"),
    "qwen2.5-vl-32b-instruct":               ("qwen-2-5-vl-32b-instruct", "osworld-default"),
    "qwen2.5-vl-72b-instruct":               ("qwen-2-5-vl-72b-instruct", "osworld-default"),
    "autoglm":                               ("autoglm-os", "autoglm"),
    "results_autoglm_v":                     ("autoglm-os", "autoglm-v"),
    "coact1-150-100-50":                     ("o3", "coact-1"),
    "evocua_20260105":                       ("evocua-32b-a3b", "evocua"),
    "evocua_8b_20260105":                    ("evocua-8b", "evocua"),
    "jedi-7b-4o":                            ("gpt-4o", "jedi-7b"),
    "jedi-7b-o3":                            ("o3", "jedi-7b"),
    "mobile-agent-v3-gui-owl-7b":            ("gui-owl-7b", "gui-owl"),
    "mobileagent_v3":                        ("gui-owl-32b", "mobile-agent-v3"),
    "opencua_agent-opencua_32b-cot_l2-action_history-3image-Ubuntu": ("opencua-32b", "opencua-cot-l2"),
    "opencua_agent-opencua_7b-cot_l2-action_history-3image-Ubuntu": ("opencua-7b", "opencua-cot-l2"),
    "opencua_agent-opencua_a3b-cot_l2-action_history-3image-Ubuntu": ("opencua-a3b", "opencua-cot-l2"),
    "opencua_agent-opencua_qwen2_7b-cot_l2-action_history-3image-Ubuntu": ("opencua-qwen2-7b", "opencua-cot-l2"),
    "OSWorldSurfer-Holo3-35B-A3B-20260330-OSWorld-Verified_20260420_verified-run_with-local-rewards": ("osworld-surfer-holo3-35b-a3b", "hcompany"),
    "OSWorldSurfer-Holo3-35B-A3B-20260330-OSWorld-Verified_hcompany-internal-runs-20260416_complete-with-rewards": ("osworld-surfer-holo3-35b-a3b", "hcompany"),
    "qwen35_osworld_eval_traj":               ("qwen-3-5-cua", "qwen-cua"),
    "qwen36-cua-think":                       ("qwen-3-6-cua", "qwen-cua"),
    "qwen36-cua-nothink":                     ("qwen-3-6-cua", "qwen-cua"),
    "result_aws_dart_gui_20260122":           ("dart-gui", "dart-gui"),
    "results_UI-TARS-2-2509":                 ("ui-tars-2", "ui-tars"),
    "results_agent_s2_gemini":                ("gemini-2-5-pro", "agent-s2"),
    "results_agent_s2_o3":                    ("o3", "agent-s2"),
    "results_maestro":                        ("o3", "maestro"),
    "results_maestro_100steps_results_only":  ("o3", "maestro"),
    "results_maestro_50steps_2_results_only": ("o3", "maestro"),
    "results_mano":                           ("deepminer-mano-7b", "mano"),
    "results_o3_tars1p5":                     ("o3", "agent-s2-5"),
    "results_omnitars":                       ("omnitars", "omnitars"),
    "results_seed18":                         ("doubao-seed-1-8", "doubao-seed"),
    "results_tianxi_action_7b":               ("tianxi-action-7b", "tianxi"),
    "results_tianxi_action_7b_turn1":         ("tianxi-action-7b", "tianxi-turn1"),
    "UI-TARS-0717":                           ("ui-tars-0717", "ui-tars"),
    "uitars-72b-dpo":                         ("ui-tars-72b-dpo", "ui-tars"),
    "uitars15-7b":                            ("ui-tars-1-5-7b", "ui-tars-1-5"),
}

ZIP_SUPPORTING_MODELS: dict[str, list[str]] = {
    "coact1-150-100-50":                    ["o4-mini", "computer-use-preview"],
    "jedi-7b-4o":                           ["jedi-7b"],
    "jedi-7b-o3":                           ["jedi-7b"],
    "results_maestro":                       ["ui-tars"],
    "results_maestro_100steps_results_only": ["ui-tars"],
    "results_maestro_50steps_2_results_only": ["ui-tars"],
    "results_o3_tars1p5":                    ["ui-tars-1-5-7b"],
}

ZIP_REASONING_EFFORT = {
    "qwen36-cua-think": "enabled",
}
