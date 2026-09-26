PLATFORM_DESCRIPTIONS = {
    "leetcode":   "LeetCode contest problems - algorithmic challenges from periodic LeetCode contests. Solutions implement a specified function signature and are graded by hidden I/O test cases.",
    "codeforces": "Codeforces contest problems - competitive programming challenges with explicit Elo difficulty ratings. Solutions read from stdin and write to stdout. They are graded by I/O test cases.",
    "atcoder":    "AtCoder contest problems - competitive programming from ABC / ARC / AGC contests. Solutions read from stdin and write to stdout. They are graded by I/O test cases.",
}

# codeforces / LeetCode rating bands -> (low, mid, high) human-effort minutes.
RATING_BANDS = [
    (1200, (1,  4,  10)),
    (1500, (2,  10, 22)),
    (1800, (3,  18, 40)),
    (2100, (5,  30, 65)),
    (2400, (8,  50, 100)),
    (10**9, (12, 80, 180)),
]

# source problems from AtCoder use a position letter instead of an Elo rating
ATCODER_POSITION_BANDS = {
    "a": (1, 3, 6), "b": (1, 5, 10), "c": (3, 12, 22), "d": (8, 25, 40),
    "e": (15, 40, 70), "f": (30, 70, 130), "g": (50, 100, 180), "h": (50, 100, 180),
}

ACTION_SPACE_DESCRIPTION = "Text responses. No tools or execution feedback are available."
