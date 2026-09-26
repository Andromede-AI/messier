# Publishing MESSIER

Releases are prepared locally, reviewed in a GitHub PR, and published to the [Hugging Face dataset](https://huggingface.co/datasets/Andromede-AI/messier) after the PR is merged.

## Setup

Authenticate the GitHub CLI:

```bash
gh auth login
```

Add a Hugging Face token with write access to the target dataset to the local `.env` file:

```text
HF_TOKEN=...
```

Add the same token to the GitHub repository so the release workflow can publish after the PR is merged:

```bash
gh secret set HF_TOKEN
```

## Release

Write the release summary and changes in [release_notes.md](release_notes.md). Commit all intended changes locally on `main`, but do not push `main` manually. Then run:

```bash
./scripts/release.sh
```

The command:

1. Increments the project patch version and creates a `release/vX.Y.Z` branch.
2. Rebuilds and tests the corpus.
3. Reruns the analyses and project-page plots.
4. Updates the corpus counts, README News, and release notes.
5. Uploads the dataset to a Hugging Face PR.
6. Pushes the release branch and asks GitHub Actions to open the GitHub release PR as `github-actions[bot]`.

The GitHub PR contains the release notes and a link to the exact Hugging Face dataset candidate. Review the repository changes on GitHub and the uploaded dataset. Merging the GitHub PR publishes the release. GitHub Actions merges the Hugging Face candidate, creates the same version tag on Hugging Face and GitHub, and creates the GitHub Release from [release_notes.md](release_notes.md). If the local command stops before opening the PR, fix the problem on the existing release branch and rerun it.

## Source snapshot

Release builds use the verified source snapshot in `data/raw`. This directory is an input to the builders and is not part of the public dataset release. Refresh it from the pinned original sources only when intended with `uv run --group dev python -c 'from messier.fetch import fetch_all; fetch_all("upstream")'`.
