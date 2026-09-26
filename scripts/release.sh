#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ $# -ne 0 ]]; then
  echo "usage: ./scripts/release.sh" >&2
  exit 1
fi

NOTES="docs/release_notes.md"
if [[ -n "$(git status --porcelain --untracked-files=no)" ]]; then
  echo "release requires all tracked changes to be committed" >&2
  exit 1
fi

CURRENT_BRANCH="$(git branch --show-current)"
if [[ "$CURRENT_BRANCH" != "main" && ! "$CURRENT_BRANCH" =~ ^release/v[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
  echo "release must run from main or a release/vX.Y.Z branch" >&2
  exit 1
fi

gh auth status >/dev/null
if ! gh secret list --app actions | awk '{ print $1 }' | grep -Fxq HF_TOKEN; then
  echo "the repository needs an HF_TOKEN Actions secret" >&2
  echo "configure it once with: gh secret set HF_TOKEN" >&2
  exit 1
fi

git fetch origin main
if ! git merge-base --is-ancestor origin/main HEAD; then
  echo "this checkout does not contain the latest origin/main commit" >&2
  exit 1
fi

if [[ "$CURRENT_BRANCH" == "main" ]]; then
  VERSION="$(uv version --bump patch --dry-run --short)"
  TAG="v$VERSION"
  RELEASE_BRANCH="release/$TAG"

  if git show-ref --verify --quiet "refs/heads/$RELEASE_BRANCH"; then
    git switch "$RELEASE_BRANCH"
    git merge --ff-only main
  else
    git switch -c "$RELEASE_BRANCH"
  fi

  if [[ "$(uv version --short)" != "$VERSION" ]]; then
    uv version "$VERSION" --no-sync >/dev/null
  fi
else
  TAG="${CURRENT_BRANCH#release/}"
  VERSION="${TAG#v}"
  RELEASE_BRANCH="$CURRENT_BRANCH"
  if [[ "$(uv version --short)" != "$VERSION" ]]; then
    echo "the project version does not match $RELEASE_BRANCH" >&2
    exit 1
  fi
fi

if git ls-remote --exit-code --tags origin "refs/tags/$TAG" >/dev/null 2>&1; then
  echo "$TAG has already been released" >&2
  exit 1
fi

perl -pi -e "s/^## MESSIER v[0-9]+\.[0-9]+\.[0-9]+$/## MESSIER $TAG/" "$NOTES"
SUMMARY="$(awk '/^## MESSIER / { found=1; next } found && NF { print; exit }' "$NOTES")"
if [[ -z "$SUMMARY" || "$SUMMARY" == "- "* ]]; then
  echo "add a one-sentence summary below the release-notes heading" >&2
  exit 1
fi

printf '\n==> Checking release access\n'
uv run --group dev python -m messier.publishing check

printf '\n==> Building and testing the corpus\n'
./scripts/update.sh

printf '\n==> Running analyses\n'
uv run --group dev python -m analysis.counterfactual_aggregation.run
uv run --group dev python -m analysis.human_time_calibration.run
uv run --group dev python -m analysis.epoch_eci.run
uv run --group dev python -m analysis.frontier_progress.run
uv run --group dev python -m analysis.performance_dimensions.run
uv run --group dev python -m analysis.difficulty_prediction.run
uv run --group dev python -m analysis.capability_subsets.run

printf '\n==> Preparing release files\n'
uv run --group dev python -m messier.publishing prepare "$SUMMARY"

git add README.md pyproject.toml uv.lock docs/data.md docs/huggingface.md docs/release_notes.md page/index.html page/static/data/results.json
if ! git diff --cached --quiet; then
  git commit -m "Release $TAG"
fi
if [[ -n "$(git status --porcelain --untracked-files=no)" ]]; then
  echo "the release produced uncommitted files that are not included in the release commit" >&2
  exit 1
fi

printf '\n==> Uploading dataset candidate\n'
HF_PR_URL="$(uv run --group dev python -m messier.publishing candidate "$TAG" "$NOTES")"

printf '\n==> Opening release pull request\n'
git push --set-upstream origin "$RELEASE_BRANCH"
PR_URL="$(gh pr view "$RELEASE_BRANCH" --json url --jq .url 2>/dev/null || true)"
if [[ -n "$PR_URL" ]]; then
  PR_BODY="$(mktemp)"
  trap 'rm -f "$PR_BODY"' EXIT
  {
    cat "$NOTES"
    printf '\n## Dataset candidate\n\n[Review the Hugging Face candidate](%s)\n' "$HF_PR_URL"
  } > "$PR_BODY"
  gh pr edit "$RELEASE_BRANCH" --title "Release $TAG" --body-file "$PR_BODY" >/dev/null
else
  gh workflow run release-pr.yml \
    --ref main \
    --field branch="$RELEASE_BRANCH" \
    --field tag="$TAG" \
    --field candidate_url="$HF_PR_URL"

  for _ in {1..30}; do
    PR_URL="$(gh pr view "$RELEASE_BRANCH" --json url --jq .url 2>/dev/null || true)"
    [[ -n "$PR_URL" ]] && break
    sleep 2
  done
  if [[ -z "$PR_URL" ]]; then
    echo "GitHub Actions did not open the release pull request" >&2
    exit 1
  fi
fi

printf '\nRelease PR: %s\n' "$PR_URL"
printf 'Merging this PR publishes the Hugging Face dataset and GitHub Release automatically.\n'
