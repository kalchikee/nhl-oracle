"""
Render cron wrapper — handles git pull before and git push after each run.
Usage: python render_runner.py morning|evening|retrain

Requires env vars:
  GITHUB_PAT          — personal access token with repo write scope
  DISCORD_WEBHOOK_URL — passed through to child scripts
  ODDS_API_KEY        — passed through (morning only)
"""

import os
import subprocess
import sys

REPO = "kalchikee/nhl-oracle"

SCRIPTS = {
    "morning": "morning_run.py",
    "evening": "evening_run.py",
    "retrain": "train_model.py",
}

# Files to commit after each task
COMMIT_FILES = {
    "morning": ["data/prediction_history.json", "data/season_state.json", "data/elo_ratings.json"],
    "evening": ["data/prediction_history.json", "data/elo_ratings.json"],
    "retrain": ["models/", "data/cache/"],
}


def run(cmd, check=True):
    print(f"$ {cmd}")
    return subprocess.run(cmd, shell=True, check=check)


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in SCRIPTS:
        print(f"Usage: python render_runner.py {'|'.join(SCRIPTS)}")
        sys.exit(1)

    task = sys.argv[1]
    token = os.environ.get("GITHUB_PAT")
    if not token:
        print("ERROR: GITHUB_PAT env var not set")
        sys.exit(1)

    # Configure git identity
    run('git config --global user.email "nhl-oracle-bot@render.com"')
    run('git config --global user.name "NHL Oracle Bot"')

    # Set authenticated remote (token embedded in URL — standard CI pattern)
    run(f"git remote set-url origin https://x-access-token:{token}@github.com/{REPO}.git")

    # Pull latest data files before running
    run("git pull origin main")

    # Run the actual script
    run(f"python src/{SCRIPTS[task]}")

    # Stage the data files this task writes
    files = " ".join(COMMIT_FILES[task])
    run(f"git add {files}", check=False)

    # Commit only if something changed
    result = run("git diff --staged --quiet", check=False)
    if result.returncode != 0:
        run(f'git commit -m "Update data [{task}] [skip ci]"')
        run("git push origin main")
    else:
        print("No data changes to commit.")


if __name__ == "__main__":
    main()
