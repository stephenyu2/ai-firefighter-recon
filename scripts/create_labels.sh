#!/usr/bin/env bash
# Create issue labels for the project.
# Requires the GitHub CLI (gh). Run `gh auth login` once first, then run this
# from inside the repo directory. You can delete this script after running it.
set -e

# Area labels (match the team's work areas)
gh label create "area:data"       --color 0E8A16 --description "Datasets, preprocessing, splits" --force
gh label create "area:modeling"   --color 1D76DB --description "Detector, fine-tuning, Molmo 2, baseline" --force
gh label create "area:evaluation" --color 5319E7 --description "Metrics, test sets, scoring" --force
gh label create "area:app"        --color FBCA04 --description "Brief assembly, CLI, demo" --force
gh label create "area:docs"       --color 006B75 --description "README, report, repo hygiene" --force

# Type labels
gh label create "type:bug"         --color D73A4A --description "Something is broken" --force
gh label create "type:enhancement" --color A2EEEF --description "New feature or improvement" --force

# Workflow labels
gh label create "blocked"       --color B60205 --description "Blocked on a dependency or decision" --force
gh label create "priority:high" --color E99695 --description "Do this first" --force

echo "Labels created."
