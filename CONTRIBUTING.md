# Contributing

Thanks for contributing to the AI Firefighter Wildfire Reconnaissance System. This guide covers how we work together on this capstone.

## Getting set up

1. Clone the repo and create a virtual environment (Python 3.11+).
2. Install dependencies: `pip install -r requirements.txt`.
3. Confirm your setup by running the tests: `pytest`.

## Workflow

- `main` is the protected default branch. Do not commit directly to it.
- Create a branch for your work, named `area/short-description` (for example `modeling/yolo-baseline`, `data/flame3-preprocess`).
- Open a pull request into `main`. At least one other team member reviews before merge.
- Keep pull requests focused and reasonably small. Reference the issue it addresses (for example, "Closes #12").
- The CI smoke test must pass before merging.

## Issues

- Use issues to track tasks, bugs, and risks.
- Apply an area label (`area:data`, `area:modeling`, `area:evaluation`, `area:app`, `area:docs`) and a type label (`type:bug`, `type:enhancement`).
- Assign an owner.

## Commit messages

- Write clear, present-tense messages ("Add mask-tracking baseline", not "added stuff").

## Code style

- Follow standard Python conventions (PEP 8). Keep functions small and documented.

## Questions

- Raise anything in the weekly team meeting, or open a discussion issue.
