## Context

The Chenyme-AAVT project uses Faster-Whisper and other local AI models stored in the `model/` directory. These model files are large binary files that should not be version controlled. Currently, there is no `.gitignore` file, which risks accidentally committing these files to the repository.

## Goals / Non-Goals

**Goals:**
- Create a `.gitignore` file at the project root
- Ignore the `model/` directory to prevent model files from being committed
- Keep the repository size small and clean

**Non-Goals:**
- Removing already tracked model files from Git history (manual step if needed)
- Ignoring other directories or file types beyond model folder

## Decisions

**Decision: Only ignore `model/` directory**
- Rationale: User explicitly requested to only ignore the model folder
- Alternative: Add comprehensive Python/IDE ignores, but this can be done later if needed

**Decision: Place `.gitignore` at project root**
- Rationale: Standard practice, applies to entire repository
- Alternative: Place in subdirectories, but root-level is cleaner and more maintainable

## Risks / Trade-offs

- **Risk**: If model files were already committed, they remain in Git history
  - Mitigation: User can run `git rm --cached -r model/` if needed, followed by a commit
