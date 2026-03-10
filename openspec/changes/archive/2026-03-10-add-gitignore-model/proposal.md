## Why

Prevent large model files and cache data from being committed to the Git repository, reducing repository size and avoiding accidental commits of generated or downloaded model files.

## What Changes

- Add `.gitignore` file to the project root
- Ignore `model/` directory (contains faster-whisper and other local AI models)

## Capabilities

### New Capabilities

- `gitignore-model`: Configure Git to ignore model directory and related cache files

### Modified Capabilities

<!-- No existing capabilities are being modified -->

## Impact

- **Git repository**: Future commits will exclude model files
- **Existing tracked files**: If any model files are already tracked, they will need to be removed from Git cache with `git rm --cached -r model/`
