## ADDED Requirements

### Requirement: Git ignores model directory
The system SHALL configure Git to ignore all files and directories under the `model/` folder at the project root.

#### Scenario: New model files are not tracked
- **WHEN** new files are created or downloaded into the `model/` directory
- **THEN** Git does not track or suggest these files for commit

#### Scenario: Existing untracked model files are ignored
- **WHEN** `git status` is executed
- **THEN** files in `model/` directory do not appear as untracked files
