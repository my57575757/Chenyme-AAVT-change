# AGENTS.md - Development Guide for Chenyme-AAVT

## Project Overview
Chenyme-AAVT is a Streamlit-based automatic video/audio translation tool. It provides subtitle recognition, translation, blog generation, and voice simulation using Faster-Whisper and various LLM APIs.

## Build & Run Commands

### Environment Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Run the application
streamlit run Chenyme-AAVT.py
# Or use the batch file on Windows
2_webui.bat
```

### Running Tests
```bash
# Run the test page (TTS testing)
# Navigate to "Test" section in the WebUI
# No formal test framework - manual testing via Streamlit pages
```

### Development
```bash
# No formal linting/formatting configured
# Recommended additions:
pip install black flake8 isort mypy
black .
flake8 .
isort .
```

## Code Style Guidelines

### Imports
- Standard library imports first (os, sys, datetime, platform)
- Third-party imports second (streamlit, openai, torch)
- Local imports last (utils.*, styles.*, page.*)
- Use absolute imports from project root

### Naming Conventions
- **Files**: snake_case (e.g., `global_style.py`, `get_font.py`)
- **Functions**: snake_case (e.g., `get_font_data()`, `style()`)
- **Variables**: snake_case for locals, UPPERCASE for constants
- **Classes**: PascalCase (none in current codebase)
- **Config keys**: PascalCase in TOML files (e.g., `ChatGPT`, `Faster_Local`)

### Formatting
- **Indentation**: 4 spaces (no tabs)
- **Line length**: ~120 chars (not strictly enforced)
- **Quotes**: Double quotes for strings
- **Spacing**: Single blank line between functions, double between classes

### Type Hints
- Currently minimal type hints in codebase
- When adding new code, add type hints for function parameters and return values
- Use `Optional[]` for nullable values
- Use `Dict[]`, `List[]` from typing module

### Error Handling
- Use try-except blocks for API calls and file operations
- Display user-friendly messages via Streamlit components (`st.error()`, `st.warning()`, `st.toast()`)
- Log specific error details for debugging
- Pattern:
```python
try:
    # API or file operation
except Exception as e:
    if "Connection error" in str(e):
        st.error("Specific user message")
    else:
        st.error(f"Generic message: {e}")
```

### Streamlit Patterns
- Call `style()` function at start of each page for consistent UI
- Load config files (TOML) at module level
- Use `st.session_state` for跨-page state
- Use `st.expander()`, `st.columns()`, `st.tabs()` for layout
- Buttons use `type="primary"` for main actions

### Configuration
- Config files in `config/` directory (TOML format)
- Main configs: `llms.toml`, `project.toml`, `whisper.toml`, `translate.toml`
- Access pattern:
```python
path = os.getcwd() + "/"
config_path = path + "config/llms.toml"
with open(config_path, 'r', encoding='utf-8') as config_file:
    config = toml.load(config_file)
```

### File Structure
```
D:\project\Chenym-AAVT/
├── Chenyme-AAVT.py      # Main entry point
├── page/                # Streamlit pages
│   ├── project/         # Main features (audio, video, translate, home)
│   ├── setting/         # Settings page
│   └── tests/           # Test pages
├── utils/               # Utility functions (public.py, get_font.py)
├── styles/              # CSS styling (global_style.py)
├── model/               # Local AI models (faster-whisper)
├── config/              # Configuration files (TOML, JSON)
├── cache/               # Temporary cache (tts, public)
└── requirements.txt     # Dependencies
```

### Key Dependencies
- streamlit==1.37.1 (UI framework)
- faster_whisper==1.0.3 (audio transcription)
- openai==1.41.0 (LLM API client)
- torch, torchvision, torchaudio (ML backend)
- opencv_python, numpy (media processing)

### Git & Commits
- No commit message conventions enforced
- Feature branches recommended for new features
- PRs reviewed before merging to main

### OpenSpec Skills
This project uses OpenSpec workflow skills in `.opencode/skills/`:
- `openspec-propose` - Propose changes with artifacts
- `openspec-explore` - Explore codebase and requirements
- `openspec-apply-change` - Implement tasks from specs
- `openspec-archive-change` - Archive completed changes

### Notes
- Windows-first development (batch files, path handling)
- Path concatenation uses `+ "/"` pattern (consider `pathlib` for new code)
- Encoding always specified as `utf-8` for file operations
- Environment variable `KMP_DUPLICATE_LIB_OK=True` set for OMP compatibility

### Page Structure
- **Home** (`page/project/home.py`): Main landing page with AI assistant chatbot
- **Audio** (`page/project/audio.py`): Audio file transcription using Faster-Whisper
- **Video** (`page/project/video.py`): Video file transcription and blog generation
- **Translate** (`page/project/translate.py`): Subtitle translation with multiple LLM providers
- **Settings** (`page/setting/set.py`): Configuration for API keys, models, and preferences
- **Test** (`page/tests/test.py`): TTS testing with OpenAI TTS API
- **Tools** (`page/tests/tools.py`): Auxiliary utilities

### API Integrations
- **OpenAI/ChatGPT**: Translation, TTS, chat assistant
- **Anthropic/Claude**: Translation engine
- **Google/Gemini**: Translation engine
- **DeepSeek, Moonshot, Zhipu, Yi**: Chinese LLM providers for translation

### Media Processing
- Uses OpenCV for video frame extraction
- FFmpeg for audio/video processing (external dependency)
- Faster-Whisper for speech-to-text transcription
- Supports VAD (Voice Activity Detection) for improved accuracy

### Caching Strategy
- `cache/tts/`: Generated TTS audio files
- `cache/public/`: Shared resources and samples
- Temporary directories created with timestamps for session outputs
