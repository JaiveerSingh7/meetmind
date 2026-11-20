# MeetMind — Phase 1 starter

A local transcript-to-meeting-report baseline for a B.Tech Neural Networks & Deep Learning project.

## Prerequisites
- macOS on Apple Silicon
- Python 3.11 or 3.12
- Ollama installed and running (`https://ollama.com/download`)

## Run
```bash
ollama pull qwen3:8b
cd meetmind_starter
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
streamlit run app.py
```
If `python3.12` is not available, install it with `brew install python@3.12` or use Python 3.11.

Upload `sample_meeting.txt` or paste its contents into the app.

## Scope
This is **not yet** a multi-agent system: it establishes a measurable baseline with one local LLM call, schema validation, exact-quote checks and manual JSON correction. Further phases add Whisper audio transcription, agent-specific tools, LangGraph orchestration, SQLite persistence and human approval checkpoints.

## Notes
- No remote LLM API is needed after installing Ollama and downloading the model.
- The evidence check only verifies exact substring presence, not semantic correctness.
- Treat model output as a draft. Do not automatically send or assign tasks.
- Do not use confidential meeting recordings without permission.
