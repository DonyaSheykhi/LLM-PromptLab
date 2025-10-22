# LLM-PromptLab

Lab for designing, running, and benchmarking zero-shot & few-shot prompts across NLP tasks.
See `configs/` for ready examples and `lab/` for extendable backends and tasks.

## Quick start
```bash
python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1
# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
set OPENAI_API_KEY (or $env:OPENAI_API_KEY on Windows)

python lab/run.py --config configs/cls_agnews/openai_0shot.yaml
```
