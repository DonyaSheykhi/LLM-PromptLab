# LLM-PromptLab

A modular, reproducible framework for evaluating **LLM prompting**, **zero-shot/few-shot classification**, and **lightweight dataset distillation** using both **OpenAI** and **HuggingFace** backends.

LLM-PromptLab is designed to emulate real-world NLP engineering workflows:  
✔ config-driven experiments  
✔ modular backends  
✔ reproducible seeds  
✔ experiment logging  
✔ clean evaluation metrics  
✔ experiment cards for research reporting  

This repository demonstrates practical competency in LLM engineering, experiment design, model evaluation, and scalable NLP system development.

---

## 🚀 Quick Start

```bash
# Clone the project
git clone https://github.com/DonyaSheykhi/LLM-PromptLab
cd LLM-PromptLab

# Create venv
python -m venv .venv
./.venv/Scripts/activate

# Install requirements
pip install -r requirements.txt

# Run an example experiment
python -m lab.run --config configs/cls_agnews/hf_tiny_fast.yaml
