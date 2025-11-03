# AISamples – Chapter 1: Context Engineering

This repository collects hands-on materials for learning about **context engineering** with large language models (LLMs). The first chapter focuses on understanding how prompt context influences model behaviour and how to design reusable prompt patterns for downstream applications.

## What's inside?

| Path | Description |
| --- | --- |
| `notebooks/chapter1_context_engineering.ipynb` | Guided Jupyter notebook that introduces the core ideas of context engineering through short exercises and examples. |

The notebooks are intentionally lightweight so you can extend them with your own experiments as you progress through the chapter.

## Getting started

1. **Clone the repository** (or download the branch `feature/Chapter1_ContextEngineering`).
2. **Create a virtual environment** (recommended):
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```
3. **Install the tooling you need**. The notebooks rely on standard Python scientific tooling. If you already have `ipykernel` and `jupyter` installed globally you can skip this step.
   ```bash
   pip install --upgrade pip
   pip install jupyter ipykernel
   ```
4. **Launch Jupyter Lab or Notebook**:
   ```bash
   jupyter lab
   ```
   Open `notebooks/chapter1_context_engineering.ipynb` and follow the instructions embedded in the notebook cells.

## Learning objectives

By the end of Chapter 1 you will be able to:

- Describe why carefully engineered context is essential for reliable LLM outputs.
- Apply prompt scaffolding techniques such as role-setting, staged instructions, and exemplar selection.
- Measure the impact of context length and information ordering on LLM responses.
- Document experiments so they can be reproduced in later chapters.

## Suggested workflow

1. **Read the overview cells** in the notebook to understand the theory.
2. **Experiment** by adapting the provided prompt templates to your own questions.
3. **Log outcomes** directly in the notebook or in a separate markdown file so you can build a prompt library over time.
4. **Iterate** by comparing how different context strategies affect response quality and cost.

## Contributing

Contributions are welcome! If you discover a useful prompt pattern or improvement:

1. Fork the repository and create a feature branch (for example `feature/new-prompt-pattern`).
2. Add your notebook updates or supporting utilities.
3. Ensure your notebook is readable (remove unnecessary outputs, restart and run all cells).
4. Open a pull request describing the techniques you explored and the results you observed.

## License

Unless stated otherwise in individual files, this project is released under the MIT License.
