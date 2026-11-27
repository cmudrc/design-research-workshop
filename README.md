# Design Research Workshop

Run two hands-on demonstrations of the [Design Research Python ecosystem](https://cmudrc.github.io/design-research/). Both use a deterministic laptop-design proxy with the ecosystem's actual agent, experiment, and analysis interfaces. The demos run on a CPU and require no API key or model weights.

| Demo | What you can change | Run |
| --- | --- | --- |
| [Prompt framing and design decisions](notebooks/laptop_prompt_study.ipynb) | Change the design goal and price limit, then score the candidate designs with the library evaluator. | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/cmudrc/design-research-workshop/blob/v0.1.0/notebooks/laptop_prompt_study.ipynb) |
| [Proposal, critique, and revision](notebooks/laptop_propose_critic.ipynb) | Change the battery and weight constraints, then inspect the critique and the revised proposal. | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/cmudrc/design-research-workshop/blob/v0.1.0/notebooks/laptop_propose_critic.ipynb) |

## Install the workshop source

Use Python 3.12 or newer and Git. Each Colab notebook runs this installation in its first code cell.

```sh
python -m pip install "design-research-workshop @ git+https://github.com/cmudrc/design-research-workshop.git@v0.1.0"
```

The `design_research_workshop` package supplies the laptop rules and a context manager that starts the local HTTP proxy. The policies cover the packaged student laptop problem, its two workshop goals, and common phrasings of price ceilings, battery minimums, and weight limits. The study outputs describe the authored policies. The separate [live-model tutorial](https://cmudrc.github.io/design-research/prompt_framing_study.html) runs the corresponding model-backed workflow.

## Run locally

Clone this repository, install the package into your Jupyter environment, then open the notebooks in `notebooks/`. Skip the first code cell when running against the local checkout.

```sh
python -m pip install .
```

To run the repository's smoke checks, including execution of both notebooks against the installed checkout, install the test dependencies and run:

```sh
python -m pip install '.[notebooks]'
python -m unittest discover -s tests -v
```
