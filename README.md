# Design Research Workshop

Three hands-on notebooks for the [Design Research Python ecosystem](https://cmudrc.github.io/design-research/). Run the cases one at a time, inspect the outputs, then change a prompt, a review priority, or a proposed action. The Python rules run on a CPU and require no API key or model weights.

| Notebook | Try this | Open |
| --- | --- | --- |
| [Peanut-sheller brainstorm](notebooks/peanut_sheller_brainstorm.ipynb) | Run the direct and function-first prompts on one brief. Compare the mechanism choices and export an idea. | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/cmudrc/design-research-workshop/blob/workshop-three-agent-demos/notebooks/peanut_sheller_brainstorm.ipynb) |
| [Peanut-sheller proposer–critic](notebooks/peanut_sheller_propose_critic.ipynb) | Import the idea or use the default. Trace how the proposal changes after feedback on kernel protection and cleaning. | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/cmudrc/design-research-workshop/blob/workshop-three-agent-demos/notebooks/peanut_sheller_propose_critic.ipynb) |
| [Planar-truss ReAct](notebooks/planar_truss_react.ipynb) | Build a triangle through legal grammar edits, then give the agent a rejected first move and inspect how it recovers. | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/cmudrc/design-research-workshop/blob/workshop-three-agent-demos/notebooks/planar_truss_react.ipynb) |

Run the two sheller notebooks in order to follow one concept from idea generation through refinement. The truss notebook is a standalone exercise. The brainstorm compares authored search rules, and the critic checks whether the concept includes provisions for two requirements. Demonstrating a physical design’s performance would require prototype tests.

## Install the workshop source

Use Python 3.12 or newer and Git. Each notebook installs the review branch in its first Colab code cell.

```sh
python -m pip install "design-research-workshop @ git+https://github.com/cmudrc/design-research-workshop.git@workshop-three-agent-demos"
```

The `design_research_workshop` package supplies the exercise rules and a context manager that starts the local HTTP proxy. The notebooks call the Design Research Agents interface. The truss tools operate on the packaged Planar Truss grammar, and the sheller brief adapts a packaged design problem.

## Run locally

Clone the review branch and install the package into your Jupyter environment. Open a notebook in `notebooks/` and skip the Colab install cell.

```sh
python -m pip install .
```

To execute the notebooks and check the example workflows in a fresh environment, run:

```sh
python -m pip install '.[notebooks]'
python -m unittest discover -s tests -v
```

The [initial laptop workshop](https://github.com/cmudrc/design-research-workshop/tree/v0.1.0) remains available at `v0.1.0`.
