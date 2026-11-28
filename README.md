# Design Research Workshop

Three independent notebooks use Workshop PyBot to teach prompt framing, critique, and tool use in the [Design Research Python ecosystem](https://cmudrc.github.io/design-research/). They run on a CPU with no API key or model download. Each notebook includes editable inputs, the agent’s decisions, and a question you can answer from its output.

| Notebook | Classroom experiment | Open |
| --- | --- | --- |
| [One-handed jar opener](notebooks/one_handed_jar_brainstorm.ipynb) | Give the same chart of mechanisms to two brainstorming prompts. Count the mechanisms each prompt covers in its first three ideas. Then change a row on a new brief and check the returned concepts against it. | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/cmudrc/design-research-workshop/blob/workshop-three-agent-demos/notebooks/one_handed_jar_brainstorm.ipynb) |
| [Peanut-sheller proposer–critic](notebooks/peanut_sheller_propose_critic.ipynb) | Start with a tabletop sheller. Enter a provision in your own words. Inspect which requirements the critic recognizes and which edit is still pending when the round budget runs out. | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/cmudrc/design-research-workshop/blob/workshop-three-agent-demos/notebooks/peanut_sheller_propose_critic.ipynb) |
| [Planar-truss ReAct](notebooks/planar_truss_react.ipynb) | Ask for a lighter truss or a higher factor of safety. Compare the mass, safety, and deflection tradeoffs. Give the agent two incompatible bounds and inspect its best alternatives. | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/cmudrc/design-research-workshop/blob/workshop-three-agent-demos/notebooks/planar_truss_react.ipynb) |

## Run PyBot

Each notebook installs this review branch in its first Colab cell. For local use, install the package with Python 3.12 or newer and Git.

```sh
python -m pip install "design-research-workshop @ git+https://github.com/cmudrc/design-research-workshop.git@workshop-three-agent-demos"
design-research-workshop-pybot --list
design-research-workshop-pybot "Give me three one-handed jar opener ideas, function by function."
```

Run `design-research-workshop-pybot` without an argument for a conversation. Try “Propose a peanut sheller with rollers and a sieve”, followed by “Review it, especially the cleanout” and “Revise it”. Ask “Why?” to see the recorded evidence or “What changed?” to compare the last two runs. Asking for more concepts excludes those already shown. Use `/reset` to start a fresh conversation. Add `--json` to a one-shot command for machine-readable output.

The same rules power the terminal, the Python interface, and the local chat-completions endpoint. The bot takes inspiration from Hugging Face’s modular [PyBot](https://github.com/huggingface/pybot). Each workshop responder supplies its own search, critique, or tool-choice policy. The selector uses the system instructions, the structure of a request, subject phrases, and recent turns to choose a responder. It asks for a choice when two subjects match equally. An instructor can also select a rule explicitly, for example with `bot.reply("Give me three ideas.", rule="jar-opener")`. The truss responder delegates structural calculations to the packaged truss evaluator. The sheller critic checks provisions described in the concept. Testing whether a prototype protects kernels still requires a physical trial.

## Try another design brief

Start with “Brainstorm a garden downspout attachment to collect rainwater and filter debris.” PyBot recognizes two requested functions and suggests a chart of mechanisms. Each row records the words it recognized and the origin of the options. To override a row in a follow-up, use its displayed label, such as `Filter debris: removable basket | coarse screen`. When the brief gives too little information to select functions, PyBot asks for a chart.

A Python session keeps the participant’s turns together. These calls generate the first concepts, return two further concepts, and explain the selection from the recorded trace.

```python
from design_research_workshop import WorkshopPyBot

session = WorkshopPyBot().session()
session.reply("Give me three jar opener ideas, function by function.")
additional = session.reply("Give me two more ideas.")
print(additional.content["concepts"])
print(session.reply("Why?").content["explanation"])
```

You can also supply a complete chart directly. Give each function a row and separate its options with a vertical bar.

```python
from design_research_workshop import WorkshopPyBot

bot = WorkshopPyBot()
reply = bot.reply("""Brief: A library stand for wet umbrellas
Support the umbrella: hanging hooks | vertical slots
Collect drips: removable tray | absorbent mat
Give me two concepts, function by function.""")
print(reply.rule, reply.content["concepts"])
```

For a reusable workshop, give the chart a name and register it once. The same bot then works with the Design Research Agents client.

```python
from design_research_workshop import DesignBrief, brainstorm_rule, workshop_client
from design_research.agents import LLMMessage, LLMRequest

stand = DesignBrief(
    title="Umbrella stand",
    statement="Collect wet umbrellas at a shared entrance.",
    functions={
        "Support": ["hanging hooks", "vertical slots"],
        "Collect drips": ["removable tray", "absorbent mat"],
    },
)
bot = bot.with_rules(brainstorm_rule(stand, name="umbrella-stand", topics=("umbrella stand",)))
with workshop_client(bot=bot) as client:
    response = client.generate(LLMRequest(messages=[
        LLMMessage(role="user", content="Give me two umbrella stand ideas, function by function.")
    ]))
    print(response.text)
```

For another packaged ideation problem, `DesignBrief.from_problem(problem_id, functions)` loads its brief from `design-research-problems` and associates your mechanism chart with its problem ID. For a new proposer–critic exercise, define which provisions count as evidence and which revision addresses each missing provision. The [cable-organizer example](tests/test_peanut_rules.py) uses the sheller’s [review and revision functions](src/design_research_workshop/rubric.py) with a different design record and criteria.

To add a different kind of task, register a `Rule(name, description, topics, respond)`. Its `respond(turn)` function receives the current prompt and relevant history and returns a JSON-compatible dictionary. Optional protocol and follow-up matchers can recognize structured agent calls and updates within a conversation. Creating an extended bot leaves other clients’ registries independent. See the executable transfer cases in [the PyBot tests](tests/test_pybot.py).

## Run the notebooks locally

From the checkout, install the notebook dependencies, open a notebook in Jupyter, and skip the Colab install cell. The tests execute every notebook in a fresh working directory.

```sh
python -m pip install '.[notebooks]'
python -m unittest discover -s tests -v
```

For a separately running local endpoint, start `design-research-workshop-proxy --port 8001` and use the `workshop-pybot-v1` model at `http://127.0.0.1:8001/v1`. It accepts nonstreaming text chat completions. The [initial laptop workshop](https://github.com/cmudrc/design-research-workshop/tree/v0.1.0) and its catalog exercise remain available.
