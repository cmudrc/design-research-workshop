"""Execute the participant notebooks against the installed workshop package."""

from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

import nbformat
from nbclient import NotebookClient

NOTEBOOK_DIR = Path(__file__).resolve().parents[1] / "notebooks"


class NotebookTests(TestCase):
    def test_notebooks_execute(self) -> None:
        expected_outputs = {
            "laptop_prompt_study.ipynb": ("predicted market share",),
            "laptop_propose_critic.ipynb": ("Critic: Approved",),
            "one_handed_jar_brainstorm.ipynb": (
                "Mechanism coverage",
                "4 of 4 concepts match the allowed options",
            ),
            "peanut_sheller_propose_critic.ipynb": (
                "Cleaning access [unrecognized]",
                "Critic: Approved",
                "Next change, still unapplied",
            ),
            "planar_truss_react.ipynb": (
                "Reported design:",
                "The runs installed the same members.",
                "no feasible completion",
                "Lightest layout meeting the safety floor: 70.89 kg",
            ),
        }
        self.assertEqual(
            {path.name for path in NOTEBOOK_DIR.glob("*.ipynb")},
            expected_outputs.keys(),
        )
        for name, expected in expected_outputs.items():
            with self.subTest(notebook=name), TemporaryDirectory() as workdir:
                notebook = nbformat.read(NOTEBOOK_DIR / name, as_version=4)
                NotebookClient(
                    notebook,
                    timeout=120,
                    kernel_name="python3",
                    skip_cells_with_tag="colab-install",
                    resources={"metadata": {"path": workdir}},
                ).execute()
                output = "".join(
                    item.get("text", "")
                    for cell in notebook.cells
                    for item in cell.get("outputs", [])
                )
                for item in expected:
                    self.assertIn(item, output)
