"""Checks input formats and representative model operations without opening Tk windows."""

import ast
import unittest
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def load_functions(script, names, **variables):
    """Read functions from the student's standalone scripts without launching GUIs."""
    tree = ast.parse(script.read_text(encoding="utf-8"))
    methods = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    scope = {"np": np, **variables}
    exec(compile(ast.Module(body=methods, type_ignores=[]), str(script), "exec"), scope)
    return scope


class DatasetTests(unittest.TestCase):
    def test_xo_files_have_25_pixels_and_one_label(self):
        expected = {"trainingset.txt": 192, "testset.txt": 41, "validationset.txt": 41}
        for name, count in expected.items():
            with self.subTest(dataset=name):
                file = ROOT / "data" / "xo" / name
                records = [[int(number) for number in line.split()] for line in file.read_text().splitlines()]
                self.assertEqual(len(records), count)
                self.assertTrue(all(len(row) == 26 for row in records))
                self.assertTrue(all(set(row) <= {-1, 1} for row in records))

    def test_xo_script_dataset_paths_are_shared(self):
        shared = (ROOT / "data" / "xo").resolve()
        for script in sorted((ROOT / "exercises").rglob("*.py")):
            tree = ast.parse(script.read_text(encoding="utf-8"))
            for node in tree.body:
                if not isinstance(node, ast.Assign) or len(node.targets) != 1:
                    continue
                target = node.targets[0]
                if not isinstance(target, ast.Name) or target.id not in (
                    "trainSetPath", "trainingSetPath", "testSetPath", "validationSetPath"
                ):
                    continue
                scope = {"Path": Path, "__file__": str(script)}
                path = eval(compile(ast.Expression(node.value), str(script), "eval"), scope)
                self.assertEqual(path.parent.resolve(), shared, f"{script}: {target.id}")
                self.assertTrue(path.is_file())

    def test_thyroid_files_have_matching_samples(self):
        inputs = pd.read_excel(ROOT / "data" / "thyroid" / "thyroidInputs.xlsx", header=None)
        targets = pd.read_excel(ROOT / "data" / "thyroid" / "thyroidTargets.xlsx", header=None)
        self.assertEqual(inputs.shape, (21, 7200))
        self.assertEqual(targets.shape, (3, 7200))
        self.assertTrue(np.all(targets.to_numpy().sum(axis=0) == 1))

    def test_saved_multicategory_model_has_two_output_neurons(self):
        file = ROOT / "exercises" / "5_MCperceptronXO" / "MCperceptronXO_results.txt"
        rows = [[float(value) for value in line.split()] for line in file.read_text().splitlines()]
        self.assertEqual(len(rows), 2)
        self.assertTrue(all(len(row) == 26 for row in rows))

    def test_saved_thyroid_model_dimensions(self):
        file = ROOT / "exercises" / "9_MLPthyroid" / "results.txt"
        rows = [[float(value) for value in line.strip().split(",") if value] for line in file.read_text().splitlines()]
        self.assertEqual([len(row) for row in rows], [5 * 3, 21 * 5, 5, 3])


class ModelTests(unittest.TestCase):
    def test_hebbian_and_rule_on_simple_samples(self):
        funcs = load_functions(
            ROOT / "exercises" / "2_hebbianXO" / "hebbianXO.py",
            {"train", "predict"}, w=np.zeros(25), b=0
        )
        examples = [[1] * 25 + [1], [-1] * 25 + [-1]]
        weight, bias = funcs["train"](examples)
        self.assertTrue(np.all(weight == 2))
        self.assertEqual(bias, 0)
        self.assertEqual(funcs["predict"]([1] * 25), 1)
        self.assertEqual(funcs["predict"]([-1] * 25), -1)

    def test_multicategory_prediction_uses_both_neurons(self):
        f = load_functions(
            ROOT / "exercises" / "5_MCperceptronXO" / "MCperceptronXO.py",
            {"fY_IN", "predict"},
            w=[np.ones(25), -np.ones(25)], b=np.zeros(2), theta=0.5
        )
        self.assertEqual(f["predict"]([1] * 25), 1)
        self.assertEqual(f["predict"]([-1] * 25), -1)
        self.assertEqual(f["predict"]([0] * 25), 0)


if __name__ == "__main__":
    unittest.main()
