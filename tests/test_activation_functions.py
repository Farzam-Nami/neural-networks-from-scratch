"""Basic tests of neuron activation functions used in the course exercises."""

import ast
from pathlib import Path
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1] / "exercises"


def load_functions(filename, names, **values):
    """Extract function definitions without running the interactive application."""
    source = ast.parse(filename.read_text(encoding="utf-8"))
    methods = [part for part in source.body if isinstance(part, ast.FunctionDef) and part.name in names]
    module = ast.Module(body=methods, type_ignores=[])
    scope = {"np": np, **values}
    exec(compile(module, str(filename), "exec"), scope)
    return scope


class ActivationTests(unittest.TestCase):
    def test_perceptron_bipolar_threshold(self):
        functions = load_functions(ROOT / "3_perceptronAND" / "perceptronAND.py", {"fY_IN"}, theta=0.2)
        self.assertEqual(functions["fY_IN"](0.5), 1)
        self.assertEqual(functions["fY_IN"](-0.5), -1)
        self.assertEqual(functions["fY_IN"](0.0), 0)

    def test_sigmoid_center(self):
        functions = load_functions(ROOT / "XOR" / "mlpXOR.py", {"fNI", "fprime_NI"})
        self.assertAlmostEqual(functions["fNI"](0.0), 0.0)
        self.assertAlmostEqual(functions["fprime_NI"](0.0), 0.5)

    def test_sigmoid_monotonicity(self):
        functions = load_functions(ROOT / "9_MLPthyroid" / "Thyroid_MLP.py", {"fNI"})
        self.assertLess(functions["fNI"](-2), functions["fNI"](0))
        self.assertLess(functions["fNI"](0), functions["fNI"](2))


if __name__ == "__main__":
    unittest.main()
