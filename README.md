# Neural Networks from Scratch

Python implementations of the neural-network exercises in the **Computational Intelligence** programming coursework. The experiments use explicit loops for weight updates, simple activation functions, and a 5 × 5 Tkinter interface for X/O pattern recognition.

## Exercises

| Task | Folder | Network | Input and output |
| --- | --- | --- | --- |
| 1 | `exercises/1_hebbianAND/` | Hebbian AND and OR | Two bipolar inputs, one bipolar target; decision boundaries plotted after each pattern |
| 2 | `exercises/2_hebbianXO/` | Hebbian classifier | 25 bipolar pixels; X = +1, O = −1 |
| 3 | `exercises/3_perceptronAND/` | Single-layer perceptron | Two binary inputs, one bipolar target; `alpha = 1`, `theta = 0.2` |
| 4 | `exercises/4_perceptronXO/` | Single-layer perceptron | 25 bipolar pixels; one output |
| 5 | `exercises/5_MCperceptronXO/` | Two-output multi-category perceptron | 25 bipolar pixels; two complementary targets |
| 6 | `exercises/6_adalineXO/` | ADALINE | 25 bipolar pixels; linear learning rule with bipolar final decision |
| 7 | `exercises/7_MLperceptronXO/` | Multilayer perceptron | 25 inputs, 8 hidden neurons, one output; backpropagation |
| 8 | `docs/XO_COMPARISON.md` | X/O network comparison | Models from Tasks 2, 4, 5, 6 and 7 |
| 9 | `exercises/9_MLPthyroid/` | Multilayer perceptron | 21 inputs, 5 hidden neurons, three outputs; thyroid classification |
| Additional | `exercises/XOR/` | Multilayer perceptron | Two bipolar inputs; XOR classification |

The numbered exercises are standalone programs. Each retains its own learning rule, stopping criterion, and user interface, so each experiment can be studied independently.

## Requirements

- Python 3.10 or later
- NumPy, Matplotlib, pandas and openpyxl (`requirements.txt`)
- Tkinter (included with most Windows Python installations; on some Linux distributions it is a separate system package)
- A graphical desktop to open the Tkinter windows and Matplotlib figures

Install the Python dependencies at the repository root:

```bash
python -m pip install -r requirements.txt
```

## How to run

Commands below are executed from the repository root. The data file paths are resolved from each Python file, so the current working directory does not determine which dataset is loaded.

### Boolean functions

```bash
python exercises/1_hebbianAND/hebbianAND.py
python exercises/1_hebbianAND/hebbianOR.py
python exercises/3_perceptronAND/perceptronAND.py
python exercises/XOR/mlpXOR.py
```

These programs train their networks and draw the relevant decision boundaries. The plotting windows can be closed to continue execution.

### X/O pattern recognition

Run one training application, then its matching evaluation script. For example:

```bash
python exercises/4_perceptronXO/perceptronXO.py
python exercises/4_perceptronXO/perceptronXO_test.py
```

Other training/evaluation pairs:

```bash
python exercises/2_hebbianXO/hebbianXO.py
python exercises/2_hebbianXO/hebbianXO_test.py

python exercises/5_MCperceptronXO/MCperceptronXO.py
python exercises/5_MCperceptronXO/MCperceptronXO_test.py

python exercises/6_adalineXO/adalineXO.py
python exercises/6_adalineXO/adalinXO_test.py

python exercises/7_MLperceptronXO/MLperceptronXO.py
python exercises/7_MLperceptronXO/MLperceptronXO_test.py
```

In the GUI, click pixels in the 5 × 5 grid and select **Predict** to classify a pattern. **Reset** clears the grid. **Enter new X** and **Enter new O** append a labeled pattern to `data/xo/trainingset.txt`. This file is shared by all X/O experiments, so a new sample will be available to every X/O model when it next trains. The application displays a restart notice because the current weights are not automatically retrained after appending a sample.

The first four X/O training programs save their weights and bias to a `*_results.txt` file in the same exercise folder. Their evaluation scripts load those parameters. Task 7's evaluation script trains its own model and therefore does not require a saved weight file.

### Thyroid classification

```bash
python exercises/9_MLPthyroid/Thyroid_MLP.py
python exercises/9_MLPthyroid/Thyroid_MLP_test.py
```

Training reads the 21-feature input matrix and three-class target matrix from `data/thyroid/`, writes parameters to `exercises/9_MLPthyroid/results.txt`, and displays elapsed training time. Evaluation reads the saved parameters and reports the test-set classification distribution. The data is split by the existing row order: 70% train, 15% test and the remaining observations for validation.

The multilayer networks start with random weights; repeated runs can therefore produce different stopping epochs and accuracy values. No test-set results are asserted as fixed benchmark scores.

## Repository layout

```text
neural-networks-from-scratch/
├── data/
│   ├── xo/
│   │   ├── trainingset.txt
│   │   ├── testset.txt
│   │   └── validationset.txt
│   └── thyroid/
│       ├── thyroidInputs.xlsx
│       └── thyroidTargets.xlsx
├── exercises/
│   ├── 1_hebbianAND/
│   ├── 2_hebbianXO/
│   ├── 3_perceptronAND/
│   ├── 4_perceptronXO/
│   ├── 5_MCperceptronXO/
│   ├── 6_adalineXO/
│   ├── 7_MLperceptronXO/
│   ├── 9_MLPthyroid/
│   └── XOR/
├── tests/
├── docs/
└── requirements.txt
```

## Documentation

- [Algorithms and learning rules](docs/ALGORITHMS.md)
- [Datasets and encoding](docs/DATASETS.md)
- [X/O model comparison](docs/XO_COMPARISON.md)
- [Functions and modules](docs/API_REFERENCE.md)
- [Test instructions and verification](docs/TEST_REPORT.md)

Run automated tests from the repository root:

```bash
python -m unittest discover -s tests -v
```
