# Verification and tests

## Automated test command

From the repository root, after installing `requirements.txt`:

```bash
python -m unittest discover -s tests -v
```

**10 automated tests pass**, including:

- The bipolar perceptron activation function, bipolar sigmoid value/derivative, and logistic sigmoid monotonicity.
- Shape, labels and pixel dimensions of the X/O datasets (192 training, 41 test, 41 validation samples).
- All X/O scripts with dataset-path variables resolving to the same shared dataset directory.
- Thyroid feature and target matrix dimensions and one-hot target consistency.
- Saved multi-category perceptron parameter row dimensions (two output neurons).
- Saved thyroid network parameter dimensions (15, 105, 5 and 3 numeric values).
- Hebbian learning/prediction on simple bipolar vectors.
- Multi-category perceptron class decisions using both output neurons.

All Python source files also pass Python syntax compilation.

## Reproducing exercise outputs

1. Install the dependencies as described in the project README.
2. Launch the desired training script from the `exercises/` directory structure using the root-relative commands in the README.
3. For Tasks 2, 4, 5 and 6, run the matching `_test.py` program after training so its saved weights are available. Task 7's `_test.py` trains and evaluates a separate network.
4. For Task 9, run `Thyroid_MLP.py` before `Thyroid_MLP_test.py` to produce and load the model parameter file.

For Tasks 2–7, the test scripts print per-category true/false counts as fractions of the 41 total test examples and an overall correctness percentage. The thyroid evaluator displays a three-class prediction/target table and overall test correctness. Stochastic experiments may yield different outcomes across runs.

## Data integrity checks

Each X/O record contains 25 bipolar features and one bipolar label. The thyroid inputs contain 7200 columns with 21 features, and the corresponding targets contain 7200 columns with three one-hot class values. The TSP dataset belongs to the separate Genetic Algorithms project.

## Recorded evaluation of saved models

Running the supplied evaluators against their saved weights gives 80.49% (Hebbian), 85.37% (single-layer perceptron), 85.37% (multi-category perceptron) and 90.24% (ADALINE) total test correctness on the 41 X/O test examples. The saved thyroid model gives 92.41% test correctness on 1080 thyroid samples. These values are observations for the included parameter files; retraining may produce other parameters and results.
