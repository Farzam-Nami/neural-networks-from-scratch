# Comparison of X/O classifiers — Task 8

The common X/O dataset enables five algorithms to be compared on the same bipolar 5 × 5 patterns. Each classifier receives 25 input pixels and targets `+1` (X) and `-1` (O).

| Task | Algorithm | Outputs | Training criterion | Evaluation |
| --- | --- | --- | --- | --- |
| 2 | Hebbian | One signed score | One pass through the training set | `hebbianXO_test.py` |
| 4 | Single-layer perceptron | One three-way bipolar output | Epoch with no weight/bias update | `perceptronXO_test.py` |
| 5 | Multi-category perceptron | Two complementary neurons | Epoch with no weight/bias update | `MCperceptronXO_test.py` |
| 6 | ADALINE | One thresholded linear score | Small per-weight difference between epochs | `adalinXO_test.py` |
| 7 | Multilayer perceptron | One sigmoid output | Validation-error and parameter-change conditions | `MLperceptronXO_test.py` |

## Comparable measurements

Each evaluation script prints counts converted to percentages relative to the **entire test set** of 41 samples:

```text
True X percentage  = 100 * correct_X / 41
True O percentage  = 100 * correct_O / 41
False X percentage = 100 * incorrect_X_predictions / 41
False O percentage = 100 * incorrect_O_predictions / 41
Total true percentage = 100 * (correct_X + correct_O) / 41
```

These are percentages of the full test set, **not** per-class precision or recall. A classifier can also return `0` (unknown) or, for Hebbian, no value when its score is exactly zero; such results do not contribute to the four X/O counts.

For a comparable run, train each of Tasks 2, 4, 5 and 6, then run its matching `_test.py` script. Task 7's `_test.py` performs both training and evaluation. The shared test set is never used by the training functions. Results of randomly initialized networks can vary between runs; report measured values from the actual run rather than assuming one fixed score.

## Results from the saved parameter files

With the stored parameter files included in the exercise folders, the test scripts report:

| Classifier | Correct test samples | Overall correctness |
| --- | ---: | ---: |
| Hebbian | 33 / 41 | 80.49% |
| Single-layer perceptron | 35 / 41 | 85.37% |
| Multi-category perceptron | 35 / 41 | 85.37% |
| ADALINE | 37 / 41 | 90.24% |
| Multilayer perceptron | Independently trained by its test script | Depends on initial random weights |

These numbers describe the stored parameters and the provided 41-sample test set, not an average across random seeds or a claim of algorithm-wide superiority.

## Thyroid / MATLAB comparison — Task 9

The thyroid task separately requests a comparison with MATLAB's Neural Network Toolbox. Report at least test accuracy and elapsed training time, and record the exact MATLAB architecture, split, epoch limit and random initialization method. The Python implementation uses a 21–5–3 architecture, positional 70/15/15 split, standard sigmoid and online backpropagation. Direct comparisons require an explicitly matched MATLAB setup; the repository does not assert MATLAB accuracy or timing figures without a corresponding reproducible run.
