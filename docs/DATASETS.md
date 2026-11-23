# Datasets

## X/O patterns

All five X/O networks use the same three text files in [`data/xo/`](../data/xo/).

| File | Samples | Shape per sample | X labels | O labels |
| --- | ---: | --- | ---: | ---: |
| `trainingset.txt` | 192 | 25 pixels + 1 target | 96 | 96 |
| `testset.txt` | 41 | 25 pixels + 1 target | 20 | 21 |
| `validationset.txt` | 41 | 25 pixels + 1 target | 21 | 20 |

A line contains 26 whitespace-separated integers. The first 25 values are the 5 × 5 image in row-major order, with each pixel encoded as `+1` (filled) or `-1` (empty). The 26th value is `+1` for X or `-1` for O. For example, the row's layout is:

```text
pixel_0 pixel_1 ... pixel_24 class_label
```

The Hebbian, perceptron and ADALINE examples use the training and test sets. The multilayer X/O example also uses the validation set for its stopping logic. The classifiers are evaluated on the supplied test split rather than on manually drawn GUI patterns.

When the GUI buttons **Enter new X** or **Enter new O** are used, the new sample is appended to the single shared training file. Restore that file from version control to return to the documented 192-sample data distribution.

## Thyroid classification

The two Excel files in [`data/thyroid/`](../data/thyroid/) contain samples arranged **by column**:

| File | Matrix shape | Meaning |
| --- | --- | --- |
| `thyroidInputs.xlsx` | 21 × 7200 | 21 numeric input features per patient/sample |
| `thyroidTargets.xlsx` | 3 × 7200 | Three-element one-hot target vectors |

The scripts transpose these matrices into 7200 samples before dividing them using positional slices:

- Training: first `int(0.70 × 7200) = 5040` samples.
- Testing: next `int(0.15 × 7200) = 1080` samples.
- Validation: remaining 1080 samples.

Target positions correspond to class 1 (normal), class 2 (hyperfunction), and class 3 (subnormal functioning). The original one-hot arrays are used directly as supervised targets. The code does not shuffle or stratify the observations before splitting.

## Boolean experiments

AND, OR and XOR examples embed small two-input truth tables directly in their Python files. They do not read from `data/xo/`.
