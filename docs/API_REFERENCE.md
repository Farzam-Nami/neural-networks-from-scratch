# Python module reference

The programs are plain Python scripts rather than an installable Python package. Training and GUI setup execute when the corresponding application script is run. The callable functions below document the learning rules and their inputs; most use the named module-level weight/bias arrays.

## Task 1 — Hebbian AND/OR

Files: [`hebbianAND.py`](../exercises/1_hebbianAND/hebbianAND.py), [`hebbianOR.py`](../exercises/1_hebbianAND/hebbianOR.py)

- `train(vector)` — Accepts one three-value record `[x1, x2, target]`. Computes a new pair of Hebbian weights and one bias from the current module-level `w1`, `w2`, `b`; appends these to `decisionBoundaries` and returns `[new_w1, new_w2, new_b]`.
- `showBoundaries()` — Draws the recorded decision boundary after each of the four bipolar patterns. The user closes a figure to show the next one.

## Task 2 — Hebbian X/O

File: [`hebbianXO.py`](../exercises/2_hebbianXO/hebbianXO.py)

- `train(set)` — Iterates over training records (25 pixels, one bipolar label), applies `w[j] += x[j]*target` and `b += target`, prints a training message, and returns `(w,b)`.
- `predict(Xin)` — Accepts 25 bipolar pixels; returns `+1` or `-1` from the sign of the weighted sum, or `None` when the sum is exactly zero.

File: [`hebbianXO_test.py`](../exercises/2_hebbianXO/hebbianXO_test.py)

- `predict(Xin)` — Applies saved weights/bias to the first 25 entries of the input. The script computes a test-set X/O count and training-set correctness.

## Task 3 — Perceptron AND

File: [`perceptronAND.py`](../exercises/3_perceptronAND/perceptronAND.py)

- `fY_IN(y)` — Bipolar threshold function: `+1` above `theta`, `-1` below `-theta`, and `0` otherwise.
- `train(vector)` — Trains on one two-input record using an error-triggered perceptron rule, modifying the module's `w1`, `w2`, `b`.
- `showBoundary()` — Plots the two threshold boundaries and the labeled Boolean examples.

## Task 4 — Single-layer perceptron X/O

Files: [`perceptronXO.py`](../exercises/4_perceptronXO/perceptronXO.py), [`perceptronXO_test.py`](../exercises/4_perceptronXO/perceptronXO_test.py)

- `fY_IN(y)` — Three-way bipolar threshold with `theta=0.5`.
- `train(set)` — Updates 25 weights and a bias when an example's predicted class differs from its target. Writes the final 26 parameters to `perceptronXO_results.txt`.
- `predict(Xin)` — Predicts one X/O sample using the learned parameters, returning `+1`, `-1`, or `0`.
- The evaluation script's `predict(Xin)` loads the parameter vector and uses the same numeric thresholds.

## Task 5 — Multi-category perceptron X/O

Files: [`MCperceptronXO.py`](../exercises/5_MCperceptronXO/MCperceptronXO.py), [`MCperceptronXO_test.py`](../exercises/5_MCperceptronXO/MCperceptronXO_test.py)

- `fY_IN(y)` — Applies the bipolar threshold to one output-neuron input.
- `train(set)` — Trains two 25-element weight vectors using complementary labels `[target, -target]`. Saves a 26-number line per output neuron in `MCperceptronXO_results.txt`.
- `predict(Xin)` — Produces a class only if both output neurons agree on the complementary pattern; otherwise returns `0`.
- The evaluation script's `predict(Xin)` reads both saved parameter rows and applies the same two-neuron class rule.

## Task 6 — ADALINE X/O

Files: [`adalineXO.py`](../exercises/6_adalineXO/adalineXO.py), [`adalinXO_test.py`](../exercises/6_adalineXO/adalinXO_test.py)

- `fY_IN(y)` — Returns `+1` for nonnegative input and `-1` for negative input.
- `train(set)` — Applies the ADALINE delta rule to each example using the continuous net input. Stops when the 25 learned weights change by less than `0.001` across an epoch. Saves parameters to `adalineXO_results.txt`.
- `predict(Xin)` — Applies a threshold to the weighted sum of one 25-pixel input.
- The evaluation script's `predict(Xin)` evaluates the parameter file against the common test records.

## Task 7 — Multilayer perceptron X/O

Files: [`MLperceptronXO.py`](../exercises/7_MLperceptronXO/MLperceptronXO.py), [`MLperceptronXO_test.py`](../exercises/7_MLperceptronXO/MLperceptronXO_test.py)

- `fNI(yNI)` — Bipolar sigmoid applied to a scalar or NumPy value.
- `fprime_NI(yNI)` — Derivative used to propagate the output error backward.
- `predict(Xin)` — Propagates 25 pixels through the hidden and output layers; returns `+1`, `-1` or `0`.
- `train(set)` — Performs online two-layer backpropagation, updates arrays `v`, `w`, `b0`, `b1` and checks validation error after each epoch. The evaluation module performs training before testing.

## Task 9 — Thyroid multilayer perceptron

Files: [`Thyroid_MLP.py`](../exercises/9_MLPthyroid/Thyroid_MLP.py), [`Thyroid_MLP_test.py`](../exercises/9_MLPthyroid/Thyroid_MLP_test.py)

- `fNI(yNI)` — Logistic sigmoid `1/(1+exp(-yNI))`.
- `fprime_NI(yNI)` — Derivative `f(yNI)*(1-f(yNI))`.
- `predict(Xin)` — Accepts 21 input features, computes five hidden activations and three output activations, then returns class `1`, `2`, or `3` for the largest output (or `None` for an exact tie).
- `target_find(target_case)` / `targetFind(targetCase)` — Maps a three-value one-hot target to class `1`, `2`, or `3`; the two scripts use their original name styles.
- `train(set,targets)` — Trains using aligned input/target lists, applies online backpropagation, checks the validation subset, and stops according to its epoch/weight-change/error conditions.

The training script writes `results.txt`: first line `5×3` output-layer weights; second line `21×5` input-layer weights; third and fourth lines hold five hidden biases and three output biases. The evaluation script reconstructs these arrays before computing the test confusion counts.

## Additional XOR model

File: [`mlpXOR.py`](../exercises/XOR/mlpXOR.py)

- `fNI(yNI)`, `fprime_NI(yNI)` — Bipolar sigmoid and derivative.
- `predict(Xin)` — Returns the bipolar predicted class for two input values.
- `train(set)` — Trains the two-hidden-neuron network using online backpropagation until the parameter-change threshold is met.
- `showBoundary()` — Draws the two hidden-neuron boundaries and the XOR training samples.

## X/O GUI callbacks

The X/O training applications (Tasks 2, 4, 5, 6 and 7) each define the following callbacks in their own file:

| Function | Behavior |
| --- | --- |
| `buttonClick(row,col)` | Toggles a single cell and tracks its zero-based `(row,col)` location in `changedButtons`. |
| `predictButtons()` | Converts the 25 cells to bipolar values and invokes that model's `predict()`. |
| `resetButtons()` | Clears the visible grid and the list of selected coordinates. |
| `storeChanges_x()` | Appends a 25-pixel sample with target `+1` to the shared training file. |
| `storeChanges_o()` | Appends a 25-pixel sample with target `-1` to the shared training file. |
| `createButton(row,col)` | Creates and places one Tkinter cell button in the 5 × 5 grid. |

The callbacks refer to the module-level Tkinter widgets and are intended to run in the matching GUI application, not as standalone library functions. The core learning rules remain in each exercise script so their update loops can be inspected separately.
