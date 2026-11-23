# Neural-network algorithms

These programs implement learning rules explicitly with scalar or NumPy-array updates. The X/O applications use a 25-dimensional bipolar input vector; the Boolean examples use two inputs.

## 1. Hebbian AND and OR

The training patterns are bipolar (`-1`, `+1`). Starting with zero weights and bias, each presented example `(x, t)` updates:

```text
w_i <- w_i + x_i * t
b   <- b + t
```

After each input vector, the program records the updated decision boundary and draws a new plot. A line with `w_2 != 0` satisfies `x_2 = -(w_1/w_2)x_1 - b/w_2`. The OR implementation also handles vertical boundaries (`w_2 = 0`).

## 2. Hebbian X/O

The Hebbian classifier applies the same weight update to all 192 training samples. The prediction uses the sign of the weighted sum plus bias: positive for X, negative for O. The script returns no class at an exact zero score. It stores 25 weights and one bias as whitespace-separated numbers in `hebbianXO_results.txt`.

## 3–4. Single-layer perceptrons

The single-layer perceptrons use a bipolar three-way threshold:

```text
f(y) = +1  if y > theta
        -1  if y < -theta
         0  otherwise
```

When the prediction differs from the target, the perceptron updates each weight by `alpha * x_i * target`, and the bias by `alpha * target`. The AND example uses `alpha=1`, `theta=0.2`; the X/O example uses `alpha=0.1`, `theta=0.5`. Training finishes when a complete epoch leaves the weights and bias unchanged.

## 5. Multi-category perceptron

There are **two output neurons**, each with 25 weights. For a target `t`, the desired outputs are `(t, -t)`. Each neuron follows the same perceptron-style update with `alpha=0.1` and `theta=0.5`. The classifier returns X only for outputs `(+1,-1)` and O only for `(-1,+1)`; other output combinations produce the unknown class `0`. Two rows of 26 numeric parameters are saved in `MCperceptronXO_results.txt`.

## 6. ADALINE

The model uses a linear net input `y_in = b + sum(w_i*x_i)` during training. For each training pair, the update is:

```text
error = target - y_in
w_i  <- w_i + alpha * error * x_i
b    <- b + alpha * error
```

The learning rate is `alpha=0.001`. A final step function at zero converts the net input to `+1` or `-1`. The loop stops when every absolute change in the weight vector is smaller than `0.001` after an epoch.

## 7. Two-layer X/O perceptron

This network has **25 input features, 8 hidden neurons and one output neuron**. Both hidden and output units use the bipolar sigmoid

`f(z) = 2/(1+exp(-z)) - 1`, with derivative `f'(z) = (1+f(z))(1-f(z))/2`.

The script implements online backpropagation using loops over input, hidden and output neurons. The learning rate is `alpha=0.2`. The network classifies positive output as X and negative output as O. After each training epoch, it counts classification errors on `validationset.txt`. The training loop also checks changes in the weight and bias arrays and a fixed no-improvement counter. The evaluation script independently trains a network, so its random initialization may give a different outcome from the GUI application.

## 8. X/O evaluation

The comparison concerns the five X/O learning rules in Tasks 2, 4, 5, 6 and 7. Scripts print true and false X/O predictions as percentages of all test observations, plus overall test correctness. See [X/O comparison](XO_COMPARISON.md) for the interpretation of these quantities and the run sequence.

## 9. Thyroid multilayer perceptron

The architecture uses **21 input features, 5 sigmoid hidden neurons and 3 sigmoid outputs**. All targets are three-element one-hot vectors. Both layers use the standard logistic sigmoid and its derivative; online backpropagation updates weights and biases with `alpha=0.1`.

The predicted class is the index of the largest output activation (1, 2 or 3); the target class is the index of the largest target component. Training checks the validation subset after every epoch and stops based on its existing error/weight-change conditions or at 1000 epochs. The model parameters are saved in four rows: output weights (`5×3`), input weights (`21×5`), five hidden biases, and three output biases.

## Additional XOR example

A two-input, two-hidden-neuron, single-output network uses bipolar sigmoid activations and online backpropagation. The four labeled patterns are built into the file. The script draws the two learned hidden-layer decision boundaries after training.
