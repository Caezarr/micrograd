# micrograd (course exercise)

## What this is

A learning log for lecture 1 of Andrej Karpathy's [Neural Networks: Zero to Hero](https://karpathy.ai/zero-to-hero.html): [The spelled-out intro to neural networks and backpropagation: building micrograd](https://www.youtube.com/watch?v=VMj-3S1tku0).

This is not original work. The design and most of the code follow the lecture and Karpathy's reference implementation, [karpathy/micrograd](https://github.com/karpathy/micrograd) (MIT). I typed it along with the video, ran the experiments and fixed my own mistakes along the way.

## What I implemented

In `micrograd.ipynb`, in order:

1. **Numerical derivatives**: the slope of `3x^2 - 4x + 5` and of `d = a*b + c` by finite differences.
2. **`Value`**: a scalar autograd node that records its parents and the op that produced it, with `+`, `*`, `**`, `-`, `/`, `tanh` and `exp`, each with its local `_backward` rule.
3. **Backpropagation**: `backward()` topologically sorts the graph and applies the chain rule from the output back to the leaves.
4. **Visualisation**: `draw_dot` renders the computation graph with data and gradients through Graphviz.
5. **A single neuron**: `tanh(x1*w1 + x2*w2 + b)`, computed with `tanh` directly and again composed from `exp`, then checked against PyTorch autograd.
6. **A small MLP**: `Neuron`, `Layer` and `MLP` classes (3 inputs, two hidden layers of 4, one output) trained by gradient descent on a 4-example toy dataset.

## What I learned

- Backprop is the chain rule applied recursively over a DAG: each op only needs its local derivative, multiplied by the gradient flowing in.
- Gradients must accumulate (`+=`) because a node can feed several others. My first version of `exp()` assigned its gradient (`=`) instead; the fix is in this notebook.
- Gradients must be zeroed before every backward pass, otherwise they add up across training steps.
- Building an op from smaller ones (`tanh` from `exp`, `/` from `**-1`) gives the same gradients, which is a cheap correctness check. For the neuron, both versions and PyTorch agree: `x1.grad = -1.5`, `w1.grad = 1.0`, `x2.grad = 0.5`, `w2.grad = 0`.
- The training loop is forward pass, zero grads, backward, update. In one run the squared-error loss went from 6.6 to under 0.01 in 20 steps at learning rate 0.1, not monotonically (it bounced back up around step 5). Exact values vary with the random initial weights.

## Run it

Python 3.10+ and the Graphviz system package (`brew install graphviz` on macOS, `apt install graphviz` on Debian/Ubuntu).

```bash
git clone https://github.com/Caezarr/micrograd.git && cd micrograd
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook micrograd.ipynb
```

Outputs are not committed: use Kernel > Restart & Run All. To execute it headless instead:

```bash
jupyter nbconvert --to notebook --execute micrograd.ipynb --output executed.ipynb
```

This runs in a few seconds on a laptop CPU. The training cell prints a loss that should fall towards zero.

## Credits

All credit for the course, the micrograd design and the teaching goes to [Andrej Karpathy](https://karpathy.ai). Lecture: <https://www.youtube.com/watch?v=VMj-3S1tku0>. Reference code: <https://github.com/karpathy/micrograd>.
