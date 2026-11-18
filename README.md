# micrograd (course exercise)

My work-through of the first lecture of Andrej Karpathy's **[Neural Networks: Zero to Hero](https://karpathy.ai/zero-to-hero.html)** course: *[The spelled-out intro to neural networks and backpropagation: building micrograd](https://www.youtube.com/watch?v=VMj-3S1tku0)*.

This is a learning exercise, not original work. The design and most of the code follow the lecture and Karpathy's reference implementation, [karpathy/micrograd](https://github.com/karpathy/micrograd) (MIT). I typed it along with the video, ran the experiments and fixed my own mistakes along the way.

## What the notebook covers

`micrograd.ipynb`, in order:

1. **Derivatives numerically**: slope of a scalar function and of `d = a*b + c` by finite differences.
2. **`Value`**: a scalar autograd node that records its parents and the operation that produced it, with `+`, `*`, `**`, `-`, `/`, `tanh` and `exp`, each with its local `_backward` rule.
3. **Backpropagation**: `backward()` builds a topological order of the graph and applies the chain rule from the output back to the leaves.
4. **Visualisation**: `draw_dot` renders the computation graph with data and gradients using Graphviz.
5. **A single neuron**: `tanh(x1*w1 + x2*w2 + b)` computed directly and by composing `exp`, with gradients checked against **PyTorch autograd**.
6. **A small MLP**: `Neuron`, `Layer` and `MLP` classes, then a gradient-descent loop on a 4-example toy dataset (forward pass, zero grads, backward, update).

## What I learned

- Backprop is the chain rule applied recursively over a DAG: each op only needs its local derivative, multiplied by the gradient flowing in.
- Gradients must **accumulate** (`+=`) because a node can feed several others. My first `exp()` overwrote its gradient instead; that bug is fixed here.
- Gradients have to be reset to zero before each backward pass, otherwise they add up across training steps.
- An op can be built from smaller ones (`tanh` from `exp`, `/` from `**-1`) and the gradients stay the same, which is a handy correctness check.
- PyTorch does the same thing on tensors instead of scalars; cross-checking against it builds confidence in a hand-written engine.

## Run it

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt   # plus the Graphviz system package, see requirements.txt
jupyter notebook micrograd.ipynb
```

Outputs are not committed: run all cells top to bottom (Kernel → Restart & Run All). The training cell prints a loss that should fall towards zero; exact values vary because the weights are random.

## Credits

All credit for the course, the micrograd design and the teaching to [Andrej Karpathy](https://karpathy.ai). Lecture: <https://www.youtube.com/watch?v=VMj-3S1tku0>. Reference code: <https://github.com/karpathy/micrograd>.
