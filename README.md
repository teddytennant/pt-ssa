JAX implementation of parallel time-aligned spiking self-attention (arXiv:2610.03291).
Install with `pip install -e .` and run the tests with `pytest`.
Cut from the paper: no Triton fused kernel, no CIFAR/ImageNet, no E-SpikeFormer.
The log-threshold surrogate uses M equal to the number of elements in the attention output, including batch axes.
