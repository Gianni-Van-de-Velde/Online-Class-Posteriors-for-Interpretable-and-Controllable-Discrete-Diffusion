# Synthetic data experiments

Code for the analytically tractable experiments of
*Online Class Posteriors for Interpretable and Controllable Discrete Diffusion*.

The data here is simple enough that everything has a closed form: the class posterior
$p(c \mid x_t)$, the conditional and unconditional denoisers, and the exact target distribution of
each logical gate. Nothing is trained. That makes this the one setting where the paper's claims
can be checked against ground truth rather than against a model's opinion of it.

Sequences are binary, over tokens `A` and `B` drawn in blue and red. Each class is a product of
independent per-position Bernoullis, so a class is nothing but a bias in colour balance, and every
sequence has non-zero likelihood under every class.

| notebook | classes | length | what it shows | paper |
|---|---|---|---|---|
| [`analytic_posterior_2_classes.ipynb`](analytic_posterior_2_classes.ipynb) | 2, each biased over the whole sequence | `L=10` | that sampling with the conditional and unconditional denoisers reproduces the data, and that the closed forms match brute-force enumeration over all $2^L$ sequences; then that the online posterior tracks the exact one during generation under both noise types, and that it is the *evaluator* kernel, not the sampler, that decides whether it is exact | Fig. 1a, Fig. 6 |
| [`analytic_posterior_4_classes.ipynb`](analytic_posterior_4_classes.ipynb) | 4, each biased over one half only | `L=20` | how the posterior arbitrates between overlapping classes; then that the AND, OR and NOT gates sample from their intended target distributions | Fig. 7 |

Two classes biased over the whole sequence are mutually exclusive, which makes AND empty. The
four-class setup exists to get **overlapping** classes: each constrains one half and leaves the
other uniform, so `A1` and `B2` are compatible and AND(`A1`, `B2`) is a real distribution.

**Time convention:** `t=0` is pure noise, `t=1` is clean data; sampling integrates from 0 to 1.

## Setup

```bash
# Python >= 3.10. Only torch, numpy and matplotlib are needed -- no GPU, no dataset, no training.
pip install torch numpy matplotlib
```

Run either notebook top to bottom. The figures are saved under `figures/` as PDFs and are also
stored inline in the committed notebooks, so they can be read without executing anything.

`analytic_posterior_2_classes.ipynb` runs in seconds. `analytic_posterior_4_classes.ipynb` takes
longer: it samples `N_METRIC = 4096` sequences per gate, variant and noise type over 500 denoising
steps. Lower `N_METRIC` for a quick check — the curves get noisier but the picture is unchanged.

## Reading the figures

Two different panels use the same colours and must be read differently. In the **sample grids**,
each row is one complete sequence and the x axis is sequence position. In the **trajectory
panels**, time runs left to right and sequence position runs down, so a horizontal streak is one
position's history. In both, blue = `A`, red = `B`, grey = `MASK`. In the posterior panels,
solid curves are the exact closed-form posterior and dashed ones the online estimate, offset
slightly upwards so the two stay distinguishable where they coincide.

The gate figure is a KL divergence on a log axis, between the generated distribution of half-counts
$(n_1, n_2)$ and the exact target. Grey reference curves are what an exact sampler scores: non-zero
before `t=1` only because $x_t$ is still noisy.
