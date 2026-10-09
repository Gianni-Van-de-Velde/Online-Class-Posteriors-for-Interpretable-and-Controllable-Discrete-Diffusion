# Antibody experiments

Code for the antibody experiments of
*Online Class Posteriors for Interpretable and Controllable Discrete Diffusion*.

A discrete diffusion model over amino-acid sequences (AMPLIFY-35M backbone, vocabulary of 27
tokens) is trained with either **masking** or **uniform** noise on human antibody heavy
chains, class-conditioned on **IGHV V-gene family** (IGHV1–IGHV7). On top of it we implement:

- the **online class posterior**, tracked during sampling (`sample_with_posterior`),
- the **NOT gate** (dynamic negative guidance driven by that posterior, `sample_NOT`), against
  two baselines built from `sample`: unconditional generation (`guid_scale=0`) and static
  negative guidance (a constant `guid_scale < 0` at every denoising step).

`sample_OR` is also implemented, but the OR gate is reported only in the image experiments.

All of these live in
[`src/discrete_flow_matching/sampling/sampling.py`](src/discrete_flow_matching/sampling/sampling.py).

**Time convention:** `t=0` is pure noise, `t=1` is clean data; sampling integrates from 0 to 1.

Unlike the image experiments, the class labels here are not given: they are produced by running
real germline-gene assignment (IgBLAST) over the corpus, and the *same* IgBLAST call is what
scores generated sequences. So "did the gate work" is measured against an external tool, not
against a learned classifier.

## Setup

```bash
# Python >= 3.10; install torch for your CUDA version first, e.g.
pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu130
pip install -r requirements.txt
pip install -e .      # makes src/sequence_model and src/discrete_flow_matching importable
```

All commands below are run from this folder. W&B logging is offline by default
(`conf/config.yaml`); set `wandb.mode=online` and `wandb.entity=<you>` to sync.

### IgBLAST setup

IgBLAST is not pip-installable and is required for every V-gene label in this repo. It
provides `igblastp` (classification); BLAST+ provides `makeblastdb` (building the germline
database):

```bash
conda install -c bioconda -c conda-forge igblast blast   # or download from NCBI:
# https://ftp.ncbi.nlm.nih.gov/blast/executables/igblast/release/LATEST/
export IGDATA=$CONDA_PREFIX/share/igblast   # must hold internal_data/ and optional_file/
python data-pipeline/build_germline_db.py   # germline database every IgBLAST call searches
```

The last line is also stage 2 of `run/prepare_data.sh`. Run it on its own when you use the
released checkpoints without rebuilding the dataset: every evaluation script needs it.

`igblastp` needs `$IGDATA` to point at the data directory shipped *with IgBLAST*. If yours
does not export it, set it yourself. `scripts/sampling_utils.py` sidesteps this for the
evaluation path by downloading a self-contained NCBI IgBLAST 1.22.0 tarball into `/tmp` on
first use, so sampling and metrics work even where no system IgBLAST exists.

## Dataset

The training set is **OAS95**, the 95%-identity-clustered OAS corpus released with
[IgLM](https://github.com/Graylab/IgLM) (Shuai et al.), filtered to human heavy chains and
labelled by V-gene family with IgBLAST. One command does all of it:

```bash
bash run/prepare_data.sh           # set NUM_WORKERS at the top of the file (default 8)
```

That runs four stages, each resumable — re-run the same command after an interruption and it
continues where it stopped:

| Stage | What happens |
|---|---|
| 1. download | `wget https://data.graylab.jhu.edu/OAS95.tar.gz` (~26GB) into `data/OAS95/` |
| 2. germline | `data-pipeline/build_germline_db.py` → `makeblastdb` over the pinned IMGT human IGHV FASTA |
| 3. classify | keep `species == human`, `chain == Heavy`, full-length V-domain; label each with `igblastp` |
| 4. split | merge shards, drop duplicate sequences, 98/2 train/val + a stratified `val_20k` |

Result: `data/oas95_igblast_human/{full,train,val,val_20k}.csv`, which
`conf/dataset/oas95_igblast_human_train{,_denseval}.yaml` read. ~54.3M training sequences.

Three details worth knowing, because they decide what the labels mean:

- **The full-length filter is doing real work.** Only ~14% of OAS95's human heavy rows are
  full-length V-domains beginning at FR1; the rest are 5′-truncated NGS fragments
  (`GGSLRLSCAASG…` rather than `QVQLQESGPGL…`). Length alone does not separate them. Many
  truncated fragments are still ≥110 residues, so the filter is on the FR1 start motif
  `^[EQD][VIAL][QTK][LVM]`. 
- **The germline database is pinned, not downloaded.** The IMGT FASTAs live in
  `data/igblast_germline/human_raw/`. IMGT revises its germline sets, and a revision silently
  shifts which family a borderline sequence lands in; committing the exact input is what makes
  the labels reproducible. 
- **The classes are very unbalanced**, as the natural repertoire is: IGHV1/3/4 dominate while
  IGHV2/5/6/7 are rare (IGHV6 ≈ 0.0061, IGHV7 ≈ 0.0047). The NOT-gate results are reported
  split by majority/minority group for this reason, and the theoretically motivated prior-weighted init mitigates this.

Stage 3 is by far the most expensive step in the repo. It is parallel and needs
no GPU, so give it as many workers as you have cores and leave it overnight.

## Pretrained checkpoints

The two class-conditional AMPLIFY-35M models used in the paper, under
`logs/<run_name>/checkpoints/checkpoint_<N>/`:

| Run name | Noise | Checkpoint | Train steps |
|---|---|---|---|
| `oas95_igblast_human_35M_uniform` | uniform | `checkpoint_20` | 200k |
| `oas95_igblast_human_35M_masking` | masking | `checkpoint_20` | 200k |

Both were trained with `label_dropout_prob=0.15`, so a single model serves as both the
conditional and the unconditional network for CFG and for the gates, unlike the image
experiments, which train a separate unconditional model per noise type.

These are weights only (no optimizer state), so they can be sampled from but not resumed.

Download `antibody_checkpoints.tar.gz` from [OSF](https://osf.io/veqbg/overview?view_only=a03e63439fd24ad0b27105a694ecc0e8) and extract it here:

```bash
tar xzf antibody_checkpoints.tar.gz   # creates logs/<run_name>/checkpoints/checkpoint_20/model.safetensors
```

## Reproducing the results

Samples are written to `logs/Figures/<run_name>/`, metrics (CSV) to
`logs/Results/<run_name>/`. Every script below runs on one GPU and takes no arguments:

| Experiment | Command |
|---|---|
| Training | `bash run/train.sh` |
| Early-training loss curves | `bash run/train.sh` (with the `denseval` overrides at the bottom of the file) |
| Class posterior vs. IgBLAST | `bash run/posterior.sh` |
| Quality vs. #steps, cond. vs uncond. | `bash run/sample_sweep.sh` |
| NOT gate + both baselines | `bash run/gates.sh` |

Each `run/` script is a single worked example: the settings are plain variables at the top,
each with a one-line note on why it has the value it does, and the alternatives we swept or
considered — per-class sweeps, cheap smoke-test settings, the other noise type — are listed in
a short block at the bottom of the file.

**To switch between the two released models, edit one line.** Each script opens with
`NOISE=uniform` and derives the run name, the checkpoint number and the sampler's
`diffusion_type` from it, so setting `NOISE=masking` moves the whole script over; an
unrecognised value exits rather than composing a half-valid run. Hand-mismatched pair either fails on
a missing path or silently samples one noise type's weights under the other's `diffusion_type`.

`run/train.sh` also forwards extra Hydra overrides, e.g. `bash run/train.sh
trainer.max_steps=1000`.

`run/gates.sh` runs the whole NOT-gate reproduction in sequence; 274 sampling configs, then one scoring pass, under `nohup` or `tmux`. Depending on your compute this can take some time.
Sampling skips any config whose `samples.csv` already exists, so re-running the same command
after an interruption continues where it stopped. **Don't score a partially-sampled grid**:
the results CSVs dedup on hyperparameters, so a partial grid gets scored once and is never
revisited, and the reported group means would then silently average a different set of classes
at each sweep point. Running the file start to finish is safe, since the scoring pass cannot
begin until the four sampling sweeps have returned; the way to get this wrong is to interrupt
it and then invoke the scoring step by hand.

## Layout

```
src/
  sequence_model/               AMPLIFY transformer, data collator, loss, training loop
                                (Accelerate), class-conditioning via AdaLN
  discrete_flow_matching/
    noise_scheduler/            κ(t) schedules, forward noising (masking / uniform)
    sampling/
      sampling_utils.py           transition kernels, denoising step, CFG mixing
      sampling_standard.py        CFG sampler, and the posterior-tracking sampler
      sampling_logical_operators.py   the NOT gate
    metrics/                    FPD, k-NN precision/recall, C2ST, IgLM & ESM2 likelihoods
scripts/                        entry points called by run/*.sh: training, the samplers and
                                the metrics passes, plus their shared helpers
data-pipeline/                  OAS95 -> human heavy chains -> IgBLAST V-gene labels
conf/                           Hydra configs (model, dataset, trainer, sampling)
run/                            one worked example per experiment family, with the
                                alternatives documented at the bottom of each file
```

## Acknowledgements

The model and training infrastructure are adapted from
[AMPLIFY](https://github.com/chandar-lab/AMPLIFY) (Fournier et al.), whose MIT license is
retained in `LICENSE`. The dataset is the OAS95 release distributed with
[IgLM](https://github.com/Graylab/IgLM) (Shuai et al.), which is also used as the naturalness
metric. V-gene assignment uses [IgBLAST](https://ncbi.github.io/igblast/) against germline
sequences from [IMGT](https://www.imgt.org/).
