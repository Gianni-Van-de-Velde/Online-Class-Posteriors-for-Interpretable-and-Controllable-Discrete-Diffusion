"""
Shared helpers for the sampling and metrics scripts: the sample-folder naming convention they
all agree on, the starting-noise builder, and the two ways a generated sequence gets a class.

The classes are the 7 IGHV V-gene families. At sampling time a generated sequence is
classified the cheap way, by nearest-neighbour (family-mean) framework identity against a
labelled reference pool -- build_classifier() returns that as a `classify_fn` mapping a
decoded sequence string to a class int, or None if unparseable. There is no closed-form
formula for V-gene family, so this is a heuristic.

The ground truth is real IgBLAST, via the helpers at the bottom of this file -- the same
binary and germline database that produced the training labels.

The metric math itself lives in src/discrete_flow_matching/metrics/ (FPD, k-NN
precision/recall, C2ST, IgLM/AR log-likelihood), not here.
"""
import os
import re
import subprocess

import numpy as np
import pandas as pd
import torch

CDR3_PATTERN = re.compile(r'.*C([A-Z]{3,30})WG[A-Z]G')

# Sample-folder naming convention shared by sample_sequences.py's output and every script
# that reads it back. "class" is either an int (one fixed class for the whole folder, the
# per-class-sweep mode) or the literal string "natural" -- per-sample labels drawn from the
# reference pool's own empirical frequency, in which case the class varies per row of that
# folder's samples.csv instead (see compute_empirical_class_probs).
FOLDER_RE = re.compile(
    r"^class_(\d+|natural)_gs_([0-9.]+)_steps_(\d+)_sched_(\w+)_n_(\d+)$"
)


def parse_folder(name):
    m = FOLDER_RE.match(name)
    if not m:
        return None
    cls_str = m.group(1)
    return {
        "class":          cls_str if cls_str == "natural" else int(cls_str),
        "guid_scale":     float(m.group(2)),
        "num_steps":      int(m.group(3)),
        "scheduler_type": m.group(4),
        "n_samples":      int(m.group(5)),
    }

VGENE_FAMILIES = ['IGHV1', 'IGHV2', 'IGHV3', 'IGHV4', 'IGHV5', 'IGHV6', 'IGHV7']



def get_prefix(seq: str):
    """Everything before the CDR3 anchor -- the framework region used for V-gene-family NN classification."""
    m = CDR3_PATTERN.match(seq)
    return seq[: m.span(1)[0]] if m else None


def _ascii_bytes(s):
    """One uint8 per character of `s`. errors="replace" keeps the byte count equal to the
    character count, so positions stay aligned even on a character outside latin-1 (which a
    decoded protein sequence should never contain)."""
    return np.frombuffer(s.encode("latin-1", "replace"), dtype=np.uint8)


def flank_identity(s1, s2):
    if not isinstance(s1, str) or not isinstance(s2, str) or not s1 or not s2:
        return 0.0
    n = min(len(s1), len(s2))
    matches = sum(a == b for a, b in zip(s1[:n], s2[:n]))
    return matches / max(len(s1), len(s2))


def decode_clean(tokenizer, special_strs, token_row):
    raw = tokenizer.decode(token_row)
    return "".join(tok for tok in raw.split(" ") if tok not in special_strs)


def sort_folders_by_setting(folders):
    """Order sample folders as: settings (class, guid_scale, scheduler_type) from low
    guid_scale to high, and within each setting, increasing num_steps -- used by the
    metrics scripts so both the per-folder progress log and the final results table read
    as a clean sweep rather than plain alphabetical order (which puts "steps_128" before
    "steps_16" as strings)."""
    def sort_key(folder):
        c = parse_folder(folder)
        # str(c["class"]): "class" is an int for a fixed-class folder or the string "natural"
        # for the natural-frequency mode -- comparing int and str directly would raise
        # TypeError if both kinds of folder are ever present together.
        return (c["guid_scale"], str(c["class"]), c["scheduler_type"], c["num_steps"])
    return sorted(folders, key=sort_key)


def append_result_row(csv_path, row, dedup_keys, str_dedup_keys=()):
    """Adds one scored row to a results CSV, replacing any existing row with the same
    `dedup_keys`.

    Called after every folder, not once at the end, so a multi-hour sweep survives an
    interruption. That makes it a read-modify-write of the whole file per row, so concurrent
    jobs must each write their own shard (the metrics scripts' --shard) -- two jobs sharing
    one path silently drop each other's rows.

    INVARIANT: every dedup key must round-trip through CSV as ONE Python type. CSV carries no
    types, so read_csv re-infers each column: a column mixing ints and strings comes back as
    str, one holding only ints as int64, and the same setting then compares unequal to itself
    depending on what else is in the file -- dedup appends a duplicate instead of replacing.
    Name such a column in `str_dedup_keys` to have it normalised to str on both sides. The
    real case is `class`, an int per class or the literal "natural".
    """
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    df = pd.DataFrame([row])
    if os.path.exists(csv_path):
        df = pd.concat([pd.read_csv(csv_path), df], ignore_index=True)
        # Normalise after the concat so the existing rows and the new one are coerced together,
        # and persist that form: writing it back str-typed is what stops the next read from
        # re-inferring something else.
        for key in str_dedup_keys:
            if key in df.columns:
                df[key] = df[key].astype(str)
        df = df.drop_duplicates(subset=dedup_keys, keep="last")
    df.to_csv(csv_path, index=False)


def compute_empirical_class_probs(reference_df, num_classes, label_col="label"):
    """Empirical class distribution of the (possibly subsampled) reference pool, indexed by
    class 0..num_classes-1. Drives sample_sequences.py's natural-frequency mode. Computed from
    the data rather than hand-maintained, so it cannot drift out of sync with the labels."""
    counts = reference_df[label_col].value_counts()
    probs = np.array([counts.get(c, 0) for c in range(num_classes)], dtype=float)
    total = probs.sum()
    if total == 0:
        raise ValueError(
            f"No rows for any of the {num_classes} classes found in the reference pool "
            f"({len(reference_df)} rows) -- can't compute an empirical distribution."
        )
    return probs / total


def load_checkpoint(run_name, checkpoint_number, model_config, device):
    """One checkpoint's weights, ready to sample from. checkpoint_number=0 returns a
    randomly initialised model instead -- the floor baseline."""
    # Imported here, not at module scope, so this module stays importable without the model
    # stack installed -- the metrics scripts only need its folder conventions and classifier.
    import safetensors.torch
    from sequence_model.model.amplify_model import AMPLIFY

    model = AMPLIFY(model_config)
    if checkpoint_number == 0:
        print(f"  [{run_name}] Random init baseline.")
        return model.to(device).eval()
    ckpt_path = f"logs/{run_name}/checkpoints/checkpoint_{checkpoint_number}/model.safetensors"
    print(f"  [{run_name}] Loading: {ckpt_path}")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(ckpt_path)
    state_dict = safetensors.torch.load_file(ckpt_path, device="cpu")
    if "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    model.load_state_dict(state_dict)
    return model.to(device).eval()


def build_x_in(df, tokenizer, max_length, n, seed, diffusion_type, scheduler_type, polynomial_power, device,
                label_col="label", target_class=None):
    """Noise `n` real reference sequences (optionally filtered to `target_class`) down to t=0,
    used as x_in so the generative trajectory starts from the right sequence LENGTH for that
    class -- <bos>/<eos>/<pad> positions are protected from resampling for the whole trajectory
    (see sampling_utils.denoising_step), so the starting length has to already match what
    the class needs."""
    from discrete_flow_matching.noise_scheduler.noise_scheduler import noising

    pool = df[df[label_col] == target_class] if target_class is not None else df
    ref_seqs = pool["sequence"].sample(n=n, random_state=seed, replace=True).tolist()
    encoded = [tokenizer.encode(s, max_length, random_truncate=False) for s in ref_seqs]
    ref_tokens = torch.full((n, max_length), tokenizer.pad_token_id, dtype=torch.long)
    for i, e in enumerate(encoded):
        ref_tokens[i, : e.shape[0]] = e
    ref_tokens = ref_tokens.to(device)
    t_start = torch.zeros(n, device=device)
    x_in, _ = noising(ref_tokens, t_start, tokenizer=tokenizer, diffusion_type=diffusion_type,
                       scheduler_type=scheduler_type, polynomial_power=polynomial_power)
    return x_in


class VGeneFamilyClassifier:
    num_classes = len(VGENE_FAMILIES)
    class_names = VGENE_FAMILIES

    def __init__(self, reference_df):
        reference_df = reference_df.copy()
        reference_df["prefix"] = reference_df["sequence"].apply(get_prefix)
        self.prefix_by_label = {
            lab: reference_df[reference_df["label"] == lab]["prefix"].dropna().tolist()
            for lab in range(self.num_classes)
        }
        # Packed form of the same pool, built once so __call__ is a numpy reduction rather
        # than one interpreted flank_identity call per reference prefix -- the difference
        # between minutes and hours per sampling config.
        #
        # Per label: a uint8 matrix of the non-empty prefixes zero-padded to the longest, their
        # true lengths, their positions in the original list, and that list's full length. Zero
        # is safe as padding because it never equals a byte of a real prefix.
        self._packed = {}
        for lab, prefixes in self.prefix_by_label.items():
            keep = [(i, p) for i, p in enumerate(prefixes) if isinstance(p, str) and p]
            width = max((len(p) for _, p in keep), default=0)
            mat = np.zeros((len(keep), width), dtype=np.uint8)
            for row, (_, p) in enumerate(keep):
                mat[row, : len(p)] = _ascii_bytes(p)
            self._packed[lab] = (
                mat,
                np.array([len(p) for _, p in keep], dtype=np.int64),
                np.array([i for i, _ in keep], dtype=np.int64),
                len(prefixes),
            )

    def _mean_identity(self, prefix, lab):
        """Mean flank_identity of `prefix` against every reference prefix of class `lab`."""
        mat, ref_lens, positions, n_total = self._packed[lab]
        if n_total == 0:
            return 0.0  # matches the original `if self.prefix_by_label[lab] else 0.0`
        # Scatter back into a full-length vector so the mean covers every reference prefix,
        # with the empty ones contributing the 0.0 that flank_identity would give them.
        per_ref = np.zeros(n_total, dtype=np.float64)
        if mat.shape[0] and prefix:
            q = _ascii_bytes(prefix)
            # Comparing only the first min(len(q), len(p)) bytes falls out of the padding:
            # past a short reference's length the pad byte matches nothing.
            w = min(q.shape[0], mat.shape[1])
            matches = (mat[:, :w] == q[:w]).sum(axis=1)
            per_ref[positions] = matches / np.maximum(ref_lens, q.shape[0])
        return float(per_ref.mean())

    def __call__(self, seq: str):
        prefix = get_prefix(seq)
        if prefix is None:
            return None
        scores = [self._mean_identity(prefix, lab) for lab in range(self.num_classes)]
        return int(np.argmax(scores))


def _read_csv_random_subset(csv_path, n, seed):
    """A uniform-random sample of n data rows, without materialising the full file: count the
    lines, pick n indices to keep, and hand pandas the complementary skiprows. Uniform rather
    than head(n) so the sample inherits no bias from file order (e.g. study batching)."""
    with open(csv_path) as f:
        n_rows = sum(1 for _ in f) - 1  # exclude header
    if n >= n_rows:
        return pd.read_csv(csv_path)
    rng = np.random.default_rng(seed)
    keep = rng.choice(n_rows, size=n, replace=False)
    skip = np.setdiff1d(np.arange(n_rows), keep) + 1  # +1: row 0 is the header
    return pd.read_csv(csv_path, skiprows=skip)


def slugify(value):
    """Filesystem-safe fragment of `value`: alphanumerics and dots kept, the rest collapsed
    to '_'. Used to build cache directory names out of model names and CLI tokens."""
    return re.sub(r"[^A-Za-z0-9.]+", "_", str(value)).strip("_")


def reference_pool_tag(reference_csv, max_ref_seqs, seed):
    """Identifies the exact reference pool a cached tensor was computed from.

    Everything the metrics scripts cache under data/ (reference embeddings, reference IgLM
    log-likelihoods) depends on WHICH reference sequences were loaded, i.e. on the csv,
    max_ref_seqs and seed together. A cache keyed on the class alone would silently score
    FPD / C2ST against a pool it never loaded -- plausible numbers, no error.

    It keys on the csv's NAME, not its contents, so rebuilding the dataset in place under the
    same filename leaves the caches stale: pair a fresh run/prepare_data.sh with deleting
    data/*_cache/.
    """
    stem = os.path.splitext(os.path.basename(str(reference_csv)))[0]
    size = "full" if max_ref_seqs is None else f"n{int(max_ref_seqs)}"
    return f"{slugify(stem)}_{size}_s{int(seed)}"


def model_tag(model_name):
    """Short identifier for an embedding/LM checkpoint, e.g.
    'facebook/esm2_t33_650M_UR50D' -> 'esm2_t33_650M_UR50D'. Part of the cache path because the
    cached tensors are that model's own outputs."""
    return slugify(os.path.basename(str(model_name).rstrip("/")))


def build_classifier(reference_csv: str, max_ref_seqs: int = None, seed: int = 0):
    """Returns (classify_fn, num_classes, class_names, reference_df). classify_fn maps a
    decoded sequence string -> predicted class int, or None if it couldn't be parsed.

    max_ref_seqs, if given, loads a uniform-random sample of N rows from reference_csv
    (see _read_csv_random_subset) instead of the full multi-million-row file."""
    if max_ref_seqs is None:
        reference_df = pd.read_csv(reference_csv)
    else:
        reference_df = _read_csv_random_subset(reference_csv, max_ref_seqs, seed)
    clf = VGeneFamilyClassifier(reference_df)
    return clf, clf.num_classes, clf.class_names, reference_df


# ── Ground-truth V-gene-family labelling via the real igblastp binary and IMGT germline db,
# as opposed to the flank-identity heuristic above. Same tool, database and family parsing as
# produced the training labels (data-pipeline/oas95_igblast_classify.py), so what a generated
# sequence scores as and what a training sequence was labelled as cannot drift apart.
# Used by metrics_sequences.py and posterior_classifier_agreement.py. ──

IGBLAST_VERSION = "1.22.0"
IGBLAST_WEAK_HIT_PIDENT_THRESHOLD = 80.0    # top V hit %identity below this => "weak hit"
IGBLAST_WEAK_HIT_COVERAGE_THRESHOLD = 0.8   # top V hit alignment_length / len(sequence) below this => "weak hit"


def ensure_igblast_installed(igblast_dir=f"/tmp/ncbi-igblast-{IGBLAST_VERSION}"):
    """Downloads the self-contained NCBI igblastp binary tarball if not already present
    (idempotent -- cheap no-op on every call after the first). Returns (bin_dir, igdata)."""
    if not os.path.isdir(igblast_dir):
        print(f"Downloading igblastp {IGBLAST_VERSION} (self-contained NCBI binary tarball)...")
        subprocess.run(["curl", "-s", "-o", "/tmp/igblast.tar.gz",
                         f"https://ftp.ncbi.nlm.nih.gov/blast/executables/igblast/release/LATEST/"
                         f"ncbi-igblast-{IGBLAST_VERSION}-x64-linux.tar.gz"], check=True)
        subprocess.run(["tar", "-C", "/tmp", "-xzf", "/tmp/igblast.tar.gz"], check=True)
    return f"{igblast_dir}/bin", igblast_dir


def run_igblast(fasta_path, out_path, germline_v, bin_dir, igdata, num_threads=4):
    env = dict(os.environ, IGDATA=igdata, PATH=f"{bin_dir}:{os.environ['PATH']}")
    subprocess.run(
        ["igblastp", "-query", fasta_path, "-germline_db_V", germline_v,
         "-organism", "human", "-domain_system", "imgt", "-outfmt", "7",
         "-num_alignments_V", "1", "-num_threads", str(num_threads),
         "-out", out_path],
        check=True, env=env,
    )


def parse_igblast_hits(out_path):
    """Returns {query_id: {'family', 'pident', 'align_length'}} for the top V hit of each
    query -- a query with 0 hits (e.g. a non-antibody-like sequence) simply never gets an
    entry."""
    hits = {}
    current_query = None
    with open(out_path) as f:
        for line in f:
            if line.startswith("# Query:"):
                current_query = line.split(":", 1)[1].strip()
            elif line.startswith("V\t") and current_query not in hits:
                fields = line.rstrip("\n").split("\t")
                fam_m = re.match(r"(IGHV\d+)", fields[2])
                if fam_m:
                    hits[current_query] = {
                        "family": fam_m.group(1), "pident": float(fields[3]), "align_length": int(fields[4]),
                    }
    return hits


def compute_igblast_metrics(seqs, ids, germline_v, requested_families=None, tmp_prefix="/tmp/igblast_metrics",
                             num_threads=4):
    """Real-igblastp classification + confidence for a batch of decoded sequences.

    requested_families, if given (one family string per seq, e.g. "IGHV3", or None entries),
    additionally reports igblast_match_rate: fraction of sequences WITH a hit whose top-hit
    family equals the requested one. hit_rate is reported separately so "wrong family" and
    "not antibody-like enough to get a hit at all" don't get conflated into one number.

    Returns a dict of aggregate stats plus the per-sequence DataFrame under "per_sequence".
    """
    bin_dir, igdata = ensure_igblast_installed()
    fasta_path, out_path = f"{tmp_prefix}.fasta", f"{tmp_prefix}.igblast.tsv"
    with open(fasta_path, "w") as f:
        for qid, seq in zip(ids, seqs):
            f.write(f">{qid}\n{seq}\n")
    run_igblast(fasta_path, out_path, germline_v, bin_dir, igdata, num_threads=num_threads)
    hits = parse_igblast_hits(out_path)

    rows = []
    for i, (qid, seq) in enumerate(zip(ids, seqs)):
        hit = hits.get(qid)
        fam = hit["family"] if hit else None
        pident = hit["pident"] if hit else None
        coverage = (hit["align_length"] / len(seq)) if (hit and len(seq) > 0) else None
        req_fam = requested_families[i] if requested_families is not None else None
        rows.append({
            "id": qid, "igblast_family": fam, "igblast_pident": pident, "igblast_coverage": coverage,
            "no_hit": hit is None,
            "weak_hit": (hit is not None) and (
                pident < IGBLAST_WEAK_HIT_PIDENT_THRESHOLD or coverage < IGBLAST_WEAK_HIT_COVERAGE_THRESHOLD),
            "requested_family": req_fam,
            "match": (fam is not None and req_fam is not None and fam == req_fam),
        })
    per_seq = pd.DataFrame(rows)

    hit_rate = float(1.0 - per_seq["no_hit"].mean())
    weak_hit_rate = float(per_seq["weak_hit"].mean())
    hits_df = per_seq[~per_seq["no_hit"]]
    mean_pident = float(hits_df["igblast_pident"].mean()) if len(hits_df) else float("nan")
    mean_coverage = float(hits_df["igblast_coverage"].mean()) if len(hits_df) else float("nan")

    match_rate = float("nan")
    if requested_families is not None:
        scored = hits_df[hits_df["requested_family"].notna()]
        match_rate = float(scored["match"].mean()) if len(scored) else float("nan")

    return {
        "igblast_hit_rate": hit_rate,
        "igblast_weak_hit_rate": weak_hit_rate,
        "igblast_mean_pident": mean_pident,
        "igblast_mean_coverage": mean_coverage,
        "igblast_match_rate": match_rate,
        "per_sequence": per_seq,
    }
