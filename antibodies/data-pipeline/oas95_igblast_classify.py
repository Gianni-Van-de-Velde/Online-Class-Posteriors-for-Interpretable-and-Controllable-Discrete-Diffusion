#!/usr/bin/env python
"""Stream a row range of the OAS95 train/test.csv released with IgLM (no header; columns:
sequence, species, chain, 6 numeric CDR/FR boundary positions), keep human heavy-chain rows that look like full-length V-domains, classify V-gene family
with IgBLAST (igblastp against an IMGT human IGHV protein germline database -- see
data/igblast_germline/), and append (name,sequence,family_str,label,total_length) rows to
the output CSV. Streaming + resumable: processes CHUNK_SIZE rows at a time, appends
immediately, and records progress so a kill/restart resumes instead of reprocessing.

This is the single-worker unit invoked (possibly many times, on disjoint row ranges) by
oas95_igblast_classify_parallel.py -- run this directly only for a small range/local run.

Normally driven by oas95_igblast_classify_parallel.py rather than run by hand.

Env vars (override when igblast lives somewhere unusual, e.g. inside a container):
  IGBLAST_BIN_DIR  -- directory containing igblastp (default: unset, i.e. take it from PATH)
  IGDATA           -- IgBLAST's own data dir, holding internal_data/ and optional_file/
                      (default: unset, i.e. inherited from your environment; a conda
                      install puts it at $CONDA_PREFIX/share/igblast)
  GERMLINE_V       -- path prefix of the germline V blastdb (default: this repo's
                      data/igblast_germline/human/human_IGHV_aa, built by
                      data-pipeline/build_germline_db.py)

Usage: python oas95_igblast_classify.py <oas95_csv_path> <out_prefix> [--start-row N] [--end-row N]
                                         [--row-index PATH]
  Row range is [start-row, end-row) over the raw (unfiltered) csv rows. Omit --end-row to
  run to the end of the file. --row-index points at a build_row_index.py output: lets a
  worker seek() to its start row instead of scanning every row before it (skiprows=N
  otherwise re-parses all N skipped rows, which is what made many concurrent workers die
  silently -- resource contention -- for large N; see build_row_index.py).
"""
import os
import re
import sys
import argparse
import subprocess

import numpy as np
import pandas as pd

# Resolved relative to this file so the script works from any working directory.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# igblastp is not pip-installable; see README ("IgBLAST setup"). Defaults assume it is on PATH
# and that the germline database was built by data-pipeline/build_germline_db.py.
IGBLAST_BIN_DIR = os.environ.get("IGBLAST_BIN_DIR", "")
# igblastp needs IGDATA to point at the IgBLAST data directory shipped with the install
# (the one containing internal_data/ and optional_file/) -- that tree is part of IgBLAST,
# not of this repo. Left unset here so an install that already exports IGDATA just works;
# export it yourself if yours does not. A conda install of igblast puts it at
# $CONDA_PREFIX/share/igblast.
IGDATA = os.environ.get("IGDATA", "")
GERMLINE_V = os.environ.get("GERMLINE_V", os.path.join(REPO_ROOT, "data/igblast_germline/human/human_IGHV_aa"))

CHUNK_SIZE = 20000  # rows read from the OAS95 csv per outer iteration

# Only ~14% of OAS95 human-heavy rows are full-length V-domains starting from FR1;
# the rest are 5'-truncated NGS fragments (e.g. "GGSLRLSCAASG..." instead of
# "QVQLQESGPGL..."). Length alone doesn't reliably separate these (many truncated
# fragments are still >=110 residues) -- filtering on the FR1 start motif is both a
# real quality filter (keeps the dataset consistent with our other heavy-chain sets)
# and a compute saver (skips ~86% of rows before they ever reach IgBLAST).
FR1_START = re.compile(r"^[EQD][VIAL][QTK][LVM]")


def run_igblast(fasta_path, out_path, num_threads=1):
    env = dict(os.environ)
    if IGBLAST_BIN_DIR:
        env["PATH"] = f"{IGBLAST_BIN_DIR}:{env['PATH']}"
    if IGDATA:
        env["IGDATA"] = IGDATA
    subprocess.run(
        [
            "igblastp", "-query", fasta_path, "-germline_db_V", GERMLINE_V,
            "-organism", "human", "-domain_system", "imgt", "-outfmt", "7",
            "-num_alignments_V", "1", "-num_threads", str(num_threads),
            "-out", out_path,
        ],
        check=True, env=env,
    )


def parse_top_families(out_path):
    """Returns {query_id: family_str} for the top V hit of each query."""
    families = {}
    current_query = None
    with open(out_path) as f:
        for line in f:
            if line.startswith("# Query:"):
                current_query = line.split(":", 1)[1].strip()
            elif line.startswith("V\t") and current_query not in families:
                m = re.match(r"V\t\S+\t(\S+)", line)
                if m:
                    fam_m = re.match(r"(IGHV\d+)", m.group(1))
                    if fam_m:
                        families[current_query] = fam_m.group(1)
    return families


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv_path")
    ap.add_argument("out_prefix")
    ap.add_argument("--start-row", type=int, default=0)
    ap.add_argument("--end-row", type=int, default=None)
    ap.add_argument("--num-threads", type=int, default=1)
    ap.add_argument("--row-index", default=None,
                     help="build_row_index.py output for csv_path; enables seek()-based "
                          "start instead of an O(start_row) skiprows scan")
    args = ap.parse_args()

    out_csv = f"{args.out_prefix}_full.csv"
    progress_path = f"{args.out_prefix}_progress.txt"
    tmp_fasta = f"{args.out_prefix}_tmp.fasta"
    tmp_out = f"{args.out_prefix}_tmp.igblast.tsv"

    start_row = args.start_row
    n_done = 0
    if os.path.exists(progress_path):
        with open(progress_path) as f:
            start_row, n_done = map(int, f.read().split())
        print(f"[{args.out_prefix}] Resuming from row {start_row}, {n_done} already processed.", flush=True)

    fam_to_label = {f"IGHV{i}": i - 1 for i in range(1, 8)}
    cols = ["sequence", "species", "chain", "c1", "c2", "c3", "c4", "c5", "c6"]

    if args.row_index:
        row_index = np.load(args.row_index)
        indexed_pos = min(start_row // CHUNK_SIZE, len(row_index) - 1)
        indexed_row = indexed_pos * CHUNK_SIZE
        residual_skip = start_row - indexed_row  # < CHUNK_SIZE regardless of start_row
        csv_file = open(args.csv_path, "rb")
        csv_file.seek(int(row_index[indexed_pos]))
        reader = pd.read_csv(csv_file, header=None, names=cols, chunksize=CHUNK_SIZE,
                              skiprows=residual_skip, usecols=["sequence", "species", "chain"])
    else:
        reader = pd.read_csv(args.csv_path, header=None, names=cols, chunksize=CHUNK_SIZE,
                              skiprows=start_row, usecols=["sequence", "species", "chain"])

    row_cursor = start_row
    for chunk in reader:
        if args.end_row is not None and row_cursor >= args.end_row:
            break
        n_rows_in_chunk = len(chunk)
        if args.end_row is not None and row_cursor + n_rows_in_chunk > args.end_row:
            chunk = chunk.iloc[: args.end_row - row_cursor]

        chunk = chunk.reset_index(drop=True)
        chunk["_local_idx"] = chunk.index
        human_heavy = chunk[(chunk["species"] == "human") & (chunk["chain"] == "Heavy")]
        human_heavy = human_heavy[~human_heavy["sequence"].astype(str).str.contains(r"[BOUXZJ*]", regex=True)]
        human_heavy = human_heavy[human_heavy["sequence"].astype(str).str.match(FR1_START)]

        if len(human_heavy) > 0:
            with open(tmp_fasta, "w") as f:
                for i, seq in enumerate(human_heavy["sequence"]):
                    f.write(f">{i}\n{seq}\n")
            run_igblast(tmp_fasta, tmp_out, num_threads=args.num_threads)
            families = parse_top_families(tmp_out)

            seqs = human_heavy["sequence"].tolist()
            local_idxs = human_heavy["_local_idx"].tolist()
            names, out_seqs, out_fams, out_labels, out_lens = [], [], [], [], []
            for i, (seq, local_idx) in enumerate(zip(seqs, local_idxs)):
                fam = families.get(str(i))
                if fam is None:
                    continue
                names.append(f"oas95_r{row_cursor + local_idx}")
                out_seqs.append(seq)
                out_fams.append(fam)
                out_labels.append(fam_to_label[fam])
                out_lens.append(len(seq))

            if names:
                out_df = pd.DataFrame({
                    "name": names, "sequence": out_seqs, "family_str": out_fams,
                    "label": out_labels, "total_length": out_lens,
                })
                write_header = not os.path.exists(out_csv)
                out_df.to_csv(out_csv, mode="a", header=write_header, index=False)
                n_done += len(out_df)

        row_cursor += n_rows_in_chunk
        with open(progress_path, "w") as f:
            f.write(f"{row_cursor} {n_done}")
        print(f"[{args.out_prefix}] row {row_cursor}: +{len(human_heavy)} human-heavy in chunk, "
              f"{n_done} classified total", flush=True)

    for p in (tmp_fasta, tmp_out):
        if os.path.exists(p):
            os.remove(p)

    print(f"[{args.out_prefix}] Done. {n_done} classified human heavy-chain sequences written to {out_csv}", flush=True)


if __name__ == "__main__":
    main()
