#!/bin/bash
# Build the training dataset: OAS95 -> human heavy chains -> IgBLAST V-gene-family labels.
#
#   bash run/prepare_data.sh
#
# Produces data/oas95_igblast_human/{full,train,val,val_20k}.csv, which
# conf/dataset/oas95_igblast_human_train{,_denseval}.yaml read.
#
# Needs `igblastp` and `makeblastdb` on PATH (see the README's "IgBLAST setup") and ~95GB of
# free disk. No GPU. Every stage is resumable: re-run the same command to continue.
set -euo pipefail

NUM_WORKERS=8                       # parallel IgBLAST workers; set to your core count
OAS95_DIR=data/OAS95
OUT_DIR=data/oas95_igblast_human

# 1. Download OAS95 -- the 95%-identity-clustered OAS corpus released with IgLM (Shuai et al.),
#    ~26GB. Columns, no header: sequence, species, chain, 6 CDR/FR boundary positions.
mkdir -p "$OAS95_DIR"
if [[ ! -f $OAS95_DIR/train.csv ]]; then
  wget -c https://data.graylab.jhu.edu/OAS95.tar.gz -O "$OAS95_DIR/OAS95.tar.gz"
  tar -C "$OAS95_DIR" --strip-components=1 -xzf "$OAS95_DIR/OAS95.tar.gz"
fi

# 2. Build the IMGT human IGHV germline databases that IgBLAST searches against.
python data-pipeline/build_germline_db.py

# 3. Classify in parallel shards, then merge into train/val splits. This is the most expensive
#    step in the repo (hours to days) and is embarrassingly parallel. total_rows is passed
#    explicitly so the orchestrator can cut disjoint row ranges without walking the 80GB file
#    first; counting takes a few minutes, so it is cached.
if [[ ! -f $OAS95_DIR/train.rows ]]; then
  wc -l < "$OAS95_DIR/train.csv" | tr -d ' ' > "$OAS95_DIR/train.rows"
fi

python data-pipeline/oas95_igblast_classify_parallel.py \
    "$OAS95_DIR/train.csv" \
    "$OUT_DIR" \
    "$(cat "$OAS95_DIR/train.rows")" \
    "$NUM_WORKERS"

echo "Done. Shards under $OUT_DIR/shards/ are only needed for resume -- delete when happy."

# ----------------------------------------------------------------------------------------------
# Only ~14% of OAS95's human heavy rows survive the filters, so the output is far smaller than
# the input: ~54.3M training sequences. The filters are species == "human", chain == "Heavy", no
# ambiguous residues (BOUXZJ*), and a full-length-V-domain FR1 start motif ^[EQD][VIAL][QTK][LVM]
# -- see data-pipeline/oas95_igblast_classify.py for why length alone will not do.
#
# A quick smoke test on the first million rows, to check IgBLAST is wired up before committing:
#   python data-pipeline/oas95_igblast_classify.py data/OAS95/train.csv /tmp/oas95_smoke \
#       --start-row 0 --end-row 1000000
#
# A held-out set from OAS95's own test.csv instead of the train split (then add a
# conf/dataset/*.yaml pointing at it, copying oas95_igblast_human_train.yaml):
#   python data-pipeline/oas95_igblast_classify_parallel.py \
#       data/OAS95/test.csv data/oas95_igblast_human_test \
#       "$(wc -l < data/OAS95/test.csv)" 8
#
# A different train/val split, or dense-eval subsample size:
#   ... "$NUM_WORKERS" --val-frac 0.05 --val-subsample 50000
#
# If igblastp cannot find its internal data, export IGDATA (see the README). GERMLINE_V
# overrides the germline database location, IGBLAST_BIN_DIR the binary.
