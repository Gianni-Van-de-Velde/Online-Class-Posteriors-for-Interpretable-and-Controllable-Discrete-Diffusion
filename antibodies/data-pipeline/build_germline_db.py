#!/usr/bin/env python
"""Build the IgBLAST germline BLAST databases used for V-gene-family assignment.

The V-gene labels of our dataset -- and every IgBLAST number reported in the paper -- come
from `igblastp` searching a query sequence against the human IGHV *protein* germline
database built here. Only `human_IGHV_aa` is actually used for labelling; the nucleotide
IGHV/IGHD/IGHJ databases are built too so that a full `igblastn` run also works out of the
box if you want it.

Why the IMGT FASTA files are committed under data/igblast_germline/human_raw/ instead of
downloaded: IMGT revises its germline sets, and a revision silently shifts which family a
borderline sequence is assigned to. Pinning the exact FASTA we used is what makes the
labels reproducible. They were obtained from IMGT/GENE-DB (https://www.imgt.org/genedb/,
Homo sapiens, IGHV/IGHD/IGHJ, F+ORF+in-frame P) and the amino-acid set is IMGT's
"amino acids" export of the same IGHV query. To refresh them against current IMGT, replace
the files in human_raw/ and re-run this script; expect the labels to move slightly.

Requires `makeblastdb` (ships with BLAST+ / IgBLAST) on PATH -- see the README's
"IgBLAST setup" section.

Usage (from the Antibody_experiments/ root):
    python data-pipeline/build_germline_db.py
"""
import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
GERMLINE_DIR = os.path.join(os.path.dirname(HERE), "data", "igblast_germline")
RAW_DIR = os.path.join(GERMLINE_DIR, "human_raw")
OUT_DIR = os.path.join(GERMLINE_DIR, "human")

# (source fasta, output db name, blast dbtype). The protein IGHV db is the one
# oas95_igblast_classify.py points GERMLINE_V at.
DATABASES = [
    ("imgt_human_IGHV_aa.fasta", "human_IGHV_aa", "prot"),
    ("imgt_human_IGHV.fasta", "human_IGHV", "nucl"),
    ("imgt_human_IGHD.fasta", "human_IGHD", "nucl"),
    ("imgt_human_IGHJ.fasta", "human_IGHJ", "nucl"),
]


def main():
    if shutil.which("makeblastdb") is None:
        sys.exit("makeblastdb not found on PATH -- install BLAST+/IgBLAST first "
                 "(see the README's 'IgBLAST setup' section).")

    os.makedirs(OUT_DIR, exist_ok=True)
    for fasta_name, db_name, dbtype in DATABASES:
        src = os.path.join(RAW_DIR, fasta_name)
        if not os.path.exists(src):
            sys.exit(f"missing germline FASTA: {src}")

        # igblast resolves -germline_db_V relative to the db files, so keep the FASTA
        # next to the db it produced. The IMGT export is not valid makeblastdb input as is:
        # cut each '|'-separated IMGT header down to the allele name (e.g. IGHV1-18*01) and
        # drop the IMGT gap dots, as IgBLAST's edit_imgt_file.pl does.
        staged = os.path.join(OUT_DIR, fasta_name)
        with open(src) as f_in, open(staged, "w") as f_out:
            lines = f_in.read().split("\n")
            f_out.write("\n".join(">" + line[1:].split("|")[1] if line.startswith(">")
                                  else line.replace(".", "") for line in lines))

        print(f"makeblastdb: {fasta_name} -> {db_name} ({dbtype})", flush=True)
        subprocess.run(
            ["makeblastdb", "-parse_seqids", "-dbtype", dbtype,
             "-in", staged, "-out", os.path.join(OUT_DIR, db_name)],
            check=True,
        )

    print(f"\nDone. Germline databases in {OUT_DIR}")
    print("oas95_igblast_classify.py uses "
          f"{os.path.join(OUT_DIR, 'human_IGHV_aa')} by default (override with $GERMLINE_V).")


if __name__ == "__main__":
    main()
