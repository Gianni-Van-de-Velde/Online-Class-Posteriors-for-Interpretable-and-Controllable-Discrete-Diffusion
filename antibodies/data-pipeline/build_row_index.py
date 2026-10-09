#!/usr/bin/env python
"""Build a byte-offset index over a large row-oriented CSV so oas95_igblast_classify.py
workers can seek() straight to their shard's start row instead of each independently
scanning/discarding every row before it. Without this, pandas' skiprows=N re-parses all N
skipped rows, and many concurrent workers doing that over an 80GB csv contend hard enough
that most of them die silently.

Driven automatically by oas95_igblast_classify_parallel.py; you should not need to run it
by hand.

Assumes no field contains an embedded newline (true for OAS95: sequence + a few scalar
columns, no quoting) so a byte offset can be recorded per physical line.

Records the byte offset of the start of every STRIDE-th row. STRIDE must match
CHUNK_SIZE in oas95_igblast_classify.py: a worker seeks to the nearest indexed row at or
before its start row, then skips at most STRIDE-1 rows to reach the exact one.

Usage: python build_row_index.py <csv_path> <out_index_path> [stride]
"""
import sys

import numpy as np

STRIDE = 20000  # keep in sync with CHUNK_SIZE in oas95_igblast_classify.py


def main():
    csv_path = sys.argv[1]
    out_path = sys.argv[2]
    stride = int(sys.argv[3]) if len(sys.argv) > 3 else STRIDE

    offsets = []
    row = 0
    with open(csv_path, "rb") as f:
        while True:
            pos = f.tell()
            line = f.readline()
            if not line:
                break
            if row % stride == 0:
                offsets.append(pos)
            row += 1
            if row % 20_000_000 == 0:
                print(f"  indexed {row} rows...", flush=True)

    np.save(out_path, np.array(offsets, dtype=np.int64))
    print(f"Wrote {len(offsets)} offsets (stride={stride}) for {row} rows to {out_path}", flush=True)


if __name__ == "__main__":
    main()
