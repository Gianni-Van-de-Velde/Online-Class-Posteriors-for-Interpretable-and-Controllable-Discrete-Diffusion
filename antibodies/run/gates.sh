#!/bin/bash
# NOT gate: suppress one V-gene family at generation time, with the guidance scale driven by
# the online class posterior -- plus the two baselines the paper compares it against
# (unconditional generation, and static negative guidance at a constant scale).
#
#   bash run/gates.sh
#
# Samples: logs/Figures/<run>/not_gate[_<tag>]/, metrics: logs/Results/<run>/<tag>/
#
# Runs the whole reproduction in sequence: four sampling sweeps, then one scoring pass. 274
# configs at ~11 min each, so give it days under `nohup` or `tmux`. Sampling skips configs whose
# samples.csv already exists, so re-running the same command resumes where it stopped.
# `set -e` matters here: it stops the scoring pass from running against a partially sampled
# grid, which the README explains gives silently incomparable group means.
set -euo pipefail

# Edit this one line to switch models. Both released runs were trained for 200k steps, so both
# use checkpoint_20.
# Every setting below can also be overridden from the environment, e.g. NOISE=masking bash run/gates.sh
NOISE=${NOISE:-uniform}         # uniform | masking
case $NOISE in
    uniform|masking) ;;
    *) echo "NOISE must be 'uniform' or 'masking', got '$NOISE'" >&2; exit 1 ;;
esac
CKPT=20
RUN_NAME=oas95_igblast_human_35M_${NOISE}

TAG=${TAG:-final}               # isolates this sweep's samples and results; see README on n=256
N_SAMPLES=${N_SAMPLES:-4096}    # n=256 sits on the FPD noise floor (README); 4096 is ~0.005
NUM_STEPS=32                    # quality vs 64 is within noise here, at half the cost
TEMP=${TEMP:-0.01}              # comma-separated list allowed (e.g. 0.003,0.01,0.03). Not 0: at temp=0 the posterior update is a no-op, so the
                                # guidance scale is frozen and the gate is not dynamic at all
OFFSET=0.0                      # not >0: log_post accumulates +offset every step and p_c is
                                # clamped, so any offset drives every initial scale to maximum
                                # guidance within a few steps, erasing the axis being studied
GUID_SCALE_MAX=10.0             # never binds on this grid (largest start scale is IGHV3 at 8.0)

# Which stages to run (all by default) and over which forbidden classes (0 = IGHV1 ... 6 = IGHV7).
# Sampling stages of one run can go to different GPUs in parallel; run `score` once, afterwards.
PARTS=${PARTS:-"unconditional static flat prior score"}
NOT_CLASSES=${NOT_CLASSES:-"0 1 2 3 4 5 6"}
STATIC_SCALES=${STATIC_SCALES:-"[0.03,0.06,0.1,0.2,0.3,0.4,0.5,0.6,0.8,0.9,1.0,1.1]"}
FLAT_SCALES=${FLAT_SCALES:-"[0.1,0.2,0.35,0.5,0.7,0.9,1.2,2.0]"}
PRIOR_SCALES=${PRIOR_SCALES:-"[0.1,0.15,0.2,0.25,0.3,0.4,0.5,0.7,0.85,1.2,1.7,2.0,2.4,2.8,3.2,3.6,4.0,5.0,6.0]"}
has_part() { [[ " $PARTS " == *" $1 "* ]]; }

MODEL_ARGS=(
    'model=[amplify,35M,35M_vgene]'
    trainer=[base,ddlm]
    trainer.train.max_length=256
    trainer.train.diffusion_type=$NOISE
    trainer.train.scheduler_type=linear
)
COMMON_ARGS=(
    sampling.run_name="$RUN_NAME"
    "sampling.to_evaluate_checkpoints=[$CKPT]"
    sampling.label_reference_csv=data/oas95_igblast_human/train.csv
    sampling.max_ref_seqs=20000
    sampling.not_tag="$TAG"
    sampling.not_n_samples="$N_SAMPLES"
    sampling.not_sample_batch_size=256   # the main throughput knob; raise as GPU memory allows
    sampling.num_steps="$NUM_STEPS"
)

# 1. Baseline: unconditional generation. guid_scale=0 reduces to the unconditional model
#    whatever the label is, so one class-independent pool is scored against every forbidden
#    class downstream rather than regenerated seven times.
if has_part unconditional; then
echo "=== unconditional baseline"
python -m scripts.sample_not_gate \
    "${MODEL_ARGS[@]}" "${COMMON_ARGS[@]}" \
    sampling.not_method=unconditional
fi

# 2. Baseline: static negative guidance -- a constant guid_scale < 0 at every denoising step.
#    Magnitudes here; the sampler applies the sign and loops over the forbidden class itself.
if has_part static; then
echo "=== static negative guidance baseline"
python -m scripts.sample_not_gate \
    "${MODEL_ARGS[@]}" "${COMMON_ARGS[@]}" \
    sampling.not_method=static \
    "sampling.not_static_guid_scales=$STATIC_SCALES"
fi

# 3. Ours: dynamic gate, flat initial scale.
has_part flat && for not_class in $NOT_CLASSES; do
    echo "=== dynamic gate, flat init, NOT IGHV$((not_class + 1))"
    python -m scripts.sample_not_gate \
        "${MODEL_ARGS[@]}" "${COMMON_ARGS[@]}" \
        sampling.not_method=dynamic \
        sampling.not_class=$not_class \
        sampling.not_use_prior_weighted_init=false \
        sampling.not_guid_scale_max="$GUID_SCALE_MAX" \
        "sampling.not_temps=[$TEMP]" \
        "sampling.not_offsets=[$OFFSET]" \
        "sampling.not_initial_guid_scales=$FLAT_SCALES"
done

# 4. Ours: dynamic gate, prior-weighted initial scale -- reads the initial scale as a ratio
#    against each class's prior odds, so majority classes start above minority ones by
#    construction. Hence the wider grid and the separate reporting.
has_part prior && for not_class in $NOT_CLASSES; do
    echo "=== dynamic gate, prior-weighted init, NOT IGHV$((not_class + 1))"
    python -m scripts.sample_not_gate \
        "${MODEL_ARGS[@]}" "${COMMON_ARGS[@]}" \
        sampling.not_method=dynamic \
        sampling.not_class=$not_class \
        sampling.not_use_prior_weighted_init=true \
        sampling.not_guid_scale_max="$GUID_SCALE_MAX" \
        "sampling.not_temps=[$TEMP]" \
        "sampling.not_offsets=[$OFFSET]" \
        "sampling.not_initial_guid_scales=$PRIOR_SCALES"
done

# 5. Score everything.
#    --skip_precision_recall: knn_precision_recall_features rebuilds a ~15k x 15k reference
#      manifold per folder and dominates runtime; neither column appears in the NOT-gate tables.
#    --skip_reuse_unconditional: essential with a tag. By default the baseline script also picks
#      up the untagged guid_scale=0 pool from run/sample_sweep.sh, whose dedup key is identical
#      to step 1's, so whichever is scored last would overwrite the other.
METRIC_ARGS=(
    --run_name "$RUN_NAME"
    --to_evaluate_checkpoints "$CKPT"
    --tag "$TAG"
    --label_reference_csv data/oas95_igblast_human/train.csv
    --max_ref_seqs 20000
    --skip_precision_recall
)
if has_part score; then
echo "=== scoring baselines"
python -m scripts.metrics_not_gate "${METRIC_ARGS[@]}" --baseline --skip_reuse_unconditional
echo "=== scoring dynamic gate"
python -m scripts.metrics_not_gate "${METRIC_ARGS[@]}"
fi

# ----------------------------------------------------------------------------------------------
# To score a sweep again from scratch -- after changing a metric, say -- pass --force to both
# scripts in step 5 rather than deleting the CSVs, so rows are replaced in place instead of
# leaving a half-written file behind if it stops early.
#
# To sweep the offset anyway, as extra rows rather than a replacement:
#   'sampling.not_offsets=[0.01,0.05,0.1]'
#
# A single forbidden class rather than all seven: replace the `for` loop with
#   sampling.not_class=0   (0 = IGHV1 ... 6 = IGHV7)
#
# If the sampling sweeps are run concurrently, give each job its own --shard/--num_shards and
# merge with scripts/merge_metric_shards.py: the results CSVs are read-modify-written per
# folder, so two jobs sharing one path silently drop each other's rows.
