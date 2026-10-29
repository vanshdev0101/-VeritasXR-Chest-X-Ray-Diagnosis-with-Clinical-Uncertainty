#!/bin/bash
# Queued behind the ablation sweep: probe whether the single shared learning
# rate handicapped anyone, and in which direction.
#
#   veritasxr @ 1e-4 and 1e-3  -- a from-scratch model may want a higher rate
#   efficientnet_b0 @ 1e-4     -- pretrained backbones are conventionally
#                                 fine-tuned well below 3e-4, so the shared
#                                 rate may have handicapped the baselines
#                                 instead, which would widen the gap rather
#                                 than narrow it
cd /home/vanshc/git/veritasxr
while pgrep -f "train.py --model veritasxr --ablation" >/dev/null; do sleep 60; done
echo "=== ablation sweep finished, starting LR probe ==="
for spec in "veritasxr 1e-4" "veritasxr 1e-3" "efficientnet_b0 1e-4"; do
  set -- $spec
  ./.venv/bin/python train.py --model "$1" --seed 0 --lr "$2" \
      --skip-existing --log-file results/lr_probe.log
done
echo "=== LR PROBE COMPLETE ==="
