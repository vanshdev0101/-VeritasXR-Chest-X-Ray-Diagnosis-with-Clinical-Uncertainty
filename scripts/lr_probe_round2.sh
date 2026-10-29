#!/bin/bash
# Round 2: complete the grid. resnet50 was missing its high rate, and
# densenet121 was not probed at all, which left the table ragged.
cd /home/vanshc/git/veritasxr
for spec in "resnet50 1e-3" "densenet121 1e-4" "densenet121 1e-3"; do
  set -- $spec
  ./.venv/bin/python train.py --model "$1" --seed 0 --lr "$2" \
      --skip-existing --log-file results/lr_probe.log
done
echo "=== LR PROBE ROUND 2 COMPLETE ==="
