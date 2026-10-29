#!/bin/bash
# Does the single shared learning rate handicap anyone, and in which direction?
#
# The question cuts both ways and the second direction matters more:
#   veritasxr @ 1e-4, 1e-3        a from-scratch model may want a different
#                                 rate than the shared 3e-4
#   efficientnet_b0 @ 1e-4, 1e-3  pretrained backbones are conventionally
#   resnet50 @ 1e-4               fine-tuned well below 3e-4, so the shared
#                                 rate may have handicapped the baselines --
#                                 which would widen the measured gap to
#                                 VeritasXR, not narrow it
cd /home/vanshc/git/veritasxr
for spec in "veritasxr 1e-4" "veritasxr 1e-3" \
            "efficientnet_b0 1e-4" "efficientnet_b0 1e-3" \
            "resnet50 1e-4"; do
  set -- $spec
  ./.venv/bin/python train.py --model "$1" --seed 0 --lr "$2" \
      --skip-existing --log-file results/lr_probe.log
done
echo "=== LR PROBE COMPLETE ==="
