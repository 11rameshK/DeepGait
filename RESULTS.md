# Experimental Results

## Target Performance Range

The project is designed with a target Rank-1 identification performance in the **85%–96% range** after full-scale training and evaluation.

These values are **target/expected performance goals**, not measured final results from the current CPU-only development machine.

| Condition | Target Range |
|---|---:|
| Normal walking (NM) | 92%–96% |
| Bag carrying (BG) | 88%–94% |
| Clothing variation (CL) | 85%–90% |

## Verified Local Result

The CPU smoke test verifies that the implemented training pipeline runs end-to-end with valid gradients and checkpoint generation.

- Mapping loss: 0.015856480225920677
- Triplet loss: 0.40004199743270874
- Total loss: 0.41589847207069397
- Gradient absolute sum: 938.8880916355047

Final NM/BG/CL Rank-1 accuracy should be populated only after a full-trained checkpoint is evaluated.
