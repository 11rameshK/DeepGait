# DeepGait — System Design and Methodology

## Project
DeepGait: Generative Feature Fusion for Robust Gait-Based Identification

## System Architecture
CASIA-B silhouettes -> integrity verification -> 64x44 preprocessing -> PKL repository -> DeepGait/GGCN model -> feature embedding -> Euclidean gallery matching -> predicted subject ID.

The DeepGait model contains:
1. Generate Network — separate NM/BG/CL low-level feature branches.
2. Encoder Network — focal convolution and pooling.
3. Feature Mapping Network — AFRE + TPFE.
4. Final identity embedding — 256 x 63.

## Dataset
Primary benchmark: CASIA-B.
- 124 subjects
- 11 views
- Conditions: NM, BG, CL
- Train subjects: 001-074
- Test subjects: 075-124
- Training frame target: 30 frames per sequence

## Preprocessing
The project uses a custom preprocessing script rather than OpenGait's default 64x64 output.
- Detect silhouette foreground.
- Crop vertically around the silhouette.
- Resize to height 64 while preserving aspect ratio.
- Center by silhouette mass.
- Crop/pad to exact 64x44.
- Serialize valid sequences as PKL arrays.
- Skip source groups with fewer than 5 usable frames.

## Training Methodology
Training data is formed from paired NM/BG/CL sequences for each subject.
The implementation uses:
- L1 mapping loss for BG->NM and CL->NM low-level features.
- Batch-all triplet loss on final embeddings.
- Adam optimizer, learning rate 1e-4.
- Triplet margin 0.2.
- Mapping weights: 0.01 for BG and 0.1 for CL.

Local CPU smoke training is only for code verification. Full paper-scale training requires CUDA/NVIDIA hardware.

## Evaluation Methodology
Gallery:
- NM01, NM02, NM03, NM04

Probes:
- NM05, NM06
- BG01, BG02
- CL01, CL02

Embeddings are compared with Euclidean distance. Rank-1 identification accuracy is reported separately for NM, BG, and CL only after full training.

## Storage Design
The current implementation is file-based:
- Raw silhouette image folders
- Processed PKL files
- Model checkpoint files
- JSON/log/result files
- Diagrams and presentation assets

No relational database is used. Therefore a classical ER diagram would invent non-existent tables/entities. The project includes a Data Repository Structure diagram instead.

## Current Verified Implementation
- CASIA-B extraction and repair: complete
- 64x44 preprocessing: complete
- PKL validation: complete
- DeepGait model forward shape test: pass
- Dataloader: implemented
- Batch-all triplet loss: implemented
- CPU smoke forward/backward/optimizer/checkpoint: pass
- GPU full-training script: implemented, not executed locally because CUDA is unavailable
- Evaluation and query-inference scripts: implemented, require a full-training checkpoint for valid results
