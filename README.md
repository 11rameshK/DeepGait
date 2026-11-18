# DeepGait Project

## Project
DeepGait: Beyond the Silhouette — Generative Feature Fusion for Robust Gait-Based Identification.

## Primary source
GGCN: Gait Recognition with Generate Network and Convolutional Neural Network (Hao Qin et al., JVCIR 2026).

## Local paths
- Raw CASIA-B silhouettes: C:\Users\RAMESH\Desktop\Projects\Nikhil 9000\CASIA-B
- Processed CASIA-B target: C:\Users\RAMESH\Desktop\Projects\Nikhil 9000\CASIA-B-pkl
- OpenGait: C:\Users\RAMESH\Desktop\Projects\Nikhil 9000\OpenGait-master

## Locked implementation plan
1. Prepare CASIA-B: 124 subjects, NM/BG/CL, 11 views.
2. Convert silhouettes to custom 64x44 pickle sequences, matching the GGCN encoder input used in this project.
3. Verify first-74 train / last-50 test partition.
4. Run a baseline smoke test.
5. Implement GGCN/DeepGait:
   - Generate Network: supervised low-level feature mapping.
   - Encoder: Focal Convolution + pooling.
   - Feature Mapping: PFRE = AFRE + TPFE.
   - Losses: BA+ triplet + L1 mapping losses.
6. Evaluate NM / BG / CL using gallery NM01-04 and probes NM05-06, BG01-02, CL01-02.
7. Add query-gallery inference demo.
8. Package report, results, screenshots and presentation.

## Paper training settings
- CASIA-B train subjects: 001-074.
- Test subjects: 075-124.
- 30 randomly selected frames per gait sequence.
- Per training batch: 4 subjects; per subject 6 NM + 2 BG + 2 CL sequences.
- Adam learning rate: 1e-4.
- Triplet margin: 0.2.
- Mapping loss weights: lambda_g1=0.01, lambda_g2=0.1.
- Paper full run: 160K iterations on 2x NVIDIA 2080Ti.

## Current machine note
This laptop has AMD Radeon 760M and PyTorch is CPU-only. Dataset preparation, code development, inference smoke tests and small-scale checks can run locally. Full paper-scale training should be run on an NVIDIA GPU environment (Kaggle/Colab/local CUDA machine).

## Verified implementation status — 2026-10-29
- CASIA-B raw extraction and source-integrity comparison completed.
- Custom 64x44 preprocessing completed.
- 13,572 processed PKLs validated previously; final re-validation script is included and rerunnable.
- Generate Network, Encoder Network, AFRE, TPFE, feature fusion and Euclidean distance utilities implemented.
- Fixed PyTorch device-transfer bug caused by overriding nn.Module._apply.
- Paired NM/BG/CL dataloader implemented.
- Batch-all triplet loss implemented.
- CPU smoke training completed successfully with non-zero gradients and checkpoint output.
- GPU full-training script implemented with CUDA guard and checkpoint resume/save support.
- CASIA-B evaluation and query-inference scripts implemented; they reject smoke-only checkpoints for accuracy/identification claims.
- Editable Draw.io diagrams and reviewed PNG exports stored in docs/diagrams.
- Full GPU training and final NM/BG/CL accuracy remain pending because this machine has no CUDA GPU.
