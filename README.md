# AdoDAS Dataset

**A Large-Scale, Privacy-Preserving Multimodal Dataset for Depression, Anxiety, and Stress Assessment in Adolescents**

---

## Overview

**AdoDAS** is a large-scale multimodal dataset designed for automated assessment of **depression, anxiety, and stress (D/A/S)** in adolescents.  
The dataset is introduced in our **ACMMM Dataset Track** paper and emphasizes **ethical data sharing** and **privacy protection for minors**.

To ensure digital safety, **no raw audio or video recordings are released**.  
Instead, AdoDAS provides **anonymized latent representations and temporal metadata**, enabling reproducible multimodal research without exposing identifiable information.

---

## Dataset Summary

- **Participants**: 6,000 adolescents  
- **Segments**: 24,000 audio-video segments  
- **Modalities**:
  - Visual
  - Acoustic
  - Temporal metadata: VAD alignments
- **Annotations**: DASS-21 (Depression / Anxiety / Stress)
- **Tasks**:
  - Multi-task binary D/A/S screening
  - DASS-21 item-level prediction
- **Data Splits**: Official subject-disjoint train / validation / test splits (70%:10%:20%)

---

## Data Access

🔗 **Dataset access link and instructions:**  
**Train Set:** https://drive.google.com/drive/folders/1OyjFeAlYFDYKcFtL_232rDSHTeEpVGQe?usp=sharing
**Validation Set** https://drive.google.com/drive/folders/1io1O2hOhvM-OpG7mFssuKsVzRK-F5lN7?usp=sharing

Access is granted **for academic research only** and requires agreement with the dataset license and ethical usage terms.

---

## Baseline Method

This repository provides:
- Baseline implementations based on latent features
- Official task definitions and evaluation metrics
- Scripts for reproducible training and evaluation


---
