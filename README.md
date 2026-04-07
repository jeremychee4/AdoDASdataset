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

**Validation Set:** https://drive.google.com/drive/folders/1io1O2hOhvM-OpG7mFssuKsVzRK-F5lN7?usp=sharing

**Test Set:** https://drive.google.com/drive/folders/1BGg5sfeCc5yRFoA3YA20QM-imERU4fFU?usp=drive_link

To obtain the **decryption password** for the dataset, please complete the following steps before contacting us:

1. **Download the AdoDAS Dataset License Agreement PDF**: [User License Agreement](https://github.com/jeremychee4/AdoDASdataset/blob/main/license.pdf) (must be signed by a full-time faculty member or researcher; applications not following this rule may be ignored).  
2. **Carefully review the agreement**. It outlines the usage specifications, restrictions, and the responsibilities and obligations of the licensee. Please ensure you fully understand all terms and conditions.  
3. **Manually sign the agreement by hand** after confirming your acceptance of the terms, and fill in all required fields.  
4. **Submit the signed agreement** via email to: `k3nwong@seu.edu.cn`  
   Keep `qitianhua@seu.edu.cn` and `luozhaojie@seu.edu.cn` in CC.

Access is granted **for academic research only** and requires agreement with the dataset license and ethical usage terms.

---

## Baseline Method

This repository provides:
- Baseline implementations based on latent features
- Official task definitions and evaluation metrics
- Scripts for reproducible training and evaluation


---
