# lie_detection_eeg
EEG Lie Detection using EEG

This project implements the baseline method from Li et al., Entropy 2025 for EEG-based lie detection. The system uses entropy features and classic machine-learning classifiers to distinguish lie vs truth on the LieWaves dataset.

## Method Summary

### Dataset: 27 subjects, 5 EEG channels (AF3, T7, PZ, T8, AF4), 50 trials per subject

### Features:

 - Fuzzy Entropy (FE)

 - Time-Shifted Multi-Scale Fuzzy Entropy (TSMFE)

 - Hierarchical Multi-Band Fuzzy Entropy (HMFE)

### Classifiers: SVM (Linear/RBF), kNN, Naive Bayes, LDA

### Evaluation:

 - Subject-dependent (LOOCV & 10-fold)

 - Cross-subject (Leave-One-Subject-Out)