# Data-Analysis

This project evaluates different **nnU-Net** training strategies to investigate the impact of **training dataset size**, **data augmentation**, and **cross-validation** on segmentation performance. By comparing models trained under different conditions, the study aims to assess how these factors influence the accuracy, robustness, and generalization capability of nnU-Net models.

# Download link for nnUNet
https://github.com/mic-dkfz/nnunet

# trained nnUNet models

**M1** - Base Model
Training images: 84
,Epochs: 100
,Fold: 0

**M2** - Reduced Dataset
Training images: 42
,Epochs: 100
,Fold: 0

**M3** - Smallest Dataset
Training images: 21
,Epochs: 100
,Fold: 0

**M4** - Five-Fold Cross-Validation
Training images: 84
,Epochs: 100
,Folds: 0–4

**M5** - augmented Dataset
Training images: 168  (application of Albumentations to generate an additional 84 augmented samples)
,Epochs: 100
,Fold: 0

# test_data 
A separate test set consisting of **9 images** was used to evaluate the performance of all models. These images were **not included in any training dataset** and remained completely unseen throughout the training process, ensuring an unbiased evaluation of each model's generalization performance.

# Link to our Google Drive with all trained models

https://drive.google.com/drive/folders/1eyQyHxYgM5Pf9fL6zXKzBZRbYSnwCMo5?usp=sharing


## 📊 Summary

The results indicate that model performance is positively associated with training dataset size, with larger datasets consistently yielding higher segmentation accuracy. In addition, 5-fold cross-validation resulted in a modest improvement over single-fold training, suggesting slightly better model generalization, although the overall effect remained limited.
