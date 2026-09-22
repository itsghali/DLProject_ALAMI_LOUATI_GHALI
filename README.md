# Deep Learning for Medical Data

### Comparative Study of MLP, CNN, RNN, LSTM, GRU and Seq2Seq Architectures

A comparative Deep Learning project implemented with **PyTorch**, studying how neural network architectures should be adapted to the underlying structure of different types of medical data:

* **Tabular clinical data** → MLP
* **Medical images** → CNN
* **Sequential ECG signals** → RNN, LSTM, GRU and Seq2Seq

The project was developed as part of the **Deep Learning module at EMSI** during the 2025–2026 academic year.

---

## 📌 Project Overview

Deep Learning models do not perform independently of the structure of the data they process. Different data modalities contain different forms of structure:

| Data modality | Dataset                  | Main architecture          | Structure exploited   |
| ------------- | ------------------------ | -------------------------- | --------------------- |
| Tabular       | Breast Cancer MSK 2018   | MLP                        | Feature relationships |
| Images        | RSNA Pneumonia Detection | CNN                        | Spatial locality      |
| Sequences     | PTB-XL ECG               | RNN / LSTM / GRU / Seq2Seq | Temporal dependencies |

The central question of the project is:

> **How should Deep Learning architectures be adapted to the geometry and temporal structure of different types of data?**

The study investigates this question experimentally through three independent but complementary experiments.

---

# 🎯 Objectives

The project has five main objectives:

1. Develop practical expertise with **PyTorch**, including:

   * `nn.Module`
   * `nn.Sequential`
   * `DataLoader`
   * GPU/CPU device management
   * model initialization
   * `state_dict`
   * model saving and loading

2. Implement and compare two **MLP architectures** for clinical tabular classification.

3. Design a **CNN inspired by LeNet-5** for pneumonia detection from chest X-rays and investigate convolutional hyperparameters.

4. Implement and compare:

   * Vanilla RNN
   * LSTM
   * GRU
   * Seq2Seq autoencoder

   for multi-label ECG classification.

5. Perform a cross-architecture analysis connecting:

   * theoretical foundations,
   * architectural choices,
   * experimental results,
   * computational cost,
   * and data structure.

---

# 🧠 Part I — MLP for Clinical Tabular Data

## Dataset

The first experiment uses the **Breast Cancer MSK 2018** dataset from Memorial Sloan Kettering Cancer Center.

The original dataset contains:

* **1,918 patients**
* **59 variables**
* clinical, genomic and pathological information

The target variable is:

**Metastatic Disease at Last Follow-up**

The task is therefore a **binary classification problem**.

To increase the training sample size, the project uses `GaussianCopulaSynthesizer` from SDV to generate **10,000 synthetic samples** based on the original dataset.

After cleaning and feature engineering, the final representation contains **45 features**.

---

## Data preprocessing

The preprocessing pipeline was designed to avoid data leakage:

1. Dataset cleaning
2. Feature engineering
3. Train / validation / test split
4. Fit encoders and scalers on training data only
5. Apply transformations to validation and test sets

The dataset is split using:

```text
70% Train
15% Validation
15% Test
```

Resulting in:

```text
7,000 training samples
1,500 validation samples
1,500 test samples
```

The preprocessing includes:

* removal of features with excessive missing values
* removal of identifiers
* removal of redundant features
* median imputation for numerical variables
* mode imputation for categorical variables
* Target Encoding for categorical features
* RobustScaler
* log transformation for highly skewed features
* QuantileTransformer for selected duration-related variables

---

## MLP architectures

Two implementations of the same architecture were compared:

### `nn.Sequential`

A compact sequential implementation.

### Custom `nn.Module`

A manually defined architecture using `__init__()` and `forward()`.

Both implement:

```text
45
 ↓
128
 ↓
64
 ↓
32
 ↓
1
```

with:

* Batch Normalization
* ReLU
* Dropout
* BCEWithLogitsLoss

Total parameters:

```text
16,705
```

---

## Weight initialization experiment

Four initialization strategies were evaluated:

* Gaussian
* Constant
* Xavier
* He / Kaiming

The Gaussian initialization produced the best validation AUC in the experiment:

```text
Validation AUC ≈ 0.7509
```

---

## Final MLP results

| Model          |   Accuracy | Precision |     Recall |         F1 |    AUC-ROC |
| -------------- | ---------: | --------: | ---------: | ---------: | ---------: |
| Custom MLP     |     0.7087 |    0.8488 |     0.7320 |     0.7861 |     0.7486 |
| Sequential MLP |     0.7087 |    0.8618 |     0.7165 |     0.7825 | **0.7577** |
| XGBoost        | **0.7287** |    0.8249 | **0.7985** | **0.8115** |     0.7437 |

### Main observation

The two PyTorch implementations produce very similar results, demonstrating that `nn.Sequential` and a custom `nn.Module` are primarily different ways of expressing the architecture when the underlying network is identical.

The experiment also shows that traditional machine-learning methods such as XGBoost remain highly competitive on relatively small heterogeneous tabular datasets.

---

# 🩻 Part II — CNN for Chest X-Ray Classification

## Dataset

The second experiment uses the:

**RSNA Pneumonia Detection Challenge**

The dataset contains chest radiographs in **DICOM** format.

The study identifies:

* **26,684 usable patients**
* 20,672 normal cases
* 6,012 pneumonia cases

For the experiments, a balanced subset of:

```text
6,000 images
├── 3,000 Normal
└── 3,000 Pneumonia
```

was used.

The classification task is:

```text
Normal vs Pneumonia
```

---

# 🧩 CNN Architecture

The implemented network is inspired by **LeNet-5**, but modernized with:

* ReLU
* BatchNorm
* MaxPooling
* AdaptiveAvgPool
* Dropout

Architecture:

```text
Input: 112 × 112

        ↓
Conv2D 1 → 32
BatchNorm
ReLU
MaxPool

        ↓
Conv2D 32 → 64
BatchNorm
ReLU
MaxPool

        ↓
Conv2D 64 → 128
BatchNorm
ReLU
MaxPool

        ↓
AdaptiveAvgPool 8 × 8

        ↓
Flatten

        ↓
Linear 8192 → 512
ReLU
Dropout

        ↓
Linear 512 → 256
ReLU
Dropout

        ↓
Linear 256 → 2
```

Total parameters:

```text
4,419,778
```

---

## Image preprocessing

The preprocessing pipeline includes:

* Resize to `128 × 128`
* RandomCrop to `112 × 112` during training
* CenterCrop during validation/test
* Horizontal flipping
* Rotation up to ±10°
* Dataset-specific normalization

Normalization:

```text
mean = 0.503
std  = 0.256
```

---

# 🔬 CNN Experiments

Several architectural factors were investigated.

### Padding

Compared:

```text
P = 0
P = 1
P = 2
```

The reference architecture uses:

```text
padding = 1
```

with 3×3 convolutions.

---

### Pooling

Compared:

* Max Pooling
* Average Pooling

---

### Number of filters

Three configurations were evaluated:

```text
Small:
16 → 32 → 64

Medium:
32 → 64 → 128

Large:
64 → 128 → 256
```

The smaller configuration achieved the best validation accuracy in the reported experiment:

```text
≈ 0.756
```

---

### 1×1 Convolution

The project also investigated adding `1×1` convolutions after convolutional blocks.

The experiment did not show a significant performance improvement on the selected dataset subset.

---

# 📊 CNN Results

The final comparison reported:

| Model        |    AUC-ROC |
| ------------ | ---------: |
| CNN          | **0.8406** |
| MLP baseline |     0.8075 |

The CNN therefore demonstrates the importance of a spatial inductive bias for image data.

The MLP must flatten the image into a vector, while the CNN preserves spatial locality and exploits:

* local receptive fields
* weight sharing
* hierarchical feature extraction

---

# ❤️ Part III — RNN, LSTM, GRU and Seq2Seq for ECG

## Dataset

The third experiment uses the **PTB-XL ECG dataset**.

The dataset contains:

* **21,799 ECG recordings**
* 12 ECG leads
* 5 diagnostic superclasses

The project processes ECG sequences containing:

```text
T = 1,000 time steps
```

The task is **multi-label ECG classification**.

---

# 🔄 Recurrent Architectures

Four approaches are investigated:

### 1. Vanilla RNN

Provides the baseline recurrent architecture.

### 2. Bidirectional LSTM

Introduces memory cells and gates to better preserve long-term dependencies.

### 3. Bidirectional GRU

Provides selective memory with fewer parameters than the LSTM.

### 4. Hybrid Seq2Seq Autoencoder

Combines:

```text
Encoder
    ↓
Latent representation
    ↓
Decoder
    ↓
ECG reconstruction
```

with a classification objective.

Teacher forcing is used during training.

---

# 🧪 ECG Data Augmentation

The project investigates ECG-specific augmentation techniques including:

* Gaussian jitter
* Time warp
* Window slicing

The objective is to increase training variability while preserving diagnostically relevant ECG morphology.

---

# 📊 ECG Results

Results on the reported PTB-XL test set:

| Architecture   | Macro AUC-ROC |   Macro F1 |  PPL Proxy |
| -------------- | ------------: | ---------: | ---------: |
| Vanilla RNN    |        0.7681 |          — |          — |
| LSTM           |        0.9158 |     0.7178 |     1.4000 |
| GRU            |    **0.9203** | **0.7233** | **1.3884** |
| Hybrid Seq2Seq |        0.9127 |          — |          — |

The results demonstrate a substantial improvement when moving from a vanilla RNN to architectures equipped with mechanisms for selective memory.

---

# 🔎 Greedy Decoding vs Beam Search

The Seq2Seq experiment also compares:

* Greedy decoding
* Beam Search with `k = 3`

Both methods achieved:

```text
Macro F1 = 0.7169
```

with:

```text
Global AUC-ROC = 0.9109
```

The experiment therefore did not observe a measurable F1 improvement from beam search in this multi-label setting.

---

# ⚙️ Training

The experiments use **PyTorch** and GPU acceleration.

The reported hardware includes:

```text
NVIDIA GeForce RTX 3070 Laptop GPU
```

The project makes use of several practical Deep Learning mechanisms:

* Adam optimizer
* Batch normalization
* Dropout
* learning-rate scheduling
* BCEWithLogitsLoss
* class weighting
* gradient clipping
* DataLoader
* GPU/CPU device management
* model checkpointing
* `state_dict`
* model reloading

---

# 📈 Global Comparison

The project demonstrates that there is no universally optimal architecture independent of the data modality.

### Tabular data

MLP provides a flexible baseline, but models such as XGBoost remain competitive on relatively small heterogeneous datasets.

### Image data

CNNs exploit spatial locality and parameter sharing, making them more appropriate for structured visual data.

### Sequential data

RNNs encode temporal ordering, while LSTM and GRU architectures introduce mechanisms that help preserve long-range dependencies.

---

## Architecture Selection Principle

The main conclusion of the project can be summarized as:

```text
Data structure
      ↓
Inductive bias
      ↓
Architecture
      ↓
Representation learning
      ↓
Performance
```

The architecture should therefore be selected according to the underlying structure of the data rather than simply by increasing model complexity.

---

# 🧪 Experimental Methodology

The project follows a common experimental workflow:

```text
Raw Medical Dataset
        ↓
Data Analysis
        ↓
Preprocessing
        ↓
Train / Validation / Test Split
        ↓
Architecture Design
        ↓
Training
        ↓
Validation
        ↓
Hyperparameter / Architecture Experiments
        ↓
Best Model Selection
        ↓
Test Evaluation
        ↓
Comparative Analysis
```

Particular attention was given to avoiding data leakage by fitting preprocessing transformations only on training data.

---

# 🛠️ Technologies

The project primarily uses:

* **Python**
* **PyTorch**
* **scikit-learn**
* **Pandas**
* **SDV / GaussianCopulaSynthesizer**
* **NumPy**
* **GPU acceleration with CUDA**
* **Jupyter Notebook / Python-based experimentation**

The reported PyTorch version is:

```text
PyTorch 2.5.1
```

---

# 📚 Datasets

### Breast Cancer MSK 2018

Clinical and genomic breast cancer data from Memorial Sloan Kettering Cancer Center, accessed through cBioPortal.

Used for:

```text
Tabular binary classification
        ↓
MLP / XGBoost
```

### RSNA Pneumonia Detection Challenge

Chest X-ray dataset containing DICOM radiographs.

Used for:

```text
Medical image classification
        ↓
CNN
```

### PTB-XL

Large publicly available electrocardiography dataset.

Used for:

```text
Multi-label ECG classification
        ↓
RNN / LSTM / GRU / Seq2Seq
```

---

# ⚠️ Limitations

The experiments have several important limitations.

## MLP

The synthetic dataset generated using Gaussian Copula may not reproduce all complex nonlinear interactions present in real clinical data.

Therefore, results obtained from the synthetic dataset should not be directly interpreted as equivalent to results obtained exclusively from the original patient population.

## CNN

The CNN experiments use a balanced subset rather than the entire available dataset.

Consequently, experiments involving padding, pooling and filter configurations have limited statistical power compared with experiments using the complete dataset.

## RNN / ECG

The classification task operates at the ECG-recording level rather than at the individual heartbeat level.

The generated synthetic ECG signals were also not clinically validated by cardiologists.

## Medical application

These models are experimental academic models and **are not clinical diagnostic systems**.

Their reported performance should not be interpreted as evidence of clinical readiness.

---

# 🚀 Future Work

The report identifies several potential directions for future development.

### Tabular Deep Learning

Investigate:

* TabTransformer
* FT-Transformer
* attention-based tabular architectures

### Medical Imaging

Investigate transfer learning and fine-tuning using architectures such as:

* DenseNet-121
* EfficientNet-B4

### ECG / Sequential Modeling

Investigate:

* Temporal Convolutional Networks
* Multi-Head Attention
* Transformer-based ECG architectures

These approaches could provide more efficient or expressive alternatives to conventional recurrent architectures.

---

# 🎓 Key Takeaways

This project demonstrates three fundamental principles:

### 1. Tabular ≠ Images ≠ Sequences

Different data modalities require different inductive biases.

### 2. More parameters do not automatically mean better performance

The CNN achieved:

```text
AUC = 0.8406
```

with approximately:

```text
4.4M parameters
```

while the reported MLP image baseline had approximately:

```text
6.5M parameters
```

and achieved:

```text
AUC = 0.8075
```

### 3. Architecture should follow data geometry

```text
Tabular
   → MLP

Spatial
   → CNN

Temporal
   → RNN / LSTM / GRU

Sequence-to-sequence
   → Seq2Seq
```

The central lesson is that **inductive bias is a fundamental component of Deep Learning architecture design**.

---

# 📖 References

The project report references the following foundational works:

1. Rosenblatt, F. (1958). *The Perceptron: A Probabilistic Model for Information Storage and Organization in the Brain.*

2. Rumelhart, D. E., Hinton, G. E., & Williams, R. J. (1986). *Learning representations by back-propagating errors.*

3. LeCun, Y., Bottou, L., Bengio, Y., & Haffner, P. (1998). *Gradient-Based Learning Applied to Document Recognition.*

4. Hochreiter, S., & Schmidhuber, J. (1997). *Long Short-Term Memory.*

5. Cho, K. et al. (2014). *Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation.*

6. Kingma, D. P., & Ba, J. (2015). *Adam: A Method for Stochastic Optimization.*

7. Srivastava, N. et al. (2014). *Dropout: A Simple Way to Prevent Neural Networks from Overfitting.*

8. Ioffe, S., & Szegedy, C. (2015). *Batch Normalization.*

9. He, K. et al. (2015). *Delving Deep into Rectifiers.*

10. Glorot, X., & Bengio, Y. (2010). *Understanding the Difficulty of Training Deep Feedforward Neural Networks.*

11. Vaswani, A. et al. (2017). *Attention Is All You Need.*

12. Wagner, P. et al. (2020). *PTB-XL, a large publicly available electrocardiography dataset.*

13. Shih, C.-F. et al. (2018). *RSNA Pneumonia Detection Challenge.*

---

# 👨‍💻 Author

**Alami Louati Ghali**

Engineering student — AI & Data Science
EMSI — École Marocaine des Sciences de l'Ingénieur

Academic Year: **2025–2026**

Module: **Deep Learning**

Supervisor: **Mme. HIDILA Zineb**

---

## ⭐ Project Summary

> **A comparative Deep Learning study demonstrating how MLP, CNN, RNN, LSTM, GRU and Seq2Seq architectures adapt to tabular clinical data, medical images and temporal ECG signals.**

The project combines theoretical foundations, PyTorch implementation, controlled experiments and quantitative evaluation to study the relationship between **data structure, inductive bias, architecture and model performance**.
