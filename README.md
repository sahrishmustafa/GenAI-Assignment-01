# Deep Learning Architectures on CIFAR-10 and Text Data

This repository contains implementations of three core deep learning architectures — **Convolutional Neural Networks (CNNs)**, **Recurrent Neural Networks (RNNs)**, and **Pixel Recurrent Neural Networks (PixelRNNs)** — across three major tasks.  
Each task focuses on a different aspect of deep learning, from image classification and sequence modeling to autoregressive image generation.

---

## Overview of Questions

### **Question 1: Implementing a CNN for CIFAR-10**
This task focuses on designing and training a **Convolutional Neural Network (CNN)** for image classification using the **CIFAR-10** dataset.  
Students explore convolutional architectures, feature extraction, and model evaluation by training a CNN and conducting a detailed **ablation study** on key hyperparameters.

**Key Objectives:**
- Load and preprocess CIFAR-10 data from Hugging Face.
- Implement a CNN using PyTorch.
- Train and evaluate using metrics (Accuracy, Precision, Recall, F1-Score).
- Visualize feature maps and confusion matrices.
- Conduct an ablation study on:
  - Learning Rate  
  - Batch Size  
  - Number of Filters  
  - Number of Layers

  
## Features

- Modular and flexible CNN architecture (`FlexibleCNN`)
- Automatic training, validation, and testing pipeline
- Data augmentation with PyTorch `transforms`
- Ablation study for:
  - Learning Rate
  - Batch Size
  - Number of Convolutional Filters
  - Number of Layers
- Metrics computed: **Accuracy, Precision, Recall, F1-score**
- Visualization tools:
  - Training vs Validation loss curves
  - Confusion Matrix
  - Feature Map visualization

---

## Model Overview

The CNN model uses multiple convolutional blocks followed by a fully connected classifier.  
Each block includes:

- 3×3 Convolution with padding  
- Batch Normalization  
- ReLU activation  
- MaxPooling (applied after every 2 convolutional layers)

The classifier head includes:
- A fully connected layer (512 units, ReLU)  
- Dropout (p=0.5)  
- Output layer with 10 units (CIFAR-10 classes)

---

## Training Pipeline

The experiment workflow is encapsulated in the `run_experiment()` function, which performs:

1. **Model initialization** with selected hyperparameters  
2. **Training loop** using SGD optimizer with momentum and weight decay  
3. **Learning rate scheduling** using `ReduceLROnPlateau`  
4. **Model checkpointing** on best validation loss  
5. **Final evaluation** on the test dataset  
6. **Visualization** of training dynamics and feature maps  

---

## Ablation Study

To study the effect of different hyperparameters, the `ablation_study()` function automatically runs multiple experiments:

| Parameter | Tested Values |
|------------|----------------|
| Learning Rate | 0.001, 0.01, 0.1 |
| Batch Size | 16, 32, 64 |
| Convolutional Filters | [16, 32, 64], [32, 64, 128], [64, 128, 256] |
| Number of Layers | 3, 5, 7 |

All results are logged and summarized in a CSV file (`ablation_summary.csv`) for easy comparison.

---

## Data Preprocessing

The CIFAR-10 dataset is downloaded using the Hugging Face `datasets` library and transformed with:

- Random cropping, flipping, rotation, color jitter
- Random affine transformations and random erasing (for regularization)
- Normalization using CIFAR-10 statistics

---

## Results

The best CNN configuration achieved:

| Metric | Value |
|--------|--------|
| **Accuracy** | ~90% |
| **Precision** | High across most classes |
| **Recall** | Balanced performance |
| **F1-Score** | Consistent improvement with deeper layers |

Visual outputs (confusion matrix and feature maps) are saved automatically during training.

---

---

### **Question 2: Implementing an RNN for Next-Word Prediction**
This task involves building a **Recurrent Neural Network (RNN)** (LSTM or GRU) trained on **Shakespeare’s text dataset** to perform next-word prediction.  
Students implement text preprocessing, train embeddings, and analyze model performance.

**Key Objectives:**
- Load and preprocess text from the Hugging Face `tiny_shakespeare` dataset.  
- Train a custom embedding layer and RNN (LSTM/GRU).  
- Generate text predictions from a seed phrase.  
- Evaluate using metrics such as perplexity and accuracy.  
- Perform an ablation study (e.g., on hidden size, layers, or dropout).  

Model Overview
--------------

The architecture consists of:

*   An **embedding layer** to convert words into dense vector representations.
    
*   One or more **LSTM layers** to model sequential context.
    
*   **Dropout** regularization (active when multiple layers are used).
    
*   A **fully connected output layer** that maps hidden states to vocabulary logits.
    

The model is trained using **cross-entropy loss**, and evaluated using **accuracy** and **perplexity**.

Data Preprocessing
------------------

The dataset is tokenized and normalized before training.Common **English stopwords** (e.g., _the, is, and, of_) are removed to emphasize meaningful content words.A vocabulary is built from the processed tokens, with and symbols included.Training samples are created using a sliding window of **five context words** predicting the next one.

Training Pipeline
-----------------

Each experiment is run using a standardized function that:

1.  Initializes the model with chosen hyperparameters.
    
2.  Trains using the **Adam** optimizer with weight decay for regularization.
    
3.  Applies **early stopping** (patience = 3) to prevent overfitting.
    
4.  Logs loss, accuracy, and perplexity per epoch.
    
5.  Generates sample text after training using **top-$k$** sampling and **temperature scaling**.
    

**Tuned Hyperparameters:**

ParameterTested ValuesBest ValueWeight Decay0, 1e-41e-4Patience2, 33Top-$k$30, 5050Temperature0.7, 1.01.0

Text Generation
---------------

After training, the model can generate text from any seed phrase using a helper function:

`   generate_from_seed(model, "To be or not to", gen_len=200)   `

This uses temperature-controlled and top-$k$ sampling to produce diverse yet coherent sequences.

Ablation Study and Results
--------------------------

A series of controlled experiments were conducted to analyze how hidden size, number of layers, and dropout affect performance.All models were trained for 20 epochs using the same optimization settings.

### Effect of Hidden Size

Hidden SizeLayersDropoutValidation Perplexity6410.018.7925610.0**17.74**

Larger hidden dimensions improved context modeling, yielding lower perplexity and higher accuracy.

### Effect of Dropout and Layers

LayersDropoutValidation Perplexity20.1**17.94**20.218.0820.318.2830.117.9430.217.9930.3**17.93**

*   With one layer, dropout had negligible effect (inactive in PyTorch).
    
*   For two and three layers, moderate dropout ($0.1$–$0.2$) offered the best balance between learning and regularization.

---

### **Question 3: Implementing PixelCNN, Row LSTM, and Diagonal BiLSTM (PixelRNN)**
The final task reproduces the **PixelRNN architectures** described in *van den Oord et al., 2016* for autoregressive image generation.  
Students implement three model types and compare them using CIFAR-10:

**Architectures Implemented:**
- **PixelCNN** using masked convolutions (Type A/B masks).  
- **Row LSTM** scanning image rows sequentially.  
- **Diagonal BiLSTM** processing diagonals using bidirectional recurrence.  

**Key Objectives:**
- Implement masked and recurrent convolutional models.  
- Train models on CIFAR-10 using negative log-likelihood (bits/dim).  
- Compare models in terms of likelihood and qualitative image generation.

### **Architectures Implemented**

*   **PixelCNN** — a fully convolutional autoregressive model using **masked convolutions** (Type A and B) to enforce the pixel generation order.
    
*   **Row LSTM** — a recurrent architecture that scans each image row by row, allowing dependencies to propagate horizontally and vertically.
    
*   **Diagonal BiLSTM** — an advanced variant processing image **diagonals** using bidirectional LSTMs to capture both forward and backward dependencies efficiently.
    

### **Key Objectives**

*   Implement and train **masked** and **recurrent** generative models.
    
*   Learn the autoregressive distribution over image pixels.
    
*   Evaluate models using **negative log-likelihood (bits-per-dimension)**.
    
*   Generate and visualize samples from trained models.
    
*   Compare the expressiveness and efficiency of PixelCNN vs. PixelRNN variants.
    

### **Understanding Bits-per-Dimension (BPD)**

The **Bits-per-Dimension (BPD)** metric measures how well the model fits the data — essentially how _compressible_ the images are under the model’s learned probability distribution.

**Interpretation:**

*   Lower BPD → Model assigns higher likelihood to real images (better predictive power).
    
*   BPD ≈ expected **number of bits per pixel** required to encode the image.
    
*   It connects deep learning with **information theory**: a model that can predict the next pixel distribution well is also one that can **compress the image efficiently**.
    

For CIFAR-10 (32×32 RGB), good models typically achieve **3.0–3.5 BPD**.Random or poorly trained models have **BPD > 8**, indicating nearly uniform (uninformative) predictions.

### **Training Pipeline**

Each model shares a consistent training and evaluation procedure:

1.  **Data Loading & Preprocessing**
    
    *   CIFAR-10 images are normalized to \[0, 1\] and converted to integers 0–255 for discrete likelihood modeling.
        
    *   Dataloaders use batch normalization and optional data augmentation.
        
2.  **Loss Function**
    
    *   The models predict a categorical distribution (256-way softmax) for each pixel channel.
        
        
3.  **Optimization**
    
    *   Optimizer: **Adam** (learning rate = 1e-3)
        
    *   Gradient clipping (max-norm = 1.0) to stabilize LSTM-based models
        
    *   Learning rate scheduling via ReduceLROnPlateau
        
4.  **Evaluation & Visualization**
    
    *   Training and validation BPD curves are plotted over epochs.
        
    *   Random image samples are generated pixel-by-pixel after training.
        
    *   Model checkpoints are saved for best validation BPD.

---

## Project Structure

```
├── q1/
│ └── cnn_pytorch.ipynb # CNN implementation and ablation study (CIFAR-10)
│
├── q2/
│ └── rnn_pytorch_.ipynb # RNN (LSTM/GRU) for next-word prediction (Shakespeare dataset)
│
├── q3/
│ └── pixelrnn.ipynb # PixelCNN, Row LSTM, Diagonal BiLSTM implementation
│
├── report.pdf # Combined project report
├── README.md # This documentation
└── requirements.txt # Dependencies (PyTorch, NumPy, etc.)
```


## Requirements

Install dependencies with:

```bash
pip install -r requirements.txt