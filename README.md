# 🧒 Child Development Fact vs Myth Checker

This Streamlit web application classifies child-development statements as **FACT**, **MYTH**, or **UNCERTAIN** using a fine-tuned **BERT (bert-base-uncased)** model.  
The goal of this project is to help reduce misinformation and promote evidence-based child-development knowledge.

---

## 🤖 BERT Classification Pipeline

### 🔹 Data Preparation
- Child-development statements were cleaned and normalized.
- Labels were standardized and encoded for binary classification (FACT / MYTH).
- Ambiguous statements were removed to improve training quality.

---

### 🔹 Tokenization
- Text is processed using the **BERT tokenizer**.
- Converts statements into:
  - Token IDs
  - Attention masks
  - Special tokens (`[CLS]`, `[SEP]`)
- Input sequences are truncated and padded to a maximum length of **128 tokens**.

---

### 🔹 Model Training
- A pretrained **BERT sequence classification model** was fine-tuned.
- Training configuration:
  - Learning rate: `2e-5`
  - Batch size: `8`
  - Epochs: `3`
  - Weight decay applied for regularization
- The best-performing checkpoint was automatically selected during training.

---

### 🔹 Model Evaluation
Model performance was evaluated using:
- Accuracy
- Precision
- Recall
- F1-score

Evaluation was performed on unseen test data.

---

### 🔹 Inference & Prediction
- User input is normalized and tokenized.
- The trained BERT model generates prediction probabilities using Softmax.
- Predictions include confidence scores.
- Low-confidence outputs are labeled as **UNCERTAIN** for reliability.

---

### 🔹 Deployment
- The fine-tuned BERT model is integrated into a **Streamlit interface**.
- Provides real-time classification.
- Displays input normalization steps for transparency.



