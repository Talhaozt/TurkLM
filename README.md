# 🇹🇷 TurkLM — Character-Level Turkish GPT Model

[![Live Demo](https://img.shields.io/badge/Live_Demo-Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://turklm-k4gp744ybqjc6772rhs87h.streamlit.app)
[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Gradio](https://img.shields.io/badge/Gradio-UI-FF6B6B?style=for-the-badge&logo=gradio&logoColor=white)](https://gradio.app/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)

---

## 🇹🇷 TÜRKÇE ÖZET / TURKISH SUMMARY

> ### **TurkLM Nedir?**
> **TurkLM**, PyTorch kullanılarak sıfırdan geliştirilmiş, karakter düzeyinde (character-level) çalışan bir Türkçe üretici dil modelidir (Generative Pre-trained Transformer). 
>
> Verilen bir başlangıç cümlesini (prompt) alarak Türkçe dil yapısına uygun biçimde devam ettirmek üzere eğitilmiştir.
>
> **Öne Çıkan Özellikler:**
> - **Karakter Seviyesinde Tokenizasyon:** Sözlük sınırı olmadan esnek metin üretimi.
> - **Görsel Web Arayüzü:** Gradio tabanlı modern ve parametre kontrollü kullanıcı arayüzü.
> - **Modüler Yapı:** Eğitim (`train.py`), üretim (`generate.py`) ve arayüz (`app.py`) modülleri ayrılmıştır.

---

## 📌 Overview

**TurkLM** is a custom, character-level Generative Pre-trained Transformer (GPT) language model built from scratch in PyTorch for the Turkish language. 

The model receives a text prompt and generates plausible Turkish text continuations character-by-character using multi-head self-attention mechanisms.

---

## 🛠 Model Architecture & Specifications

| Parameter | Value | Description |
| :--- | :--- | :--- |
| **Model Type** | Decoder-Only GPT | Autoregressive character-level Transformer |
| **Embedding Dimension (`d_model`)** | `384` | Dimensionality of token & positional embeddings |
| **Attention Heads (`num_heads`)** | `6` | Number of multi-head attention mechanisms |
| **Transformer Layers (`num_layers`)** | `6` | Number of Transformer block stacks |
| **Context Size (`block_size`)** | `256` | Maximum context window sequence length |
| **Dataset** | `vngrs-ai/vngrs-web-corpus` | Streamed, filtered, & cleaned Turkish web text |

---

## 🚀 Quick Start & Installation

### 1. Clone the Repository
```bash
git clone https://github.com/YOUR_USERNAME/TurkLM.git
cd TurkLM
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Place Model Checkpoint
Ensure your trained model checkpoint file is placed at:
```text
checkpoints/model.pt
```

---

## 🎨 Running the Web Interface (Gradio)

Launch the interactive web user interface:

```bash
python app.py
```

Once launched, open your browser and navigate to:
👉 **`http://127.0.0.1:7860`**

### Interactive Features:
- **Prompt Input:** Enter any starting Turkish sentence or phrase.
- **Temperature Control:** Adjust creativity (`0.1` for deterministic outputs, `0.8` - `1.0` for creative generation).
- **Max Characters:** Set maximum character length to generate (`50` to `1000`).

---

## 🖥 Command Line Interface (CLI)

To generate text directly from the terminal:

```bash
python generate.py
```

---

## 🏋️ Training the Model

To train the model from scratch or on your own dataset:

```bash
python train.py
```

---

## 📁 Repository Structure

```text
TurkLM/
├── app.py            # Gradio Web Application UI
├── generate.py       # Text generation & inference engine
├── model.py          # PyTorch GPT Architecture & Multi-Head Attention
├── dataset.py        # Data streaming, cleaning & CharacterTokenizer
├── train.py          # Model training loop & checkpointing
├── requirements.txt  # Project dependencies
└── checkpoints/
    └── model.pt      # Model weights & configuration dictionary
```

---

## 📜 License
Distributed under the MIT License. See `LICENSE` for more information.
