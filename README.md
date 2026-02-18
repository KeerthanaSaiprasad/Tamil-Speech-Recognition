# Tamil Speech Recognition Using XLSR Wav2Vec2.0 & CTC Algorithm
Automatic Speech Recognition (ASR) system for low-resource Tamil language via cross-lingual transfer learning.

## Overview

This project implements an end-to-end Automatic Speech Recognition (ASR) pipeline for Tamil speech using the Wav2Vec2 framework. The notebook demonstrates dataset preparation, tokenizer creation, preprocessing of audio-text pairs, model training, and evaluation.

The objective is to build a Tamil speech-to-text model capable of learning directly from raw audio using modern deep learning methods.

---

## Features

* Tamil speech dataset loading and cleaning
* Audio preprocessing and normalization
* Custom tokenizer creation
* Wav2Vec2 fine-tuning pipeline
* Model training and evaluation
* Word Error Rate (WER) tracking

---

## Dataset

The project uses a Tamil speech dataset consisting of audio recordings paired with text transcriptions.

Dataset preparation steps include:

* Removing unnecessary metadata
* Cleaning punctuation and special characters
* Normalizing Tamil text
* Audio resampling

---

## Installation

Install all required dependencies before running the notebook:

```bash
pip install datasets==2.2.2
pip install transformers==4.18.0
pip install torchaudio
pip install librosa
pip install jiwer
pip install soundfile
pip install huggingface_hub
pip install numpy
pip install pandas
pip install tqdm
```

Git LFS is required for handling model assets:

```bash
sudo apt-get install git-lfs
git lfs install
```

---

## Workflow

### 1. Authentication

Log in to Hugging Face to access datasets and model storage.

### 2. Dataset Preparation

* Load Tamil speech dataset
* Clean transcription text
* Convert audio into model-ready format

### 3. Tokenizer Creation

Build a character-level tokenizer suited for Tamil script.

### 4. Feature Extraction

Prepare audio features using the Wav2Vec2 processor.

### 5. Model Training

Fine-tune a Wav2Vec2 CTC model on Tamil speech.

### 6. Evaluation

Evaluate transcription quality using Word Error Rate (WER).

---

## Running the Notebook

1. Open the Tamil_Speech_Recognition.ipynb notebook in Jupyter or Colab
2. Run cells sequentially
3. Complete Hugging Face authentication
4. Monitor training progress

> Training time depends on dataset size and hardware. GPU acceleration is recommended.

---

## Output

The notebook produces:

* A trained Tamil speech recognition model
* Tokenizer and processor artifacts
* Evaluation metrics

These outputs can be reused for inference or deployment.

---

## Requirements

* Python 3.8+
* GPU recommended for training
* Hugging Face account

---

## Future Improvements

* Larger Tamil datasets
* Noise augmentation
* Real-time inference pipeline
* Deployment as an API

---

## Acknowledgements

* Open-source Tamil speech datasets
* Hugging Face ecosystem

---

## License

Use according to the dataset and framework licensing terms.

---

## Notes

This notebook is intended for experimentation and learning. Performance may vary depending on dataset size, training duration, and hardware availability.

---

Happy building Tamil ASR systems!

