# Word2Vec

## Introduction

Word2Vec is a classic static word embedding model developed by Tomas Mikolov, et al. at Google in 2013. The original paper, "[Efficient Estimation of Word Representations in Vector Space](https://arxiv.org/abs/1301.3781)," introduced a method for learning high-quality, distributed vector representations of words from large amounts of unstructured text data.

A key breakthrough of Word2Vec is its ability to capture linguistic regularities and semantic relationships as linear directions in the vector space. This means that vector operations can reveal meaningful connections between words, famously demonstrated by the analogy "king - man + woman = queen."

## Implementation

This repository provides a high-performance implementation of three popular Word2Vec architectures:

1.  **CBOW (Continuous Bag-of-Words):** Predicts the current word based on a window of surrounding context words.
2.  **Skip-gram:** Predicts the surrounding context words given the current word.
3.  **NNLM (Neural Network Language Model):** A feedforward neural network that learns word embeddings as part of a language modeling task.

All models are trained using a **frequency-sampled negative cross-entropy loss** for efficient and effective learning.

## ✨ Special Feature: Speed

**Why another Word2Vec implementation?** Many existing repositories are toy implementations, not designed to handle large, real-world datasets. They can be slow and memory-intensive, making them impractical for serious use.

This implementation is built from the ground up for **speed and scalability**. It is highly optimized to leverage modern hardware, allowing you to process massive datasets and train models rapidly.

* **Benchmark:** Train on the Wiki103 dataset (over 5 million examples) in **less than 1 hour** on a single GPU.
* **Cost-Effective:** Train your own high-quality word embedding model for **less than $1**.

## Tested Setup

This project was developed and tested in the following environment. While it may work on other setups, this is the configuration that is confirmed to work.

* **OS:** Ubuntu 22.04
* **GPU:** CUDA 12.4

## Demo: Embedding Quality with Analogies

The quality of the learned embeddings can be demonstrated through word analogy tasks. The model can solve analogies like:

* `king - man + woman ≈ queen`
* `paris - france + germany ≈ berlin`
* `big - bigger + small ≈ smaller`

An analogy script or evaluation function will be provided to test these relationships with your trained model.

## Training Configurations

Training parameters and model configurations can be specified in a configuration file. Below is an example structure:

```yaml
# config/cbow-2.yaml
train_model.model_type = "cbow"
train_model.epochs=10
train_model.sparse_lr=0.01
train_model.dense_lr=0.001
```

## Getting Started
Follow these steps to get started with training your own Word2Vec model.

1. Clone the Repository:
```bash
git clone https://github.com/your-username/word2vec.git
cd word2vec
```
2. Set up the Environment:
```bash
sudo ./cuda_setup.sh
pip install -r requirements.txt
# alternatively just install this version of torch
pip install torch --index-url https://download.pytorch.org/whl/cu124/
# install the word2vec package for editable builds.
pip install -e .
```

3. Start Training:
```bash
mkdir emb ckpt data exps
python3 word2vec/main.py --gin-config-file=word2vec/configs/cbow-2.gin
```

4. Monitoring Training
```bash
tensorboard --logdir exps --port 8082 --bind_all
```

![Loss](img/loss.jpg "Loss")
![Loss](img/analogy.jpg "Analogy")

5. Evaluate Embeddings:

python evaluate.py --model_path /path/to/your/model.pt

Limitations
It is important to note that static word embeddings like Word2Vec are no longer considered state-of-the-art for most natural language processing tasks. A major limitation is that they assign a single, fixed vector to each word, regardless of its context.

Modern contextual language models, such as BERT (Bidirectional Encoder Representations from Transformers), generate dynamic embeddings that change based on the surrounding words in a sentence. This allows them to capture nuances and polysemy (multiple meanings of a word) far more effectively. While Word2Vec is foundational and still useful for certain applications and for learning purposes, for cutting-edge performance, models like BERT are recommended.

Author
This project was written by a human.