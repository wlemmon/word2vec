import argparse


import gin
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import random
from tqdm import tqdm
import os
from torch.utils.tensorboard import SummaryWriter
from typing import Optional, Dict, List
import torch.multiprocessing as mp

import sys
from word2vec.utils import NegativeSampler

from word2vec.models import SkipGramModel, CBOWModel, NNLMModel
import logging
from word2vec.data import load
from word2vec.analogy import analogy_test, analogies

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"  # Hide excessive tensorflow debug messages

logging.basicConfig(stream=sys.stdout, level=logging.INFO)


def in_trainer_eval(model, eval_dataloader, writer):
    """Evaluate the model on the validation set."""
    # This function can be implemented to evaluate the model's performance
    # on a validation set, if available.
    pass


@gin.configurable
def train_model(
    name="",
    epochs=3,
    dense_lr=0.001,
    sparse_lr=0.01,
    log_step=100,
    model_type="skipgram",
    embedding_dim=100,
    negative_samples=10,
    emb_dir="emb",
    checkpoint_dir="ckpt",
    data_dir="tmp",
):
    # Load and preprocess data
    train_loader, eval_loader, actual_vocab_size, id_counts, tokenizer, window_size = load(
        model_type=model_type,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info("Running on Device", device)

    neg_sampler = NegativeSampler(
        id_counts, negative_samples, power=0.75, device=device
    )

    # Initialize model
    if model_type == "skipgram":
        model = SkipGramModel(actual_vocab_size, embedding_dim, neg_sampler)
    elif model_type == "cbow":
        model = CBOWModel(actual_vocab_size, embedding_dim, neg_sampler)
    elif model_type == "nnlm":
        model = NNLMModel(
            actual_vocab_size,
            embedding_dim,
            hidden_dim=embedding_dim,
            context_size=window_size * 2,
            neg_sampler=neg_sampler,
        )
    model.to(device)

    log_dir = f"./exps/{model_type}-{name}"
    writer = SummaryWriter(log_dir=log_dir)

    # Create lists to hold the parameter groups
    sparse_params = []
    dense_params = []

    for name, param in model.named_parameters():
        # Identify sparse parameters by layer name
        if "embed" in name:
            sparse_params.append(param)
            logging.info(f"Found sparse parameter: {name}")
        else:
            dense_params.append(param)
            logging.info(f"Found dense parameter: {name}")

    optimizer_sparse = torch.optim.SparseAdam(sparse_params, lr=sparse_lr)
    optimizer_dense = torch.optim.AdamW(dense_params, lr=dense_lr)

    for epoch in range(epochs):
        model.train()

        for batch_idx, train_data in enumerate(
            tqdm(train_loader, desc=f"Epoch {epoch+1}")
        ):
            if batch_idx==0:
                logging.info(f"Effective batch size {train_data[0].shape}")
            optimizer_sparse.zero_grad()
            optimizer_dense.zero_grad()
            loss = model.train_forward(train_data, device)

            if batch_idx % log_step == 0:
                
                step = epoch * len(train_loader) + batch_idx
                logging.info(f"Batch {batch_idx}, Loss: {loss.item():.4f}")
                writer.add_scalar("losses/main_loss", loss.item(), step)
                for word1, word2, word3 in analogies:
                    result = analogy_test(
                        word1, word2, word3, model.get_embeddings(), tokenizer
                    )
                    writer.add_text(
                        f"analogy/{word1}_{word2}_{word3}",
                        f"{word1}:{word2} :: {word3}: {result}",
                        step,
                    )
                in_trainer_eval(model, eval_loader, writer)

            loss.backward()
            optimizer_sparse.step()
            optimizer_dense.step()

    writer.flush()
    writer.close()

    # Save model and embeddings

    torch.save(model.state_dict(), f"{checkpoint_dir}/{model_type}-{name}_model.pth")
    torch.save(model.get_embeddings(), f"{emb_dir}/{model_type}-{name}_embeddings.pth")

    logging.info(f"\nModel saved as {model_type}-{name}_model.pth")
    logging.info(f"Embeddings saved as {model_type}-{name}_embeddings.pth")



# python3 word2vec/main.py --gin-config-file=word2vec/configs/cbow.gin
# python3 word2vec/main.py --gin-config-file=word2vec/configs/skipgram.gin
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A simple script with arguments.")
    parser.add_argument("--gin-config-file", help="")
    args = parser.parse_args()
    gin.parse_config_file(args.gin_config_file)
    train_model()
