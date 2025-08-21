import sys
import os
import torch
import gin
from torch.utils.data import DataLoader
import logging
from datasets import load_dataset, load_from_disk
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace
from itertools import islice

os.environ["TOKENIZERS_PARALLELISM"] = "true"
logging.basicConfig(stream=sys.stdout, level=logging.INFO)


class SkipGramCollator:
    def __init__(self, window_size):
        self.window_size = window_size

    def __call__(self, batch):
        all_centers = []
        all_contexts = []
        full_window_size = self.window_size * 2 + 1

        # Create indices for context words, excluding the center
        context_indices = torch.arange(full_window_size)
        context_indices = context_indices[context_indices != self.window_size]

        for example in batch:
            # Get the number of actual tokens, excluding padding
            present = example["attention_mask"].sum().item()

            if present >= full_window_size:
                ids = example["input_ids"][:present]
                unfolded = ids.unfold(0, full_window_size, 1)

                # Extract centers and contexts
                centers = (
                    unfolded[:, self.window_size]
                    .reshape(-1, 1)
                    .tile(1, self.window_size * 2)
                    .flatten()
                )
                contexts = unfolded[:, context_indices].flatten()

                all_centers.append(centers)
                all_contexts.append(contexts)

        if not all_centers:
            # Return empty tensors with correct shape if batch is too small
            return torch.empty(0, self.window_size * 2, dtype=torch.long), torch.empty(
                0, 1, dtype=torch.long
            )

        # Concatenate all examples in the batch into single tensors
        final_contexts = torch.cat(all_contexts, dim=0)
        final_centers = torch.cat(all_centers, dim=0)

        # Return in (input, target) format
        return (final_centers, final_contexts)


class CBOWCollator:
    def __init__(self, window_size):
        self.window_size = window_size

    def __call__(self, batch):
        all_centers = []
        all_contexts = []

        full_window_size = self.window_size * 2 + 1

        # Create indices for context words, excluding the center
        context_indices = torch.arange(full_window_size)
        context_indices = context_indices[context_indices != self.window_size]

        for example in batch:
            # Get the number of actual tokens, excluding padding
            present = example["attention_mask"].sum().item()

            if present >= full_window_size:
                ids = example["input_ids"][:present]

                # Use .unfold() to create sliding windows
                unfolded = ids.unfold(0, full_window_size, 1)

                # Extract centers and contexts
                centers = unfolded[:, self.window_size].unsqueeze(1)
                contexts = unfolded[:, context_indices]

                all_centers.append(centers)
                all_contexts.append(contexts)

        if not all_centers:
            # Return empty tensors with correct shape if batch is too small
            return torch.empty(0, self.window_size * 2, dtype=torch.long), torch.empty(
                0, 1, dtype=torch.long
            )

        # Concatenate all examples in the batch into single tensors
        final_contexts = torch.cat(all_contexts, dim=0)
        final_centers = torch.cat(all_centers, dim=0)

        # Return in (input, target) format
        return (final_centers, final_contexts)


def batch_iterator_words(stream, batch_size=1000):
    it = iter(stream)
    while True:
        batch = list(islice(it, batch_size))
        if not batch:
            break
        yield [ex["text"] for ex in batch]


@gin.configurable
def load(
    dataset=[],
    vocab_size=10000,
    window_size=5,
    model_type="",
    batch_size=40,  # explicit batch size is smaller than effective batch size since we explode the data on the fly. effecive batch size can be around 1400.
    num_readers=8,
    padding_max_length=128,
    data_dir="tmp",
):
    # load and preprocess sentence datasets
    ds = "_".join(dataset)

    def build_tokenizer(dataset):
        # Build tokenizer
        tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(
            vocab_size=vocab_size,
            special_tokens=["[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]"],
        )
        tokenizer.train_from_iterator(
            batch_iterator_words(dataset), trainer=trainer, length=len(dataset)
        )
        from transformers import PreTrainedTokenizerFast

        hf_tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=tokenizer,
            unk_token="[UNK]",
            pad_token="[PAD]",
            cls_token="[CLS]",
            sep_token="[SEP]",
            mask_token="[MASK]",
        )
        hf_tokenizer.save_pretrained(f"{data_dir}/{ds}_wordlevel_tokenizer")
        return hf_tokenizer

    def count_id_freq(tokenized_train, vocab_size):
        def counter(batch):
            B = batch.flatten()
            return {
                "input_ids": [
                    torch.zeros(vocab_size, dtype=torch.long).index_add(
                        0, B, torch.ones_like(B)
                    )
                ]
            }

        def reduce(batch):
            return {"input_ids": [batch.sum(dim=0)]}

        id_counts = tokenized_train.map(
            counter,
            batched=True,
            input_columns="input_ids",
            remove_columns=tokenized_train.column_names,
            keep_in_memory=True,
        )
        while id_counts.num_rows > 1:
            id_counts = id_counts.map(
                reduce, batched=True, input_columns="input_ids", keep_in_memory=True
            )
        id_counts = id_counts[0]["input_ids"]

        return id_counts

    try:
        from transformers import PreTrainedTokenizerFast

        logging.info("started loading cached datasets...")
        tokenized_train = load_from_disk(f"{data_dir}/{ds}_tokenized_train")
        tokenized_val = load_from_disk(f"{data_dir}/{ds}_tokenized_eval")
        tokenizer = PreTrainedTokenizerFast.from_pretrained(
            f"{data_dir}/{ds}_wordlevel_tokenizer"
        )
        logging.info("Loaded cached dataset...")
    except Exception as e:
        """Load dataset and preprocess it"""
        logging.info(f"Loading {ds} dataset...")
        dataset = load_dataset(*dataset)

        # Use `train_test_split` to downsample
        train_dataset = dataset["train"]
        eval_dataset = dataset["validation"]

        tokenizer = build_tokenizer(train_dataset)

        # Define a tokenization function
        def tokenize_function(examples):
            return tokenizer(
                examples["text"],  # column name in dataset
                padding="max_length",  # pad to fixed length
                truncation=True,  # truncate too-long sequences
                max_length=padding_max_length,
            )

        logging.info("Tokenizing dataset...")

        tokenized_train = train_dataset.map(
            tokenize_function, batched=True
        ).remove_columns("text")
        tokenized_val = eval_dataset.map(
            tokenize_function, batched=True
        ).remove_columns("text")
        tokenized_train.set_format(type="torch")
        tokenized_val.set_format(type="torch")

        tokenized_train.save_to_disk(f"{data_dir}/{ds}_tokenized_train")
        tokenized_val.save_to_disk(f"{data_dir}/{ds}_tokenized_eval")
    vocab_size = len(tokenizer)
    id_counts = count_id_freq(tokenized_train, vocab_size)

    collate_fn = (
        SkipGramCollator(window_size=window_size)
        if model_type == "skipgram"
        else CBOWCollator(window_size=window_size)
    )

    logging.debug("=" * 60 + "\n")

    # Create data loaders
    train_loader = DataLoader(
        tokenized_train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_readers,
        pin_memory=True,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        tokenized_val,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_readers,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    return train_loader, val_loader, vocab_size, id_counts, tokenizer, window_size
