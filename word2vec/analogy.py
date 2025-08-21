import argparse
import gin
import torch
from word2vec.data import load


def find_nearest_neighbors(word_embed, embeddings, tokenizer, k=5):
    """Find k nearest neighbors for a given word"""

    # Compute cosine similarities
    similarities = torch.cosine_similarity(word_embed, embeddings)

    # Get top k similar words (excluding the word itself)
    _, top_indices = torch.topk(similarities, k + 1)

    neighbors = []
    for idx in top_indices.flatten():  # Skip the first one (the word itself)
        neighbors.append((tokenizer.convert_ids_to_tokens(idx.item()), similarities[idx].item()))

    return neighbors


def analogy_test(word1, word2, word3, embeddings, tokenizer):
    """Test analogy: word1 is to word2 as word3 is to ?"""
    vocab = set(tokenizer.get_vocab())
    if not all(word in vocab for word in [word1, word2, word3]):
        return None

    # Get embeddings
    embed1 = embeddings[tokenizer.convert_tokens_to_ids(word1)]
    embed2 = embeddings[tokenizer.convert_tokens_to_ids(word2)]
    embed3 = embeddings[tokenizer.convert_tokens_to_ids(word3)]

    # Compute analogy vector: word2 - word1 + word3
    analogy_vector = embed2 - embed1 + embed3

    # Find most similar words
    return find_nearest_neighbors(analogy_vector.unsqueeze(0), embeddings, tokenizer)


analogies = [
    ("man", "king", "woman"),  # Should give "queen"
    ("paris", "france", "italy"),  # Should give "rome"
    ("good", "better", "bad"),  # Should give "worse"
    ("big", "bigger", "small"),  # Should give "smaller"
]


@gin.configurable
def main(
    embeddings_file="emb/none.pth",
):

    _, _, _, _, tokenizer, _ = load(
    )

    embeddings = torch.load(embeddings_file, map_location=torch.device("cpu"))

    for word1, word2, word3 in analogies:
        result = analogy_test(word1, word2, word3, embeddings, tokenizer)
        print(
            f"analogy/{word1}_{word2}_{word3}",
            f"{word1}:{word2} :: {word3}: {result}",
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A simple script with arguments.")
    parser.add_argument("--gin-config-file", help="")
    args = parser.parse_args()
    if args.gin_config_file:
        gin.parse_config_file(args.gin_config_file)
    main()
