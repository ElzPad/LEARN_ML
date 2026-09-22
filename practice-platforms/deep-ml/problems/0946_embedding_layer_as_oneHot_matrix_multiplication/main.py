import numpy as np

def embedding_via_one_hot(token_ids, W):
    """
    Compute token embeddings via one-hot encoding and matrix multiplication.

    Args:
        token_ids: list or 1D array of integer token IDs
        W: numpy array of shape (vocab_size, embed_dim)

    Returns:
        numpy array of shape (len(token_ids), embed_dim)
    """

    # Solution via embedding lookup: return W[token_ids]

    num_tokens, vocab_size = len(token_ids), W.shape[0]
    H = np.zeros((num_tokens, vocab_size))
    H[range(num_tokens), token_ids] = 1

    return H @ W