# ---------------------------------------------------------------------------------------
# Very simple Bigram Language Model
# ---------------------------------------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F


class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_s):
        """
        Bigram Model.
        A type of statistical language model that predicts the probability of a word in a
        sequence based on the previous word. It considers pairs of consecutive words
        (bigrams) and estimates the likelihood of encountering a specific word given the
        preceding word in a text or sentence.
        :param vocab_s:
        """
        super().__init__()

        # Creates an embeddings table.
        # This module is often used to store word embeddings and retrieve them using indices.
        # The input to the module is a list of indices, and the output is the corresponding
        # word embeddings.
        #
        # This is simply a linear layer with some convenience functions to use indices
        # instead of input vectors
        # -------------------------------------------------------------------------------
        self.token_embedding_table = nn.Embedding(num_embeddings=vocab_s, embedding_dim=vocab_s)

    def forward(self, indices):
        """
        :param indices:  [b, t]; b = num batches, t = num time steps ()
        :return: embeddings
        """

        # For each input index, replace it with its word embedding
        embeddings = self.token_embedding_table(indices)  # [b, t, d] where d = embedding dimension

        b, t, ch = embeddings.shape

        logits1 = embeddings  # logits because we consider this the output of the model

        return embeddings

    def generate(self, indices, max_new_tokens):
        """

        :param indices: [b, t] matrix
        :param max_new_tokens:
        :return:
        """

        for _ in range(max_new_tokens):
            # Get the predictions
            logits1 = self(indices)  # calls forward function, (nn.module)

            # Focus on the last time step
            logits1 = logits1[:, -1, :]  # [b, t, c] --> [b, c]. Note that when this function is used b == 1

            # get probabilities from logits
            probs = F.softmax(logits1, dim=1)  # softmax across the channel dimension, [b,1]

            # randomly select an index based on the probability distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # one next_idx for each batch

            # append sampled index to the running index sequence
            indices = torch.cat((indices, idx_next), dim=1) # [b, t+1]

        return indices


if __name__ == "__main__":
    torch.manual_seed(10)

    word_to_idx = {
        "Noor": 0,
        "you": 1,
        "want": 2,
        "to": 3,
        "build": 4,
        "a": 5,
        "?": 6
    }
    vocab_size = len(word_to_idx)

    model = BigramLanguageModel(vocab_size)

    input1 = torch.tensor([5, 1, 1, 0], dtype=torch.long)
    input1 = torch.unsqueeze(input1, dim=0)  # add the batch dimension

    logits, loss = model(input1)

    # print("Loss {}".format(loss))
    print("Embeddings\n{}".format(logits))
    print("logits.shape {}".format(logits.shape))

    import pdb
    pdb.set_trace()
