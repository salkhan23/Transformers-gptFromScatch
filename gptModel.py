# ---------------------------------------------------------------------------------------
# Gpt Model
# ---------------------------------------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F


class GptModel(nn.Module):
    def __init__(self, vocab_s, embed_dim):
        """

        :param vocab_s:
        :return:
        """
        super().__init__()

        self.vocab_s = vocab_s

        # Declare the layers
        self.embedding_table = nn.Embedding(num_embeddings=vocab_s, embedding_dim=embed_dim)
        self.linear = nn.Linear(in_features=embed_dim, out_features=vocab_s)

    def forward(self, x_in, y_in=None):
        """

        :param x_in: [B,T]  [Batch, time] matrix of input embeddings indexes
        :param y_in: [B,T], [Batch, time] matrix of output embeddings indexes

        :return:
        """
        x = self.embedding_table(x_in)  # [B,T, embed_dim]
        logits1 = self.linear(x)        # [B,T, vocab_s]
        b, t, ch = logits1.shape  # ch = vocab_s

        loss1 = None
        if y_in is not None:

            logits_a = logits1.reshape(b*t, ch)  # [b*t, ch]
            y_in = y_in.reshape(b*t)
            # cross entropy loss expects input in the format (..., ch, ...)
            loss1 = F.cross_entropy(logits_a, y_in)

        return logits1, loss1

    def generate(self, x_in, max_new_tokens):

        for idx in range(max_new_tokens):
            # Get the predictions
            logits1, _ = self(x_in)  # calls forward function, (nn.module)
            # logits1 = [b, t, c]

            # Focus on the last time step
            logits1 = logits1[:, -1, :]  # [b, t, c] --> [b, c]. Note that when this function is used b == 1

            # get probabilities from logits
            probs = F.softmax(logits1, dim=1)  # softmax across the channel dimension, [b,1]

            # randomly select an index based on the probability distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # one next_idx for each batch

            # Append sampled index to the running index sequence
            x_in = torch.cat((x_in, idx_next), dim=1)  # [b, t+1]

        return x_in


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

    model = GptModel(vocab_s=vocab_size, embed_dim=256)

    input1 = torch.tensor([5, 1, 1, 0], dtype=torch.long)
    input1 = torch.unsqueeze(input1, dim=0)  # add the batch dimension

    logits, loss = model(input1)

    # print("Loss {}".format(loss))
    print("Embeddings\n{}".format(logits))
    print("logits.shape {}".format(logits.shape))

    import pdb
    pdb.set_trace()
