# ---------------------------------------------------------------------------------------
# GptModel using pytorch builtin transformer module functions
#
# * Model uses transformer module's encoder block with a mask.
# * Pytorch's transformer decoder block uses cross-attention and expects an encoder output
#   (memory) in its forward function.
# * The encoder block allows you to specify a causal mask. [essentially the transformer
#   'decoder' block we want].
# ---------------------------------------------------------------------------------------
import torch
import torch.nn as nn
import numpy as np


def get_positional_embeddings_matrix(max_pos, embed_dim):
    """
    Returns a [max_pos, embed] matrix of positional embeddings.
    Uses exp(log of the division term)to simplify implementation

    https://medium.com/@hunter-j-phillips/positional-encoding-7a93db4109e6
    https://pytorch.org/tutorials/beginner/transformer_tutorial.html

    :param max_pos:
    :param embed_dim:
    :return:
    """
    mat = torch.zeros((max_pos, embed_dim), dtype=torch.float64)

    pos_idxs = torch.arange(0, max_pos).reshape(max_pos, 1)  # [max_pos, 1]

    div_term = torch.exp(torch.arange(0, embed_dim, 2) * np.log(10000) / embed_dim)

    mat[:, 0::2] = torch.sin(pos_idxs * div_term)
    mat[:, 1::2] = torch.cos(pos_idxs * div_term)

    return mat


class GptModel(nn.Module):
    def __init__(self, vocab_s, embed_dim, block_s, n_attn_heads, n_layers, device, p_dropout=0.5):
        """
        :param vocab_s:
        :param embed_dim:
        :param block_s:
        :param n_attn_heads:
        :param n_layers:
        :param p_dropout:
        :param device:
        """
        super().__init__()

        self.vocab_s = vocab_s
        self.embed_dim = embed_dim
        self.block_s = block_s
        self.n_attn_heads = n_attn_heads
        self.p_dropout = p_dropout
        self.n_layers = n_layers
        self.device = device

        # 'Decoder' block = Pytorch's encoder block with a causal mask.
        # Pytorch's builtin decoder block includes a cross attention blocks & expects a memory
        # (encoder output) parameter in its forward function. It is based on the original transformer
        # paper which includes a cross attention layer.
        # Each encoder block consists of: multi-headed attention block + feedforward layer block
        # + layer normalization layers before each block.
        encoder = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_attn_heads,
            dim_feedforward=4*embed_dim,
            dropout=p_dropout,
            activation='relu',
            batch_first=True,
            norm_first=True
        )

        # Layers  ---------------------------------------------------------------

        # Word/ Symbol Embeddings. [vocab_s, embed_dim] mat.
        self.token_embed = nn.Embedding(num_embeddings=vocab_s, embedding_dim=embed_dim)

        # Positional embeddings
        pos_embed_table = get_positional_embeddings_matrix(max_pos=block_s, embed_dim=embed_dim)
        self.pos_embed = nn.Embedding.from_pretrained(pos_embed_table, freeze=True)
        # Note: PyTorch does not like creation of new variables in the forward function. Cannot move them to
        # device. Need to either (1) add device as an input parameter, or (2) declare the variable in
        # initialization and use it in the forward function. Register buffer, tells pytorch that this is
        # a variable whose gradient does not need to be tracked.
        self.register_buffer('pos_v', torch.arange(block_s))

        # Stack of decoder blocks
        # TODO: investigate the optional norm parameter of the Transformer Encoder
        self.blocks = nn.TransformerEncoder(encoder_layer=encoder, num_layers=n_layers)

        # Map from embedding dimension to output classes for final output.
        self.linear = nn.Linear(in_features=embed_dim, out_features=vocab_s)

    def forward(self, x_in, y_in=None):
        """
        :param x_in: [B, T]  [Batch, time] matrix of input embeddings indexes
        :param y_in: [B,T], [Batch, time] matrix of target embeddings indexes
        """
        b, t = x_in.shape

        token_embeddings = self.token_embed(x_in)   # [b,t, embed_dim]
        pos_embeddings = self.pos_embed(x_in)  # [t, embed_dim]
        pos_embeddings = pos_embeddings.to(torch.float32)  # float32 like the token embed

        x = token_embeddings + pos_embeddings  # [B, T, embed_dim]

        mask = nn.Transformer.generate_square_subsequent_mask(sz=t, device=self.device)
        x = self.blocks(x, mask=mask, is_causal=True)  # [B, T, embed_dim] (after concatenation)

        logits1 = self.linear(x)  # [B,T, vocab_s]

        loss1 = None
        if y_in is not None:
            logits_a = logits1.reshape(b * t, self.vocab_s)  # [b*t, vocab_s]
            y_in = y_in.reshape(b * t)
            # cross entropy loss expects input in the format (..., ch, ...)
            loss1 = nn.functional.cross_entropy(logits_a, y_in)

        return logits1, loss1

    def generate(self, x_in, max_new_tokens):
        for idx in range(max_new_tokens):
            # Get the predictions
            logits1, _ = self(x_in[:, -self.block_s:])  # calls forward function, (nn.module). logits1 = [b, t, c]

            # Focus on the last time step
            logits1 = logits1[:, -1, :]  # [b, t, c] --> [b, c]. Note that when this function is used b == 1

            # Get probabilities from logits
            probs = nn.functional.softmax(logits1, dim=1)  # softmax across the channel dimension, [b, 1]

            # Randomly select an index based on the probability distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # one next_idx for each batch

            # Append sampled index to the running index sequence
            x_in = torch.cat((x_in, idx_next), dim=1)  # [b, t+1]

        return x_in


if __name__ == "__main__":

    model = GptModel(vocab_s=4, embed_dim=64, block_s=8, n_attn_heads=2, n_layers=1, p_dropout=0.1, device='cpu')
    print(model)

    from torchsummary import summary
    summary(model, input_size=(8, 8))


    import pdb
    pdb.set_trace()