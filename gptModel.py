# ---------------------------------------------------------------------------------------
# Gpt Model
# ---------------------------------------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F


def get_positional_embeddings_matrix(max_pos, embed_dim):
    """
    Returns a [max_pos, embed] matrix of positional embeddings.

    :param max_pos:
    :param embed_dim:
    :return:
    """
    mat = torch.zeros((max_pos, embed_dim), dtype=torch.float64)

    for pos in range(max_pos):
        for i in range(embed_dim // 2):
            angle = torch.tensor(pos/(10000**(2*i/embed_dim)))
            mat[pos, 2*i] = torch.sin(angle)
            mat[pos, 2*i+1] = torch.cos(angle)

    return mat


def print_model_parameters(m):
    """
    Print Model Parameters, their size and whether they are trainable or not
    :param m:
    :return:
    """
    print("{}\nModel Details\n{}".format('-' * 90, '-' * 90))

    n_train_p = 0
    n_fixed_p = 0

    for name, param in m.named_parameters():
        print("Parameter: {}, size {}, Trainable {}".format(name, param.shape, param.requires_grad))

        if param.requires_grad:
            n_train_p += torch.numel(param)
        else:
            n_fixed_p += torch.numel(param)

    print("\nTotal Number of Parameters {}, Trainable {}, Fixed {}".format(n_train_p + n_fixed_p, n_train_p, n_fixed_p))


class AttentionHead(nn.Module):
    def __init__(self, embed_dim, head_dim, max_context_len, causal=False):
        """

        :param embed_dim: input embeddings dimensionality
        :param head_dim: output dimensionality of head
        :params max_context_len: max length of context window
        :param causal: whether inputs can look at future time steps or not. Default = True
        """
        super().__init__()
        self.causal = causal
        self.embed_dim = embed_dim
        self.head_dim = head_dim

        # Layers
        self.key = nn.Linear(embed_dim, head_dim, bias=False)
        self.query = nn.Linear(embed_dim, head_dim, bias=False)
        self.value = nn.Linear(embed_dim, head_dim, bias=False)

        # for masked attention
        if causal:
            self.register_buffer('tril', torch.tril(torch.ones((max_context_len, max_context_len)))),  # [T, T]

    def forward(self, x_in):
        """
        :param x_in: [B,T, embed_dim] matrix of indices
        """
        b, t, ch = x_in.shape

        q = self.query(x_in)  # [B,T, head_dim]
        k = self.key(x_in)  # [B, T, head_dim]
        v = self.value(x_in)  # [B,T,head_dim]

        s = q @ torch.transpose(k, 2, 1) / self.head_dim**0.5  # [B,T, head_dim] * [B,head_dim, T]= [B, T, T]

        if self.causal:
            # Ali Ghodsi way: softmax(k.T @ v + m)
            # Mask should be  [0 -inf, -inf]
            #                 [0,   0, -inf]
            #                 [0,   0,    0]
            # TODO: add Ali Ghodsi way

            # Andrej Karpathy way
            # Mask [1,-inf, -inf]
            #      [1,   1, -inf]
            #      [1,   1,    1]
            mask = self.tril[0:t, 0:t]
            s = s.masked_fill(mask == 0, float("-inf"))

        a = F.softmax(s, dim=1)  # [B, head_dim, T] @ [B, T, T] = [B, head_dim, T]
        # TODO: Better Performance and generated text if softmax over second dimension vs. last.
        #  Both are T Why is one better than the other?

        z = a @ v  # [B, T, T] @ [B, T, head_dim]   [B, T, head_dim]

        # Note diff b/w notes and this implementation. Due to PyTorch's different way of storing mats

        return z


class MultiHeadedAttention(nn.Module):
    def __init__(self, embed_dim, n_heads, max_context_len, causal=False):
        """

        :param embed_dim:
        :param n_heads:
        :param max_context_len:
        :param causal:
        """

        super().__init__()
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.causal = causal
        self.max_context_len = max_context_len

        if embed_dim % n_heads != 0:
            raise Exception(
                "Embedded dimension ({}) should be perfectly divisible by num attention heads ({})".format(
                    embed_dim, n_heads))
        self.single_head_dim = embed_dim // n_heads

        # layers ---------
        self.attention_heads = nn.ModuleList(
            AttentionHead(embed_dim=embed_dim, head_dim=self.single_head_dim, causal=causal,
                          max_context_len=max_context_len) for _ in range(n_heads))

    def forward(self, x_in):
        """

        :param x_in: [B,T, embed_dim]
        :return:
        """
        z = torch.concat([head(x_in) for head in self.attention_heads], dim=2)  # concatenate on embed dim
        return z


class FeedForward(nn.Module):
    def __init__(self, in_ch, out_ch):
        """
        Simple Feedforward Layer
        :param in_ch:
        :param out_ch:
        """
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch

        # Layers ---------------------
        self.ff = nn.Linear(in_ch, out_ch)

    def forward(self, x_in):
        """
        :param x_in: [B,T,in_ch]
        :return: [B,T, out_ch]
        """
        x = self.ff(x_in)
        x = torch.relu(x)
        return x


class DecoderBlock(nn.Module):
    def __init__(self, embed_dim, n_heads, max_context_len):
        """
        Decoder block of a Transformer
        :param embed_dim:
        :param n_heads:
        :param max_context_len:
        """
        super().__init__()

        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.max_context_len = max_context_len

        # Layers -------------------------------------------------------------------------
        self.self_attention = \
            MultiHeadedAttention(embed_dim=embed_dim, n_heads=n_heads, causal=True, max_context_len=max_context_len)
        self.attn_layer_norm = nn.LayerNorm(embed_dim)

        self.feedforward = FeedForward(embed_dim, embed_dim)
        self.ff_layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, x_in):
        """

        :param x_in: [B,T,embed_dim]
        :return:
        """
        x = self.self_attention(x_in)
        x = x + self.attn_layer_norm(x)  # Layer norm + residual connection

        x = self.feedforward(x)
        x = x + self.ff_layer_norm(x)  # Layer norm + residual connection

        return x


class GptModel(nn.Module):
    def __init__(self, vocab_s, embed_dim, block_s, n_attn_heads):
        """

        :param vocab_s:
        :return:
        """
        super().__init__()

        self.vocab_s = vocab_s
        self.embed_dim = embed_dim
        self.block_s = block_s
        self.n_attn_heads = n_attn_heads

        # Declare the layers  ----------------------------------------------------------------

        # Word/ Symbol Embeddings
        # self.embedding_table.weight = [vocab_s, embed_dim] mat. Each row is a separate vocab word
        self.token_embed = nn.Embedding(num_embeddings=vocab_s, embedding_dim=embed_dim)

        # Positional embeddings
        pos_embed_table = get_positional_embeddings_matrix(max_pos=block_s, embed_dim=embed_dim)
        self.pos_embed = nn.Embedding.from_pretrained(pos_embed_table, freeze=True)
        # Note: PyTorch does not like creation of new variables in the forward function. Cannot move them to
        # device. Need to either (1) add device as an input parameter, or (2) declare the variable in
        # initialization and use it in the forward function. Register buffer, tells pytorch that this is
        # a variable whose gradient does not need to be tracked.
        self.register_buffer('pos_v', torch.arange(block_s))

        self.transformer_decoder_block = \
            DecoderBlock(embed_dim=embed_dim, n_heads=n_attn_heads, max_context_len=block_s)

        # Map from embedding dimension to output classes for final output.
        self.linear = nn.Linear(in_features=embed_dim, out_features=vocab_s)

    def forward(self, x_in, y_in=None):
        """

        :param x_in: [B, T]  [Batch, time] matrix of input embeddings indexes
        :param y_in: [B,T], [Batch, time] matrix of output embeddings indexes

        :return:
        """
        b, t = x_in.shape

        token_embeddings = self.token_embed(x_in)  # [B,T, embed_dim]
        pos_embeddings = self.pos_embed(self.pos_v[:t])  # [t, embed_dim]
        pos_embeddings = pos_embeddings.to(torch.float32)  # float32 like the token embed

        # Add position and token embedding.
        # Broadcasting handles the extension to the batch dim.
        # pos_emb: [t,embed_dim] --> [1,t, embed_dim]  -->[b,t,embed_dim]
        x = token_embeddings + pos_embeddings  # [B, T, embed_dim]

        x = self.transformer_decoder_block(x)

        logits1 = self.linear(x)  # [B,T, vocab_s]

        loss1 = None
        if y_in is not None:

            logits_a = logits1.reshape(b*t, self.vocab_s)  # [b*t, vocab_s]
            y_in = y_in.reshape(b*t)
            # cross entropy loss expects input in the format (..., ch, ...)
            loss1 = F.cross_entropy(logits_a, y_in)

        return logits1, loss1

    def generate(self, x_in, max_new_tokens):

        for idx in range(max_new_tokens):
            # Get the predictions
            logits1, _ = self(x_in[:, -self.block_s:])  # calls forward function, (nn.module). logits1 = [b, t, c]

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
    block_size = 8  # context window length

    model = GptModel(vocab_s=vocab_size, embed_dim=32, block_s=block_size, n_attn_heads=8)

    input1 = torch.tensor([5, 1, 1, 0], dtype=torch.long)
    input1 = torch.unsqueeze(input1, dim=0)  # add the batch dimension

    logits, loss = model(input1)

    # print("Loss {}".format(loss))
    print("Embeddings\n{}".format(logits))
    print("logits.shape {}".format(logits.shape))

    print_model_parameters(model)

    import pdb
    pdb.set_trace()
