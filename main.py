# ---------------------------------------------------------------------------------------
# NanoGPT Tutorial from Andrej Karpathy (starting point)
#
#   Get the tiny Shakespeare DataSet
#      https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
#
# https://www.youtube.com/watch?v=kCc8FmEb1nY
# reference code repository: https://github.com/karpathy/nanoGPT
# ---------------------------------------------------------------------------------------

import torch
import numpy as np
import bigramLanguageModel


def generate_data_from_model(m, n_tokens):
    """
    Generate n_tokens from a model
    :param m:
    :param n_tokens:
    :return:  sequence of generated embeddings
    """
    starting_encoding = torch.zeros((1, 1), dtype=torch.long)  # [1(b)x1(t)], one word
    predicted_next_indices = \
        m.generate(starting_encoding, max_new_tokens=n_tokens)  # [b, max_new_tokens]

    predicted_next_indices = predicted_next_indices.squeeze()  # remove batch dim

    return predicted_next_indices


def main():

    # Read in the tiny Shakespeare Dataset.
    with open("../../data/tinyShakespeare/input.txt") as f:
        text = f.read()  # text is a string object
    print("Length of Dataset {}".format(len(text)))

    # -----------------------------------------------------------------------------------
    # Set up a tokenizer (convert a string to embeddings)
    # We use character-level tokens
    #
    # Multiple other forms exist
    # Ref: https://towardsdatascience.com/top-5-word-tokenizers-that-every-nlp-data-scientist-should-know-45cc31f8e8b9
    # NLTK is a commonly used package for natural language processing applications.
    # The nltk.tokenize module offers several options for tokenizers.
    # google uses: https://github.com/google/sentencepiece
    # openai (GPT) uses: https://github.com/openai/tiktoken: a byte pair encoding scheme
    # -----------------------------------------------------------------------------------
    # Get the set of all characters
    characters = sorted(list(set(text)))
    vocab_size = len(characters)
    print("Text contains {} characters:{}".format(vocab_size, ''.join(characters)))

    # Create numerical representation of characters
    ctoi = {c: i for i, c in enumerate(characters)}
    itoc = {i: c for i, c in enumerate(characters)}

    # Functions to facilitate changing between string and encoding versions and visa-versa
    def encode(string):
        return [ctoi[c] for c in string]

    def decode(int_arr):
        return ''.join(itoc[i] for i in int_arr)
    # encode = lambda string: [ctoi[c] for c in string]
    # decode = lambda int_arr: ''.join(itoc[i] for i in int_arr)

    # -----------------------------------------------------------------------------------
    # Tokenize all the text
    # -----------------------------------------------------------------------------------
    data = encode(text)
    # Make tokenized inputs into pytorch tensors
    data = torch.tensor(data, dtype=torch.long)

    # Verify everything is working
    sample_encoded_data = data[:100]
    print("Sample data: {}".format(text[:100]))
    print("Sample encoded date:\n{}".format(sample_encoded_data))
    print("Sample data dtype {}".format(sample_encoded_data.dtype))
    print("Sample decoded data {}".format(decode(sample_encoded_data.numpy())))

    # -----------------------------------------------------------------------------------
    # Set up train and test dataset
    # -----------------------------------------------------------------------------------
    train_test_split = 0.9
    n_train = int(train_test_split * len(data))
    # n_test = len(data) - n_train

    train_data = data[:n_train]
    val_data = data[n_train:]

    # -----------------------------------------------------------------------------------
    # Set up data loaders for inputs to the model
    # -----------------------------------------------------------------------------------
    block_size = 8

    x = train_data[:block_size]
    y = train_data[1:block_size + 1]

    # [1] Task is to predict the next data.
    # note that in each block size, there are 8 example trainings
    for i in range(block_size):
        print("{}: Train {} --> Label {}".format(i, x[:i+1], y[i]))
    # x[:i+1] characters upto i and including i

    # note that seeing the data this way also has the advantage that the model sees inputs of
    # lengths up to the context length, not just the condext length.

    # [2] Setup multi batch inputs to speed up compute time
    batch_size = 8

    def get_batch(batch_s, blk_s, data_type='train'):
        x_batch = torch.zeros(batch_s, blk_s, dtype=torch.long)
        y_batch = torch.zeros(batch_s, blk_s, dtype=torch.long)

        if data_type.lower() == 'train':
            b_data = train_data
        else:
            b_data = val_data

        start_idx = np.random.randint(len(b_data) - blk_s, size=batch_s)
        for b_idx in range(batch_s):
            x_batch[b_idx, :] = b_data[start_idx[b_idx]: start_idx[b_idx] + blk_s]
            y_batch[b_idx] = b_data[start_idx[b_idx] + 1: start_idx[b_idx] + blk_s + 1]

        return x_batch, y_batch

    xb,  yb = get_batch(batch_size, block_size, 'train')
    print("Input Data [Size {}]\n{}".format(xb.shape, xb))
    print("Label [Size {}]\n{}".format(yb.shape, yb))
    print("-"*80)

    # -------------------------------------------------------------------------------------
    # training
    # -------------------------------------------------------------------------------------
    net = bigramLanguageModel.BigramLanguageModel(vocab_size)

    # Generate Code from the untrained model
    generated_embeddings = generate_data_from_model(net, 100)
    generated_text = decode(generated_embeddings.numpy())
    print("Generated Text\n{}".format(generated_text))

    # Start of Training
    # ---------------------------------------
    # Setup an optimizer
    lr = 1e-3
    optimizer = torch.optim.AdamW(net.parameters(), lr=lr)

    batch_size = 32
    n_iters = 10000

    optimizer.zero_grad(set_to_none=True)

    for n_idx in range(n_iters):
        bx, by = get_batch(batch_size, block_size, 'train')

        logits, loss = net(bx, by)
        loss.backward()
        optimizer.step()

        if n_idx % 1000 == 0:
            print("Iteration {}. Loss: {}".format(n_idx, loss.item()))

    # Generate Code from the trained model
    generated_embeddings = generate_data_from_model(net, 300)
    generated_text = decode(generated_embeddings.numpy())
    print("Generated Text\n{}".format(generated_text))


    import pdb
    pdb.set_trace()


if __name__ == "__main__":
    torch.manual_seed(10)
    # Get the tiny Shakespeare DataSet
    # (https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt)

    main()
