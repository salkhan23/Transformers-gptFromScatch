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
from datetime import datetime
import torch.utils.data as data

import bigramLanguageModel
import gptModel
import gptModelPytorchBlocks
import shakespeareDataSet

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Using {} device".format(device))


def generate_data_from_model(m, n_tokens):
    """
    Generate n_tokens from a model
    :param m:
    :param n_tokens:
    :return:  sequence of generated embeddings
    """
    starting_encoding = torch.zeros((1, 1), dtype=torch.long).to(device)  # [1(b)x1(t)], one word
    predicted_next_indices = \
        m.generate(starting_encoding, max_new_tokens=n_tokens)  # [b, max_new_tokens]

    predicted_next_indices = predicted_next_indices.squeeze()  # remove batch dim

    return predicted_next_indices


def main():

    # -----------------------------------------------------------------------------------
    # Model variables
    # -----------------------------------------------------------------------------------
    block_size = 256  # context size
    batch_size = 64
    lr = 1e-6
    eval_iters = 200
    train_test_split = 0.9
    n_epochs = 10
    embed_dim = 384
    n_heads = 6
    n_layers = 6
    p_dropout = 0.1

    # -----------------------------------------------------------------------------------
    # Data Access
    # -----------------------------------------------------------------------------------
    data_set = shakespeareDataSet.ShakespeareDataSet(max_context_len=block_size, device=device)
    train_set, val_set = torch.utils.data.random_split(data_set, [train_test_split, 1 - train_test_split])

    train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)

    # ------------------------------------------------------------------------------------
    # Loss function
    # ------------------------------------------------------------------------------------
    loss_fcn = torch.nn.CrossEntropyLoss()

    @torch.no_grad()
    def estimate_loss(model):
        model.eval()
        out = {}
        for split in ['val', 'train']:
            losses1 = torch.zeros(eval_iters)

            data_loader = train_dataloader if split == 'train' else test_dataloader

            for iter_idx, (x1, y1) in enumerate(data_loader):

                logits1 = model(x1)
                b1, t1, ch1 = logits1.shape  # Use mat sizes vs know dim as list batch might be of a different shape
                # cross entropy loss expects input in the format (..., ch, ...). Reshape matrices to include
                # batch and context length into a single dimension and such that ch appears in the second dimension.
                loss1 = loss_fcn(logits1.reshape(b1 * t1, ch1), y1.reshape(b1 * t1))

                losses1[iter_idx] = loss1.item()

                if iter_idx == eval_iters - 1:
                    break

            out[split] = losses1.mean()
        model.train()
        return out

    # -------------------------------------------------------------------------------------
    # training
    # -------------------------------------------------------------------------------------
    # net = bigramLanguageModel.BigramLanguageModel(data_set.vocab_size)
    # net = gptModel.GptModel(
    #     data_set.vocab_size, embed_dim=embed_dim, block_s=block_size, n_attn_heads=n_heads, n_layers=n_layers,
    #     p_dropout=p_dropout)
    net = gptModelPytorchBlocks.GptModel(
        data_set.vocab_size, embed_dim=embed_dim, block_s=block_size, n_attn_heads=n_heads, n_layers=n_layers,
        p_dropout=p_dropout, device=device)

    net = net.to(device)
    print("Model Details: {}".format('*'*60))
    gptModel.print_model_parameters(net)

    # # DEBUG: Generate code from the untrained model
    # generated_embeddings = generate_data_from_model(net, 100)
    # generated_text = decode(generated_embeddings.cpu().numpy())
    # print("Pretraining Generated Text\n{}".format(generated_text))

    # Start of Training
    # ---------------------------------------
    # Setup an optimizer
    optimizer = torch.optim.AdamW(net.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    print("Start Training ...")
    train_start_time = datetime.now()

    optimizer.zero_grad(set_to_none=True)
    for e_idx in range(n_epochs):

        epoch_start_time = datetime.now()

        for step_idx, (bx, by) in enumerate(train_dataloader):
            logits = net(bx)

            # Loss
            b, t, ch = logits.shape  # Use mat sizes vs know dim as list batch might be of a different shape
            # cross entropy loss expects input in the format (..., ch, ...). Reshape matrices to include
            # batch and context length into a single dimension and such that ch appears in the second dimension.
            loss = loss_fcn(logits.reshape(b * t, ch), by.reshape(b * t))

            loss.backward()
            optimizer.step()

        # Estimate Losses
        losses = estimate_loss(net)

        print("Epoch {}, Duration {}. Train loss {:0.4f} Test Loss {:0.4f}, lr={}".format(
            e_idx, datetime.now() - epoch_start_time, losses['train'], losses['val'], scheduler.get_last_lr()))

        scheduler.step()

    print("Training Finished. Duration {}".format(datetime.now() - train_start_time))

    # Generate Code from the trained model
    generated_embeddings = generate_data_from_model(net, 300)
    generated_text = data_set.decode(generated_embeddings.cpu().numpy())
    print("Post Training Generated Text\n{}".format(generated_text))

    import pdb
    pdb.set_trace()


if __name__ == "__main__":
    torch.manual_seed(10)
    # Get the tiny Shakespeare DataSet
    # (https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt)

    main()
