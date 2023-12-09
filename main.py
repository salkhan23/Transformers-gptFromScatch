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
from datetime import datetime
import torch.utils.data as data
from tqdm import tqdm
import os

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


def save_checkpoint(model, optim, epoch, loss, results_dir, save_optim_weights=False):
    """
    :param model:
    :param optim:
    :param epoch:
    :param loss:
    :param results_dir:
    :param save_optim_weights:
    :return:
    """
    checkpoint_file = os.path.join(results_dir, 'training_checkpoints', "model_" + str(epoch) + ".pt")
    if not os.path.exists(os.path.dirname(checkpoint_file)):
        os.makedirs(os.path.dirname(checkpoint_file))

    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optim.state_dict(),
        'loss': loss,
    }, checkpoint_file)

    if save_optim_weights:
        optim_weights_file = os.path.join(results_dir, 'training_checkpoints', "model_optim.pt")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optim.state_dict(),
            'loss': loss,
        }, optim_weights_file)


def main(prev_saved_model_file=None):

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
    base_results_store_dir = './results'

    # -----------------------------------------------------------------------------------
    # Data Access
    # -----------------------------------------------------------------------------------
    data_set = shakespeareDataSet.ShakespeareDataSet(max_context_len=block_size, device=device)
    train_set, val_set = torch.utils.data.random_split(data_set, [train_test_split, 1 - train_test_split])

    train_dataloader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=True)

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
    # Model, Optimizer & LR scheduler
    # -------------------------------------------------------------------------------------
    # net = bigramLanguageModel.BigramLanguageModel(data_set.vocab_size)
    # net = gptModel.GptModel(
    #     data_set.vocab_size, embed_dim=embed_dim, block_s=block_size, n_attn_heads=n_heads, n_layers=n_layers,
    #     p_dropout=p_dropout)
    net = gptModelPytorchBlocks.GptModel(
        data_set.vocab_size, embed_dim=embed_dim, block_s=block_size, n_attn_heads=n_heads, n_layers=n_layers,
        p_dropout=p_dropout, device=device)

    net = net.to(device)

    # Setup an optimizer
    optimizer = torch.optim.AdamW(net.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    if prev_saved_model_file is None:
        print("Model Details: {}".format('*'*60))
        gptModel.print_model_parameters(net)

        results_store_dir = os.path.join(
            base_results_store_dir,
            net.__class__.__name__ + '_' + datetime.now().strftime("_%Y%m%d_%H%M%S"))

        if not os.path.exists(results_store_dir):
            os.makedirs(results_store_dir)

        start_epoch = 0
        losses = {"train": 10000, "val": 10000}
        print("Train from scratch.")

    else:
        results_store_dir = os.path.dirname(prev_saved_model_file)
        checkpoint = torch.load(prev_saved_model_file)
        net.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        start_epoch = checkpoint['epoch']
        losses = checkpoint['loss']
        print("Resume training. epoch {}".format(start_epoch))

    # -----------------------------------------------------------------------------------
    # Training
    # ----------------------------------------------------------------------------------
    print("Start Training ...")
    train_start_time = datetime.now()

    optimizer.zero_grad(set_to_none=True)
    min_val_loss = losses['val']
    for e_idx in range(start_epoch, n_epochs):

        epoch_start_time = datetime.now()

        for step_idx, (bx, by) in enumerate(tqdm(train_dataloader)):
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
        if losses['val'] < min_val_loss:
            min_val_loss = losses['val']

        print("Epoch {}, Duration {}. Train loss {:0.4f} Test Loss {:0.4f}, lr={}".format(
            e_idx, datetime.now() - epoch_start_time, losses['train'], losses['val'], scheduler.get_last_lr()))

        save_checkpoint(net, optimizer, e_idx, losses, results_store_dir, min_val_loss == losses['val'])
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

    # saved_model = "./results/GptModel__20231209_113956/training_checkpoints/model_3.pt"
    saved_model = None
    main(saved_model)
