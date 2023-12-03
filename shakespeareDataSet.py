import torch


class ShakespeareDataSet(torch.utils.data.Dataset):
    def __init__(self, max_context_len=10, device=None):
        """
        A custom Dataset class must implement three functions: __init__, __len__, and __getitem__.
        https://pytorch.org/tutorials/beginner/basics/data_tutorial.html

        :param max_context_len:
        :param device:
        """
        super().__init__()

        self.data_dir = "../../data/tinyShakespeare/input.txt"
        self.max_context_len = max_context_len

        # Read in the tiny Shakespeare Dataset.
        with open(self.data_dir) as f:
            text = f.read()  # text is a string object
        print(" Total number data points {}".format(len(text)))

        # Set up a tokenizer (convert words/characters to embedding indexes)
        # We use character-level tokens. Multiple other forms exist:
        # Ref: https://towardsdatascience.com/top-5-word-tokenizers-that-every-
        # nlp-data-scientist-should-know-45cc31f8e8b9
        #
        # NLTK is a commonly used package for natural language processing applications.
        # The nltk.tokenize module offers several options for tokenizers.
        # google uses: https://github.com/google/sentencepiece
        # openai (GPT) uses: https://github.com/openai/tiktoken: a byte pair encoding scheme
        # -----------------------------------------------------------------------------------
        # Get the set of all characters
        self.characters = sorted(list(set(text)))
        self.vocab_size = len(self.characters)
        print("Data set contains {} characters:{}".format(self.vocab_size, ''.join(self.characters)))

        # Create numerical representation of characters
        self.ctoi = {c: i for i, c in enumerate(self.characters)}
        self.itoc = {i: c for i, c in enumerate(self.characters)}

        self.data = self.encode(text)  # encode data
        self.data = torch.tensor(self.data, dtype=torch.long)
        if device is not None:
            self.data = self.data.to(device)

    def encode(self, string):
        return [self.ctoi[c] for c in string]

    def decode(self, indices):
        return "".join(self.itoc[i] for i in indices)

    def __len__(self):
        return len(self.data) - self.max_context_len - 1

    def __getitem__(self, i):
        return \
            self.data[i: i + self.max_context_len], \
            self.data[i + 1:i + 1 + self.max_context_len]


if __name__ == "__main__":
    torch.manual_seed(10)
    context_window_s = 10

    data_set = ShakespeareDataSet(max_context_len=context_window_s)

    # Test the Dataset ------------------------------------------------------------------------------------------
    idx = 0
    data, label = data_set[idx]
    print("Data: index {};\nEncoded Data  {},\nEncoded Label {}.".format(idx, data, label))
    print("Decoded Data  : {}.\nDecoded Label: {}".format(
        data_set.decode(data.cpu().numpy()),
        data_set.decode(label.cpu().numpy())))
    print("Shape of input {}. Shape of label {}".format(data.shape, label.shape))

    idx = data_set.__len__()
    data, label = data_set[idx]
    print("Data: index {};\nEncoded Data  {},\nEncoded Label {}.".format(idx, data, label))
    print("Decoded Data  : {}.\nDecoded Label: {}".format(
        data_set.decode(data.cpu().numpy()),
        data_set.decode(label.cpu().numpy())))

    # split the dataset into train and validation sets -----------------------------------------------
    train_test_split = 0.9
    n_train = int(train_test_split * data_set.__len__())
    train_set, val_set = torch.utils.data.random_split(data_set, [train_test_split, 1-train_test_split])

    # DataLoaders  -----------------------------------------------------------------------------------
    batch_size = 1
    train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=False)
    # Note shuffle is for between epochs, not within

    # Iterate over the dataset
    # -------------------------------------------------------------------------------------------------
    for idx, (b_xin, b_label) in enumerate(train_dataloader):

        # b_xin, b_label  # [batch_size, context_window_s]

        # print first batch
        print('*' * 80)
        print(data_set.decode(b_xin[0, ].cpu().numpy()))
        print(data_set.decode(b_label[0, ].cpu().numpy()))

        if idx == 10:
            break

    import pdb
    pdb.set_trace()
