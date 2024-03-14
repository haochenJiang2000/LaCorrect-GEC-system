from torch.utils.data import Dataset
from transformers import AutoTokenizer, BertTokenizer

padding = "max_length"
max_source_length = 512


class MyDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.tokenizer = tokenizer
        self.data = data

    def __getitem__(self, index):
        input = self.data[index].lower()
        inputs = input
        model_inputs = self.tokenizer(inputs, max_length=max_source_length, padding=padding, truncation=True)

        return model_inputs

    def __len__(self):
        return len(self.data)

# dataset = MyDataset("data/test.src", "data/train.info", "T5-small")
