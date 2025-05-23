import torch
from torch.utils.data import IterableDataset

import os
import pyarrow.parquet as pq
import pyarrow as pa
from datasets import load_dataset

import huggingface_hub as hf

# # Define an IterableDataset for shakespeare dataset
# class ShakespeareDataset(IterableDataset):
#     def __init__(self, root, train=True, block_size=256):
#         self.train = train
#         self.block_size = block_size
#         with open(os.path.join(root, 'shakespeare/input.txt'), 'r', encoding='utf-8') as f:
#             text = f.read()
#         # here are all the unique characters that occur in this text
#         self.chars = sorted(list(set(text)))
#         self.vocab_size = len(self.chars)
#         # create a mapping from characters to integers
#         self.stoi = { ch:i for i,ch in enumerate(self.chars) }
#         self.itos = { i:ch for i,ch in enumerate(self.chars) }

#         data = torch.tensor(self.encode(text), dtype=torch.long)
#         n = int(0.9*len(text)) # first 90% will be train, rest validation
#         if train:
#             self.dataset =  data[:n]
#         else:
#             self.dataset = data[n:]

#     def encode(self, s):               
#         return [self.stoi[c] for c in s] # encoder: take a string, output a list of integers
    
#     def decode(self, l):
#         return ''.join([self.itos[i] for i in l]) # decoder: take a list of integers, output a string

#     def __iter__(self):
#         indices = torch.randperm(len(self.dataset) - self.block_size)
#         for i in indices:
#             x = self.dataset[i:i+self.block_size].clone()
#             y = self.dataset[i+1:i+self.block_size+1].clone()
#             yield x, y

#     def __len__(self):
#         return len(self.dataset) - self.block_size


# Define an IterableDataset
class TextDataset(IterableDataset):
    def __init__(self, dataset, tokenizer, block_size, text_key='text', dp_chunks=1):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.text_key = text_key
        self.dp_chunks = dp_chunks
        self.dp_chunk = 0
        self.chunk_size = len(self.dataset) // self.dp_chunks
        self.start_idx = 0
        self.end_idx = self.chunk_size

    def _shard_dataset(self, dp_chunk):
        self.start_idx = dp_chunk * self.chunk_size
        self.end_idx = self.start_idx + self.chunk_size

    def __iter__(self):
        buffer = []
        # indices = torch.randperm(self.chunk_size) + start_idx  # Randomize indices within the chunk (affects PP - don't do it!)
        indices = torch.arange(self.start_idx, self.end_idx)  # Sequential indices within the chunk
        for i in indices:
            # example = self.dataset[i]
            example = self.dataset[i.unsqueeze(0)]
            if not example or not example.get(self.text_key):
                continue
            # Tokenize the text
            assert isinstance(example[self.text_key], list)
            # tokens = self.tokenizer.encode(example, add_special_tokens=False)
            tokens = self.tokenizer.encode(example[self.text_key][0], add_special_tokens=False)
            # assert tokens, "Tokens should not be empty"
            if not tokens:
                continue
            buffer.extend(tokens)
            while len(buffer) >= self.block_size + 1:
                x = torch.tensor(buffer[:self.block_size], dtype=torch.long)
                y = torch.tensor(buffer[1:self.block_size+1], dtype=torch.long)
                buffer = buffer[self.block_size:]
                yield x, y

    def __len__(self):
        return len(self.dataset)    # incorrect estimation! (it should be # tokens/block size)

# Define an IterableDataset
class StreamingTextDataset(IterableDataset):
    def __init__(self, dataset, tokenizer, block_size, text_key='text', dp_chunks=1):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.text_key = text_key
        self.dp_chunks = dp_chunks
        self.dp_chunk = 0
        self.shard = self.dataset.shard(num_shards=self.dp_chunks, index=self.dp_chunk)

    def _shard_dataset(self, dp_chunk):
        self.dp_chunk = dp_chunk
        self.shard = self.dataset.shard(num_shards=self.dp_chunks, index=self.dp_chunk)

    def __iter__(self):
        buffer = []
        for example in self.shard:
            if not example or not example.get(self.text_key):
                continue
            # Tokenize the text
            assert isinstance(example[self.text_key], str)
            # tokens = self.tokenizer.encode(example, add_special_tokens=False)
            tokens = self.tokenizer.encode(example[self.text_key], add_special_tokens=False)
            # assert tokens, "Tokens should not be empty"
            if not tokens:
                continue
            buffer.extend(tokens)
            while len(buffer) >= self.block_size + 1:
                x = torch.tensor(buffer[:self.block_size], dtype=torch.long)
                y = torch.tensor(buffer[1:self.block_size+1], dtype=torch.long)
                buffer = buffer[self.block_size:]
                yield x, y


# Define an TextDataset
class ShakespeareDataset(TextDataset):
    def __init__(self, tokenizer, train=True, block_size=256, dp_chunks=1):
        split = 'train' if train else 'test'
        dataset = load_dataset('Trelis/tiny-shakespeare')[split]
        super(ShakespeareDataset, self).__init__(dataset, tokenizer, block_size, text_key='Text', dp_chunks=dp_chunks)

# Define an TextDataset
class WikiTextDataset(TextDataset):
    def __init__(self, tokenizer, train=True, block_size=256, dp_chunks=1):
        split = 'train' if train else 'validation'
        dataset = load_dataset('wikitext', 'wikitext-103-v1')[split]
        super(WikiTextDataset, self).__init__(dataset, tokenizer, block_size, text_key='text', dp_chunks=dp_chunks)

# Define an TextDataset
class OpenWebTextDataset(TextDataset):
    def __init__(self, tokenizer, train=True, block_size=256, dp_chunks=1):
        split = 'train' if train else 'test'
        dataset = load_dataset('openwebtext', trust_remote_code=True)
        dataset = dataset['train'].train_test_split(test_size=0.1, seed=42)[split]
        super(OpenWebTextDataset, self).__init__(dataset, tokenizer, block_size, text_key='text', dp_chunks=dp_chunks)

# Define an TextDataset
class BookCorpusDataset(TextDataset):
    def __init__(self, tokenizer, train=True, block_size=256, dp_chunks=1):
        split = 'train' if train else 'test'
        dataset = load_dataset('bookcorpus/bookcorpus', trust_remote_code=True)
        dataset = dataset['train'].train_test_split(test_size=0.1, seed=42)[split]
        super(BookCorpusDataset, self).__init__(dataset, tokenizer, block_size, text_key='text', dp_chunks=dp_chunks)

    # Define an TextDataset
class FineWebDataset(StreamingTextDataset):
    def __init__(self, tokenizer, train=True, block_size=256, dp_chunks=1):
        hf.login(token=os.environ['HF_TOKEN'])
        # split = 'train' if train else 'test'
        val_size = 10000
        stream = load_dataset("HuggingFaceFW/fineweb", name="CC-MAIN-2024-10", split="train", streaming=True, trust_remote_code=True)
        if train:
            dataset = stream.skip(val_size) # first val_size examples
        else:
            dataset = stream.take(val_size) # first val_size examples
        super(FineWebDataset, self).__init__(dataset, tokenizer, block_size, text_key='text', dp_chunks=dp_chunks)

# Helper functions
def get_batch(loader_iter, batch_size):
    x_list, y_list = [], []
    try:
        for _ in range(batch_size):
            x, y = next(loader_iter)
            x_list.append(x)
            y_list.append(y)
        x = torch.stack(x_list)
        y = torch.stack(y_list)
        return x, y
    except StopIteration:
        return None, None

