import os
import numpy as np
import torch
from torchtext import data, datasets
from nltk.tokenize import word_tokenize
import nltk
nltk.download('punkt')
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

from datasets import load_dataset, Dataset

from torchtext.vocab import Vocab
from collections import Counter

GLOVE_PATH = "GloVe_embeds.nosync/glove.840B.300d.txt"

# "/Users/zoetzikra/Documents/2023-2024/ATCS/practical_1/GloVe_embeds/glove.840B.300d.txt"

# the vectors from glove come as a string of space separated numbers
# np.fromstring converts this into a numpy array
def load_glove_vectors(glove_path, w2i):
     word_vectors = {}
     with open(glove_path) as f:
          for line in f:
               word, vec = line.split(' ', 1)
               if word in w2i:
                    word_vectors[word] = np.fromstring(vec, sep=' ')
     print('Found {0} words with word vectors, out of {1} words'.format(len(word_vectors), len(w2i)))
     return word_vectors


def build_word_dict(sentences):
     word_counter = Counter()
     for sentence_dict in sentences:
          for sentence in [sentence_dict['premise'], sentence_dict['hypothesis']]:
               for word in sentence:
                    word_counter[word] += 1

     sorted_words = sorted(word_counter.items(), key=lambda x: -x[1])
     w2i = {w: i for i, (w, _) in enumerate(sorted_words)}

     return word_counter, w2i


def build_vocab_embeddings(sentences, glove_path):
     word_counter, w2i = build_word_dict(sentences)
     glove_vectors = load_glove_vectors(glove_path, w2i)

     vocab = Vocab(word_counter, specials=['<unk>', '<pad>'])

     # Create a dictionary that maps indices to vectors
     # index_to_vector = {vocab.stoi[word]: vector for word, vector in glove_vectors.items() if word in vocab.stoi}
     # index_to_vector = {i: glove_vectors.get(word, np.zeros(300)) for word, i in vocab.stoi.items()}
     index_to_vector = {i: torch.tensor(glove_vectors.get(word, np.zeros(300))) for word, i in vocab.stoi.items()}

     vocab.set_vectors(stoi=vocab.stoi, vectors=index_to_vector, dim=300)
     print('Vocab size : %s' % len(vocab))
     return vocab, index_to_vector


# NOTE: This is not used in the end
# This function returns a tensor of the batch of sentences and a list of the lengths of the sentences in the batch
# sentences in decreasing order of lengths (bsize, max_len, word_dim)
def get_batch(batch, vocab_vectors):
    # calculate the length of each sentence in the batch and store it in array along with the max length
    lengths = np.array([len(x) for x in batch])
    max_len = np.max(lengths)
    # initialize the embedding tensor with zeros. 
    # The shape of the tensor is (max_len, bsize, 300)
    # 300 is because the glove vectors are of length 300
    embed = np.zeros((max_len, len(batch), 300))

    # for each word in each sentence, fill the embedding tensor with the glove vector for that word
    for i in range(len(batch)):
        for j in range(len(batch[i])):
               word_index = torch.tensor(batch[i][j], dtype=torch.long)  # Convert word index to LongTensor
               embed[j, i, :] = vocab_vectors(word_index).numpy()  # Use forward method to get word embedding

    return torch.from_numpy(embed).float(), lengths



def preprocess_dataset(dataset):
     # use the nltk tokenizer
     dataset = dataset.map(lambda x: {'premise': word_tokenize(x['premise']), 'hypothesis': word_tokenize(x['hypothesis'])})
     # lowercase the tokens
     dataset = dataset.map(lambda x: {'premise': [word.lower() for word in x['premise']], 'hypothesis': [word.lower() for word in x['hypothesis']]})
     return dataset


def collate_fn(batch, vocab):
     premises = [example['premise'] for example in batch]
     hypotheses = [example['hypothesis'] for example in batch]
     labels = [example['label'] for example in batch]
     
     # Convert words to their indices in the vocabulary
     premises = [torch.tensor([vocab.stoi[word] for word in p if word in vocab.stoi]) for p in premises]
     hypotheses = [torch.tensor([vocab.stoi[word] for word in h if word in vocab.stoi]) for h in hypotheses]

     premises = [p for i, p in enumerate(premises) if labels[i] != -1]
     hypotheses = [h for i, h in enumerate(hypotheses) if labels[i] != -1]
     labels = [l for l in labels if l != -1]
     # print("labels processed:", labels)

     # Pad sequences BUT save and return the original lengths of the sequences
     original_lengths_p = [len(p) for p in premises]
     original_lengths_h = [len(h) for h in hypotheses]
     combined = premises + hypotheses
     combined_padded = pad_sequence(combined, batch_first=True)
     premises_padded = combined_padded[:len(premises)]
     hypotheses_padded = combined_padded[len(premises):]

     return (premises_padded, hypotheses_padded, original_lengths_p, original_lengths_h), labels



# function that loads the SNLI dataset
def load_snli(batch_size=64, development=False):
     """
     Inputs:
          batch_size - Size of the batches. Default is 64
          development - Whether to use development dataset. Default is False

     Outputs:
          vocab - GloVe embedding vocabulary from the alignment
          train_iter - BucketIterator of training batches
          dev_iter - BucketIterator of validation/development batches
          test_iter - BucketIterator of test batches
     """

     # load the SNLI dataset
     dataset = load_dataset('snli')
     train_dataset = dataset['train']
     dev_dataset = dataset['validation']
     test_dataset = dataset['test']
     if development: #we want to keep the dataset small for development.
          # we tak ethe first 64*100 examples from the training dataset, 64*50 from the dev dataset and 64*50 from the test dataset
          train_dataset = train_dataset.select(range(64*400))
          dev_dataset = dev_dataset.select(range(64*100))
          test_dataset = test_dataset.select(range(64*100))
     
     # preprocess the datasets
     train_dataset = preprocess_dataset(train_dataset)
     dev_dataset = preprocess_dataset(dev_dataset)
     test_dataset = preprocess_dataset(test_dataset)
     

     # build the vocabulary of the dataset
     sentences = [sentence for dataset in [train_dataset, dev_dataset, test_dataset] for sentence in dataset]
     vocab, index_to_vector = build_vocab_embeddings(sentences, GLOVE_PATH)
     

     # create batch iterators for the datasets
     train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=lambda b: collate_fn(b, vocab))
     dev_loader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False, collate_fn=lambda b: collate_fn(b, vocab))
     test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=lambda b: collate_fn(b, vocab))

     return vocab, train_loader, dev_loader, test_loader
