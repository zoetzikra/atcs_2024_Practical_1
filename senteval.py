
import sys
import io
import numpy as np
import logging

import torch
from collections import Counter
from torchtext.vocab import Vocab


from main import *

# Set PATHs
SENTEVAL_PATH = './SentEval'
DATA_PATH = './SentEval/data'
VEC_PATH = './SentEval/pretrained/glove.840B.300d.txt'

# import SentEval
sys.path.insert(0, PATH_TO_SENTEVAL)
import senteval

# added global variables for the model and embedding dimension
MODEL = None
EMBED_DIM = 300

# differences to Pre-process data: 
# - it assigns a unique index to each word
# - sorts the words by their frequency
# - returns a list:  with the words sorted in descending order of freq 
#        and a dictionary: with the words as keys and their index as values
# To get the index for a word, you can look it up in word2id, and 
# to get the word for an index, you can index into id2word
def build_word_dict(sentences, threshold=0):
    word_counter = Counter()
    for sentence_dict in sentences:
        for sentence in [sentence_dict['premise'], sentence_dict['hypothesis']]:
            for word in sentence.split():
                word_counter[word] += 1

    # Apply frequency threshold
    if threshold > 0:
        for word in list(word_counter):
            if word_counter[word] < threshold:
                del word_counter[word]

    # Add special tokens
    word_counter['<s>'] = 1e9 + 4
    word_counter['</s>'] = 1e9 + 3
    word_counter['<p>'] = 1e9 + 2

    # Sort words by frequency
    sorted_words = sorted(word_counter.items(), key=lambda x: -x[1])

    # Create word-to-id and id-to-word dictionaries
    id2word = [w for w, _ in sorted_words]
    word2id = {w: i for i, w in enumerate(id2word)}

    return id2word, word2id


# difference to the preprocess_data function: here the word vectors are loaded only for the words that are in the word2id dictionary.
def load_glove_vectors(glove_path, word2id):
    word_vectors = {}
    with open(glove_path) as f:
        for line in f:
            word, vec = line.split(' ', 1)
            if word in word2id:
                word_vectors[word] = np.fromstring(vec, sep=' ')
    print('Found {0} words with word vectors, out of {1} words'.format(len(word_vectors), len(word2id)))
    return word_vectors


# do an equivalent to the above but using build_word_dict and load_glove_vectors from this file
def build_vocab_embeddings(sentences, glove_path, threshold=0):
    # Get unique words and their ids
    _, word2id = build_word_dict(sentences, threshold=threshold)
    word_vectors = load_glove_vectors(glove_path, word2id)

    return word_vectors


def get_batch(batch, vocab_vectors):
    lengths = np.array([len(x) for x in batch])
    max_len = np.max(lengths)
    embed = np.zeros((max_len, len(batch), 300))
    # for each word in each sentence, fill the embedding tensor with the glove vector for that word
    for i in range(len(batch)):
        for j in range(len(batch[i])):
            embed[j, i, :] = vocab_vectors[batch[i][j]]

    return torch.from_numpy(embed).float(), lengths

def get_batch(batch, vocab_vectors, model):
    # Replace empty sentences with a sentence containing a single period
    batch = [sentence if sentence != [] else ['.'] for sentence in batch]
    embeddings = []

    for sentence in batch:
        sentvec = []
        for word in sentence:
            # Check if the word is in vocab_vectors
            if word in vocab_vectors:
                # Convert the numpy array to a PyTorch tensor and append it to sentvec
                sentvec.append(torch.tensor(vocab_vectors[word]))
        # If a sentence does not have any word vectors, append a zero vector to sentvec
        if not sentvec:
            vec = torch.zeros((1, 300))
            sentvec.append(vec)
        # Stack all the word vectors of a sentence into a tensor
        sentvec = torch.stack(sentvec, dim=0)
        embeddings.append(sentvec)

    # Pad embeddings into a tensor
    sentence_lengths = torch.tensor([x.shape[0] for x in embeddings])
    embeddings = torch.nn.utils.rnn.pad_sequence(embeddings, padding_value=0.0, batch_first=True)

    # Pass through the model
    embeddings = model.encoder(embeddings.float(), sentence_lengths)

    # Cast back to numpy
    embeddings = embeddings.cpu().detach().numpy()

    return embeddings, sentence_lengths




# Set params for SentEval
params_senteval = {'task_path': DATA_PATH, 'usepytorch': True, 'kfold': 10}
params_senteval['classifier'] = {'nhid': 0, 'optim': 'adam', 'batch_size': 64,
                                 'tenacity': 5, 'epoch_size': 4}

# Set up logger
logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.DEBUG)



if __name__ == "__main__":

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument('--model', default='AWE', type=str,
                        help='What model to use. Default is AWE',
                        choices=['AWE', 'UniLSTM', 'BiLSTM', 'BiLSTMMax'])
    args = parser.parse_args()


    # set the global variables
    if (args.model == 'AWE'):
        MODEL = NLIModule.load_from_checkpoint('pl_logs/lightning_logs/awe/checkpoints/epoch=10.ckpt')
        EMBED_DIM = 300
    elif (args.model == 'UniLSTM'):
        MODEL = NLIModule.load_from_checkpoint('pl_logs/lightning_logs/unilstm/checkpoints/epoch=11.ckpt')
        EMBED_DIM = 2048
    elif (args.model == 'BiLSTM'):
        MODEL = NLIModule.load_from_checkpoint('pl_logs/lightning_logs/bilstm/checkpoints/epoch=13.ckpt')
        EMBED_DIM = 2*2048
    else:
        MODEL = NLIModule.load_from_checkpoint('pl_logs/lightning_logs/bilstmmax/checkpoints/epoch=7.ckpt')
        EMBED_DIM = 2*2048

    # run the senteval
    se = senteval.engine.SE(params_senteval, get_batch, build_vocab_embeddings)
    transfer_tasks = ['MR', 'CR', 'SUBJ', 'MPQA', 'SST2', 'TREC', 'MRPC',
                      'SICKRelatedness', 'SICKEntailment', 'STS14']
    results = se.eval(transfer_tasks)

    # save the results
    torch.save(results, args.model + "SentEvalResults.pt")

    # print the results
    print(results)
