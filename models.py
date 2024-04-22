
"""
This file contains the definition of encoders used in https://arxiv.org/pdf/1705.02364.pdf
"""

import numpy as np
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class AWEsEncoder(nn.Module):
    def __init__(self):
        """
        Average Word Embeddings Encoder
        """
        super().__init__()

    def forward(self, sentences, sentence_lengths):
        """
        Input:
        sentences - A tensor of shape [B, T, D]: batch size, maximum sentence length, and embedding dimension.
        sentence_lengths - A list of the unpadded lengths of the sentences in the batch.

        Output:
        sentence_representations - A tensor of shape [B, 300] where 300 is the glove embedding dimension.
        """
        sentence_representations = []
        # Loop through each sentence and its corresponding length
        for sentence, length in zip(sentences, sentence_lengths):
            sentence_mean = torch.mean(sentence[:length], dim=0)  # not sure if the torch is needed
            sentence_representations.append(sentence_mean)

        # Stack the list of tensors into a single tensor
        sentence_representations = torch.stack(sentence_representations)

        return sentence_representations



class LSTMEncoder(nn.Module):
    def __init__(self, input_size=300, hidden_size=2048, num_layers=1, dropout=0.5, bidirectional=False):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=bidirectional, dropout=dropout)
        # you cant actually init these here bc you don't have access to the sentences yet
        # self.hidden_state = torch.zeros((num_layers * (1+int(bidirectional)), sentences.shape[0], hidden_size), dtype=torch.float, device=self.device)
        # self.cell_state = torch.zeros((num_layers * (1+int(bidirectional)), sentences.shape[0], hidden_size,) dtype=torch.float, device=self.device)

                   
    def forward(self, sentences, sentence_lengths):
        """
        Input:
        sentences - A tensor of shape [B, T, D]: batch size, maximum sentence length, and embedding dimension.
        sentence_lengths - A list of the unpadded lengths of the sentences in the batch.

        Output:
        sentence_representations - A tensor of shape [B, 2048] where 2048 is the hidden size of the LSTM.
        """
        # initialize the hidden and cell states
        self.hidden_state = torch.zeros((self.num_layers * (1+int(self.bidirectional)), sentences.shape[0], self.hidden_size), dtype=torch.float).to(device)
        self.cell_state = torch.zeros((self.num_layers * (1+int(self.bidirectional)), sentences.shape[0], self.hidden_size), dtype=torch.float).to(device)
    
        # Convert sentence_lengths to a tensor
        sentence_lengths = torch.tensor(sentence_lengths).to(device)

        # sort the sentences and lengths
        sorted_lengths, sorted_indices = torch.sort(sentence_lengths, descending=True)
        # re-order the sentences based on the sorted_indices. dim=0 is the batch dim that indexes different sentences in the batch
        sorted_sentences = torch.index_select(sentences, dim=0, index=sorted_indices)
        
        # pack the sequences to a more compact version (without the padding) for more efficient processing
        packed_sentences = nn.utils.rnn.pack_padded_sequence(sorted_sentences, sorted_lengths, batch_first=True)
        
        # forward pass through LSTM; we don't care about the packed sequence outputs and the output cell states
        _, (hidden_states, _) = self.lstm(packed_sentences, (self.hidden_state, self.cell_state))
        # remove the dimension of size 1 at position 0 from the hidden_states tensor bc it's useless
        hidden_states = hidden_states.squeeze(dim=0)
       
        # unsort the hidden state
        ordered_indices_sequential_numbers, reconstruction_indices = torch.sort(sorted_indices)
        # hidden_states = hidden_states.index_select(1, unsorted_indices).squeeze(0)
        hidden_states = torch.index_select(hidden_states, dim=0, index=reconstruction_indices) 
        
        return hidden_states


# Note1: typically the shape of the hidden state is [num_layers * num_directions, batch, hidden_size]
# Note2: typically LSTM outpus in PyTorch don't have a leading singleton dimension unless it's
# a biLSTM returning both directions concatenated along one dimension.
# Note3: If the LSTM is unidirectional AND it has num_layers=1 THEN there's a leading singleton dimension
# in the hidden state that can be removed and that's why we're doing .squeeze(dim=0) on the hidden states
# Note4: batch_first=True means that the batch data is the first dimension
# Note5: resetting the hidden and cell states in the forward method is done to to clear any remnants from previous inputs or batches
# The LSTM must start fresh for each new batch of input data, maintaining the independence between different input sequences


class BiLSTMEncoder(nn.Module):
    def __init__(self, input_size=300, hidden_size=2048, num_layers=1, dropout=0.5, bidirectional=True, pooling=None):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.pooling = pooling

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=bidirectional, dropout=dropout)

    def forward(self, sentences, sentence_lengths):
        """
        Input:
        sentences - A tensor of shape [B, T, D]: batch size, maximum sentence length, and embedding dimension.
        sentence_lengths - A list of the unpadded lengths of the sentences in the batch.

        Output:
        sentence_representations - A tensor of shape [B, 2*2048] because the LSTM is bidirectional and hence the hidden state is concatenated along the last dimension
        """

        # initialize the hidden state and cell state
        self.hidden_state = torch.zeros((self.num_layers * (1+int(self.bidirectional)), sentences.shape[0], self.hidden_size), dtype=torch.float).to(device)
        self.cell_state = torch.zeros((self.num_layers * (1+int(self.bidirectional)), sentences.shape[0], self.hidden_size), dtype=torch.float).to(device)

        # Convert sentence_lengths to a tensor
        sentence_lengths = torch.tensor(sentence_lengths)

        # sort the embeddings on sentence length
        sorted_lengths, sorted_indices= torch.sort(sentence_lengths, descending=True)
        sorted_sentences = torch.index_select(sentences, dim=0, index=sorted_indices)


        # pack the embeddings
        packed_sentences = nn.utils.rnn.pack_padded_sequence(sorted_sentences, sorted_lengths, batch_first=True)

        # run through the model
        # hidden_states, _ = self.lstm(packed_sentences, (self.hidden_state, self.cell_state))
        output, (hidden_states, _) = self.lstm(packed_sentences, (self.hidden_state, self.cell_state))

        if self.pooling == 'max':
            # pad the output hidden states
            output, _ = nn.utils.rnn.pad_packed_sequence(hidden_states, batch_first=True)

            # apply max pooling
            max_sentences = []
            for sentence, length in zip(hidden_states, sentence_lengths):
                # again, take only the sentence without the padding
                sentence = sentence[:length]

                # max the sentence. The [0] is to get the max values and not the indices
                sentence = torch.max(sentence, dim=0)[0]

                # add to the list
                max_sentences.append(sentence)

            # stack the tensors
            hidden_states = torch.stack(max_sentences, dim=0)
        else:
            # get the forward and backward hidden states
            forward_state = hidden_states[0].to(sentences.device)
            backward_state = hidden_states[1].to(sentences.device)

            print(forward_state.shape)
            print(backward_state.shape)
            # remove the dimension of size 1 at position 0 from the hidden_states tensor bc it's useless
            # forward_state = forward_state.squeeze(dim=0)
            # backward_state = backward_state.squeeze(dim=0)

            hidden_states = torch.cat([forward_state, backward_state], dim=1)

        # unsort the embeddings
        ordered_indices_sequential_numbers, reconstruction_indices = torch.sort(sorted_indices)
        hidden_states = torch.index_select(hidden_states, dim=0, index=reconstruction_indices) 

        # return the sentence representations
        return hidden_states


'''
From  the paper:
For the classifier, we use a multi-layer perceptron with 1 hidden-layer of 512 hidden units.
'''
# THE CLASSIFIERS ARE WRITTEN UNDER THE ASSUMPTION THAT 
# THE INPUT SENTENCES ARE ALREADY PROCESSED BY THE ENCODER

class MLP_classifier(nn.Module):
    def __init__(self, input_dim=4*300):
        super().__init__()

        self.classifier = nn.Sequential(
                nn.Linear(input_dim, 512),
                nn.Linear(512, 512),
                nn.Linear(512, 3),
            )
        
    def forward(self, sentence_embeddings):
        """
        Inputs:
            sentence_embeddings - Tensor of sentence representations of shape varying shape. AWE is [B, 4*300]
        Outputs:
            predictions - Tensor of predictions (entailment, neutral, contradiction) of shape [B, 3]
        """
        output = self.classifier(sentence_embeddings) 

        return output


class MLP_classifier_nonlinearities(nn.Module):
    def __init__(self, input_dim=4*300):
        super().__init__()

        # initialize the classifier
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.Tanh(),
            nn.Linear(512, 512),
            nn.Tanh(),
            nn.Linear(512, 3)
        )

    def forward(self, sentence_embeddings):
        """
        Inputs:
            sentence_embeddings - Tensor of sentence representations of shape varying shape. AWE is [B, 4*300]
        Outputs:
            predictions - Tensor of predictions (entailment, neutral, contradiction) of shape [B, 3]
        """

        # pass sentence embeddings through model
        predictions = self.net(sentence_embeddings)

        # return the  predictions
        return predictions
