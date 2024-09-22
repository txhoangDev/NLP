# models.py

import torch
import torch.nn as nn
from torch import optim
import numpy as np
import random
import json
from nltk import edit_distance
from sentiment_data import *


class SentimentClassifier(object):
    """
    Sentiment classifier base type
    """

    def predict(self, ex_words: List[str], has_typos: bool) -> int:
        """
        Makes a prediction on the given sentence
        :param ex_words: words to predict on
        :param has_typos: True if we are evaluating on data that potentially has typos, False otherwise. If you do
        spelling correction, this parameter allows you to only use your method for the appropriate dev eval in Q3
        and not otherwise
        :return: 0 or 1 with the label
        """
        raise Exception("Don't call me, call my subclasses")

    def predict_all(self, all_ex_words: List[List[str]], has_typos: bool) -> List[int]:
        """
        You can leave this method with its default implementation, or you can override it to a batched version of
        prediction if you'd like. Since testing only happens once, this is less critical to optimize than training
        for the purposes of this assignment.
        :param all_ex_words: A list of all exs to do prediction on
        :param has_typos: True if we are evaluating on data that potentially has typos, False otherwise.
        :return:
        """
        return [self.predict(ex_words, has_typos) for ex_words in all_ex_words]


class TrivialSentimentClassifier(SentimentClassifier):
    def predict(self, ex_words: List[str], has_typos: bool) -> int:
        """
        :param ex:
        :return: 1, always predicts positive class
        """
        return 1


class NeuralSentimentClassifier(SentimentClassifier):
    """
    Implement your NeuralSentimentClassifier here. This should wrap an instance of the network with learned weights
    along with everything needed to run it on new data (word embeddings, etc.). You will need to implement the predict
    method and you can optionally override predict_all if you want to use batching at inference time (not necessary,
    but may make things faster!)
    """
    def __init__(self, dan_model, word_embeddings, optimizer):
        self.model = dan_model
        self.model.eval()
        self.word_embeddings = word_embeddings
        self.optimizer = optimizer
        
    def predict(self, ex_words: List[str], has_typos: bool) -> int:
        # embedding words
        word_indices = np.array([self.word_embeddings.get_embedding(word) for word in ex_words])
        average = np.mean(word_indices, axis=0)
        with torch.no_grad():
            output = self.model(torch.tensor(average))
            return output.argmax(dim=0).item()
        
    def predict_all(self, all_ex_words: List[List[str]], has_typos: bool) -> List[int]:
        return [self.predict(word_example, has_typos) for word_example in all_ex_words]

class DAN(nn.Module):
    def __init__(self, embedding_size, hidden_layer_size, output_size):
        """
        Constructs the computation graph by instantiating the various layers and initializing weights.

        :param inp: size of input (integer)
        :param hid: size of hidden layer(integer)
        :param out: size of output (integer), which should be the number of classes
        """
        super(DAN, self).__init__()
        
        self.non_linear = nn.ReLU()
        
        # hidden layer
        self.hidden_layers = nn.ModuleList()
        for _ in range(hidden_layer_size):
            self.hidden_layers.append(nn.Linear(embedding_size, hidden_layer_size, dtype=torch.float64))
        
        # weights
        self.weight = nn.Linear(hidden_layer_size, output_size, dtype=torch.float64)
        
        #softmax
        self.log_softmax = nn.LogSoftmax(dim=0)

    def forward(self, x):
        # run non-linearity
        output = self.non_linear(x)
        
        # run average through each layer
        for hidden_layer in self.hidden_layers:
            output = hidden_layer(output)
            
        # run weights
        output = self.weight(output)
        
        #get log probability 
        return self.log_softmax(output)

class PrefixEmbeddings(WordEmbeddings):
    """
    Wraps an Indexer and a list of 1-D numpy arrays where each position in the list is the vector for the corresponding
    word in the indexer. The 0 vector is returned if an unknown word is queried.
    """
    def __init__(self, word_indexer, vectors):
        self.word_indexer = word_indexer
        self.vectors = vectors

    def get_embedding(self, word):
        """
        Returns the embedding for a given word
        :param word: The word to look up
        :return: The UNK vector if the word is not in the Indexer or the vector otherwise
        """
        prefix_idx = self.word_indexer.index_of(word[:3])
        if prefix_idx != -1:
            return self.vectors[prefix_idx]
        else:
            return self.vectors[self.word_indexer.index_of("UNK")]

def train_deep_averaging_network(args, train_exs: List[SentimentExample], dev_exs: List[SentimentExample],
                                 word_embeddings: WordEmbeddings, train_model_for_typo_setting: bool) -> NeuralSentimentClassifier:
    """
    :param args: Command-line args so you can access them here
    :param train_exs: training examples
    :param dev_exs: development set, in case you wish to evaluate your model during training
    :param word_embeddings: set of loaded word embeddings
    :param train_model_for_typo_setting: True if we should train the model for the typo setting, False otherwise
    :return: A trained NeuralSentimentClassifier model. Note: you can create an additional subclass of SentimentClassifier
    and return an instance of that for the typo setting if you want; you're allowed to return two different model types
    for the two settings.
    """
    model = DAN(word_embeddings.get_embedding_length(), 1, 2)
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), 0.01)
    
    if train_deep_averaging_network:
        prefix_indexer = Indexer()
        prefix_vectors = []
        words = []
        
        for example in train_exs:
            for word in example.words:
                prefix = word[:3]
                prefix_idx = prefix_indexer.index_of(prefix)
                if prefix_idx == -1:
                    prefix_idx = prefix_indexer.add_and_get_index(prefix)
                    prefix_vectors.append(np.zeros_like(word_embeddings.get_embedding(word)))
                prefix_vectors[prefix_idx] += word_embeddings.get_embedding(word)
                words.append(word)
        for i in range(len(prefix_vectors)):
            prefix_vectors[i] /= len([word_embeddings.get_embedding(word) for word in words if word.startswith(prefix_indexer.get_object(i))])

        # Create PrefixEmbeddings object
        prefix_embeddings = PrefixEmbeddings(prefix_indexer, prefix_vectors)
    
    for _ in range(args.num_epochs):
        for example in train_exs:
            # zero out gradients
            optimizer.zero_grad()
            
            # train model
            if train_deep_averaging_network:
                word_indices = np.array([prefix_embeddings.get_embedding(word) for word in example.words])
            else:
                word_indices = np.array([word_embeddings.get_embedding(word) for word in example.words])
            average = np.mean(word_indices, axis=0)
            output = model.forward(torch.tensor(average))
            loss = criterion(output, torch.tensor(example.label))
            loss.backward()
            optimizer.step()
        random.shuffle(train_exs)
    
    if train_deep_averaging_network:
        return NeuralSentimentClassifier(model, prefix_embeddings, optimizer)
    else:
        return NeuralSentimentClassifier(model, word_embeddings, optimizer)
