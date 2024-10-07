# models.py

import numpy as np
import torch.nn as nn
import random
import torch
from torch import optim
from transformer import PositionalEncoding

class LanguageModel(object):

    def get_next_char_log_probs(self, context) -> np.ndarray:
        """
        Returns a log probability distribution over the next characters given a context.
        The log should be base e

        NOTE: You should make sure you call model.eval() to determinize inference here (turns off dropout
        layers in TransformerEncoder).
        :param context: the string context that the LM conditions on
        :return: A numpy vector log P(y | context) where y ranges over the output vocabulary.
        """
        raise Exception("Only implemented in subclasses")


    def get_log_prob_sequence(self, next_chars, context) -> float:
        """
        Scores a bunch of characters following context. That is, returns
        log P(nc1, nc2, nc3, ... | context) = log P(nc1 | context) + log P(nc2 | context, nc1), ...
        The log should be base e

        NOTE: You should make sure you call model.eval() to determinize inference here (turns off dropout
        layers in TransformerEncoder).
        :param next_chars:
        :param context:
        :return: The float probability
        """
        raise Exception("Only implemented in subclasses")


class UniformLanguageModel(LanguageModel):
    def __init__(self, voc_size):
        self.voc_size = voc_size

    def get_next_char_log_probs(self, context):
        return np.ones([self.voc_size]) * np.log(1.0/self.voc_size)

    def get_log_prob_sequence(self, next_chars, context):
        return np.log(1.0/self.voc_size) * len(next_chars)


class NeuralLanguageModel(LanguageModel):
    def __init__(self, transformer, indexer):
        self.transformer = transformer
        self.indexer = indexer
        self.log_softmax = nn.LogSoftmax(dim=-1)

    def get_next_char_log_probs(self, context):
        with torch.no_grad():
            characters = [self.indexer.index_of(char) for char in context]
            while len(characters) < 20:
                characters = [self.indexer.index_of(' '), *characters]
            output = self.transformer(torch.LongTensor(characters))
            output = self.log_softmax(output)
            output = output[:][-1]
            return output.detach().numpy()
        
    def get_log_prob_sequence(self, next_chars, context):
        log_prob = 0.0
        curr_context = context
        for char in next_chars:
            output = self.get_next_char_log_probs(curr_context)
            log_prob += output[self.indexer.index_of(char)]
            if len(curr_context) == 20:
                curr_context = curr_context[1:] + char
            else:
                curr_context += char
        return log_prob

class TransformerModel(nn.Module):
    def __init__(self, d_model, d_internal, num_layers, nhead, vocab_size, num_classes, batched=False):
        super().__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=d_internal, batch_first=batched, dropout=0.01)
        self.encoder = nn.TransformerEncoder(encoder_layer=self.encoder_layer, num_layers=num_layers)
        self.embeddings = nn.Embedding(vocab_size, d_model)
        self.position_encoder = PositionalEncoding(d_model=d_model, num_positions=20, batched=batched)
        self.linear_layer = nn.Linear(d_model, num_classes)
        self.mask = torch.triu(torch.ones(20, 20) * float('-inf'), diagonal=1)
    
    def forward(self, input):
        output = self.embeddings(input)
        output = self.position_encoder(output)
        output = self.encoder(src=output, mask=self.mask)
        output = self.linear_layer(output)
        return output
        

def train_lm(args, train_text, dev_text, vocab_index):
    """
    :param args: command-line args, passed through here for your convenience
    :param train_text: train text as a sequence of characters
    :param dev_text: dev text as a sequence of characters
    :param vocab_index: an Indexer of the character vocabulary (27 characters)
    :return: a NeuralLanguageModel instance trained on the given data
    """
    tokenized_text = [vocab_index.index_of(char) for char in train_text]
    text_chunks = []
    chunk = []
    for char in tokenized_text:
        if len(chunk) < 20:
            chunk.append(char)
        else:
            text_chunks.append(torch.LongTensor(chunk))
            chunk = []
    transformer_model = TransformerModel(d_model=32, d_internal=64, num_layers=2, nhead=2, vocab_size=vocab_index.__len__(), num_classes=27, batched=False)

    optimizer = optim.Adam(transformer_model.parameters(), lr=1e-3)
    loss_fct = nn.NLLLoss()
    log_softmax = nn.LogSoftmax(dim=1)
    
    num_epochs = 10
    transformer_model.zero_grad()
    transformer_model.train()
    for _ in range(0, num_epochs):
        total_loss = 0.0
        random.shuffle(text_chunks)
        for chunk in text_chunks:
            optimizer.zero_grad()
            output = transformer_model.forward(torch.cat([torch.tensor([vocab_index.index_of(" ")]), chunk[:-1]]))
            output = log_softmax(output)
            loss = loss_fct(output, chunk)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
    transformer_model.eval()
    return NeuralLanguageModel(transformer=transformer_model, indexer=vocab_index)

