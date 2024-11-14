# models.py

import numpy as np
import collections
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

#####################
# MODELS FOR PART 1 #
#####################

class ConsonantVowelClassifier(object):
    def predict(self, context):
        """
        :param context:
        :return: 1 if vowel, 0 if consonant
        """
        raise Exception("Only implemented in subclasses")


class FrequencyBasedClassifier(ConsonantVowelClassifier):
    """
    Classifier based on the last letter before the space. If it has occurred with more consonants than vowels,
    classify as consonant, otherwise as vowel.
    """
    def __init__(self, consonant_counts, vowel_counts):
        self.consonant_counts = consonant_counts
        self.vowel_counts = vowel_counts

    def predict(self, context):
        # Look two back to find the letter before the space
        if self.consonant_counts[context[-1]] > self.vowel_counts[context[-1]]:
            return 0
        else:
            return 1


class RNNClassifier(ConsonantVowelClassifier):
    def predict(self, context):
        raise Exception("Implement me")


def train_frequency_based_classifier(cons_exs, vowel_exs):
    consonant_counts = collections.Counter()
    vowel_counts = collections.Counter()
    for ex in cons_exs:
        consonant_counts[ex[-1]] += 1
    for ex in vowel_exs:
        vowel_counts[ex[-1]] += 1
    return FrequencyBasedClassifier(consonant_counts, vowel_counts)


def train_rnn_classifier(args, train_cons_exs, train_vowel_exs, dev_cons_exs, dev_vowel_exs, vocab_index):
    """
    :param args: command-line args, passed through here for your convenience
    :param train_cons_exs: list of strings followed by consonants
    :param train_vowel_exs: list of strings followed by vowels
    :param dev_cons_exs: list of strings followed by consonants
    :param dev_vowel_exs: list of strings followed by vowels
    :param vocab_index: an Indexer of the character vocabulary (27 characters)
    :return: an RNNClassifier instance trained on the given data
    """
    raise Exception("Implement me")


#####################
# MODELS FOR PART 2 #
#####################


class LanguageModel(object):

    def get_log_prob_single(self, next_char, context):
        """
        Scores one character following the given context. That is, returns
        log P(next_char | context)
        The log should be base e
        :param next_char:
        :param context: a single character to score
        :return:
        """
        raise Exception("Only implemented in subclasses")


    def get_log_prob_sequence(self, next_chars, context):
        """
        Scores a bunch of characters following context. That is, returns
        log P(nc1, nc2, nc3, ... | context) = log P(nc1 | context) + log P(nc2 | context, nc1), ...
        The log should be base e
        :param next_chars:
        :param context:
        :return:
        """
        raise Exception("Only implemented in subclasses")


class UniformLanguageModel(LanguageModel):
    def __init__(self, voc_size):
        self.voc_size = voc_size

    def get_log_prob_single(self, next_char, context):
        return np.log(1.0/self.voc_size)

    def get_log_prob_sequence(self, next_chars, context):
        return np.log(1.0/self.voc_size) * len(next_chars)


# inherits the language model
class RNNLanguageModel(LanguageModel):
    
    # initilaize the model with the embedding layer
    def __init__(self, vocab_size, embedding_dim, hidden_dim, vocab_index):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.RNN(embedding_dim, hidden_dim, batch_first=True)
        self.output_layer = nn.Linear(hidden_dim, vocab_size)
        self.vocab_index = vocab_index

    # takes an input sequnce
    def forward(self, input_seq):
        embeds = self.embedding(input_seq)
        # process it through rnn
        rnn_out, _ = self.rnn(embeds)
        # applying a linear layer to generate logits (unnormalized)
        logits = self.output_layer(rnn_out)
        return logits

    # calculates the log prob of a single character after a given context
    def get_log_prob_single(self, next_char, context):
        context_idx = torch.tensor([self.vocab_index.index_of(c) for c in context], dtype=torch.long).unsqueeze(0)
        output = self.forward(context_idx)
        next_char_idx = self.vocab_index.index_of(next_char)
        log_prob = F.log_softmax(output[0, -1], dim=-1)[next_char_idx]
        return log_prob.item()

    # itereates and calls the above method and summing the log prob (total log of the sequence)
    def get_log_prob_sequence(self, next_chars, context):
        log_prob_sum = 0.0
        full_context = context + next_chars
        for i in range(len(next_chars)):
            log_prob_sum += self.get_log_prob_single(full_context[len(context) + i], full_context[:len(context) + i])
        return log_prob_sum

# class RNNLanguageModel(LanguageModel):
#     def __init__(self, model_emb, model_dec, vocab_index):
#         self.model_emb = model_emb
#         self.model_dec = model_dec
#         self.vocab_index = vocab_index

#     def get_log_prob_single(self, next_char, context):
#         raise Exception("Implement me")

#     def get_log_prob_sequence(self, next_chars, context):
#         raise Exception("Implement me")


# def train_lm(args, train_text, dev_text, vocab_index):
#     """
#     :param args: command-line args, passed through here for your convenience
#     :param train_text: train text as a sequence of characters
#     :param dev_text: dev texts as a sequence of characters
#     :param vocab_index: an Indexer of the character vocabulary (27 characters)
#     :return: an RNNLanguageModel instance trained on the given data
#     """
#     raise Exception("Implement me")



def train_lm(args, train_text, dev_text, vocab_index):
    """
    Trains the RNN language model on the provided training text.
    :param args: command-line arguments
    :param train_text: Training text as a sequence of characters
    :param dev_text: Development text as a sequence of characters
    :param vocab_index: Indexer object for vocabulary
    :return: Trained RNNLanguageModel instance
    """
    vocab_size = len(vocab_index)
    embedding_dim = 32      # dimension of embedding layer
    hidden_dim = 64         # no of units in the rnn hidden layer
    model = RNNLanguageModel(vocab_size, embedding_dim, hidden_dim, vocab_index)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    # Preping the training data
    train_indices = [vocab_index.index_of(c) for c in train_text]
    inputs = torch.tensor(train_indices[:-1], dtype=torch.long)
    targets = torch.tensor(train_indices[1:], dtype=torch.long)

    num_epochs = 5
    batch_size = 32
    dataset = TensorDataset(inputs, targets)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for batch_inputs, batch_targets in dataloader:
            optimizer.zero_grad()
            output = model(batch_inputs.unsqueeze(0))
            loss = criterion(output.view(-1, vocab_size), batch_targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss / len(dataloader)}")

    # evaluate on the development set
    model.eval()
    # avg log prob gives a measurement of the model's confidence in its prediction
    dev_log_prob = model.get_log_prob_sequence(dev_text, context=" ")
    dev_perplexity = np.exp(-dev_log_prob / len(dev_text))
    print(f"Development set perplexity: {dev_perplexity}")

    return model

