import numpy as np
import collections
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
from torch.utils.data import DataLoader, TensorDataset

# For testing only
import matplotlib.pyplot as plt

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


class RNNClassifier(ConsonantVowelClassifier, nn.Module):
    def __init__(self, vocab_index, embedding_size, hidden_size, output_size):
        # Initialize both parent classes
        ConsonantVowelClassifier.__init__(self)
        nn.Module.__init__(self)

        self.vocab_index = vocab_index
        self.embedding = nn.Embedding(len(vocab_index), embedding_size)
        self.rnn = nn.LSTM(embedding_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, input_sequence):
        # Embedding and LSTM
        seq_embedding = self.embedding(input_sequence)  # [batch_size, seq_len, embedding_size]
        _, (hidden, _) = self.rnn(seq_embedding)  # hidden: [1, batch_size, hidden_size]

        logits = self.fc(hidden.squeeze(0))  # [batch_size, output_size]
        return F.log_softmax(logits, dim=1)

    def predict(self, input_sequence):
        with torch.no_grad():
            input_indices = [self.vocab_index.index_of(char) for char in input_sequence]
            input_tensor = torch.tensor(input_indices, dtype=torch.long).unsqueeze(0)  # Add batch dimension
            output = self.forward(input_tensor)
            return torch.argmax(output, dim=1).item()


def train_frequency_based_classifier(cons_exs, vowel_exs):
    consonant_counts = collections.Counter()
    vowel_counts = collections.Counter()
    for ex in cons_exs:
        consonant_counts[ex[-1]] += 1
    for ex in vowel_exs:
        vowel_counts[ex[-1]] += 1
    return FrequencyBasedClassifier(consonant_counts, vowel_counts)


def train_rnn_classifier(args, train_cons_exs, train_vowel_exs, dev_cons_exs, dev_vowel_exs, vocab_index):
    embedding_size = 40
    hidden_size = 25
    output_size = 2
    batch_size = 32
    learning_rate = 0.001
    epochs = 15

    print("Embedding Size:", embedding_size)
    print("Hidden Size:", hidden_size)

    model = RNNClassifier(vocab_index, embedding_size, hidden_size, output_size)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Prepare training and dev data as tensors
    train_data = [(torch.tensor([vocab_index.index_of(char) for char in ex], dtype=torch.long), torch.tensor(0))
                  for ex in train_cons_exs] + \
                 [(torch.tensor([vocab_index.index_of(char) for char in ex], dtype=torch.long), torch.tensor(1))
                  for ex in train_vowel_exs]

    dev_data = [(torch.tensor([vocab_index.index_of(char) for char in ex], dtype=torch.long), torch.tensor(0))
                for ex in dev_cons_exs] + \
               [(torch.tensor([vocab_index.index_of(char) for char in ex], dtype=torch.long), torch.tensor(1))
                for ex in dev_vowel_exs]

    # Function to create mini-batches
    def create_batches(data, batch_size):
        random.shuffle(data)  # Shuffle data for each epoch
        batches = [data[i:i + batch_size] for i in range(0, len(data), batch_size)]
        return batches

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct_train_predictions = 0

        # Create mini-batches for training
        train_batches = create_batches(train_data, batch_size)

        for batch in train_batches:
            input_sequences = [item[0] for item in batch]  # Extract sequences
            labels = torch.stack([item[1] for item in batch])  # Extract labels and stack them
            input_sequences = torch.nn.utils.rnn.pad_sequence(input_sequences, batch_first=True, padding_value=0)  # Pad sequences

            optimizer.zero_grad()
            output = model(input_sequences)
            loss = criterion(output, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient clipping
            optimizer.step()

            total_loss += loss.item()
            correct_train_predictions += (torch.argmax(output, dim=1) == labels).sum().item()

        avg_train_loss = total_loss / len(train_batches)
        train_accuracy = correct_train_predictions / len(train_data) * 100

        # Evaluate on dev set
        model.eval()
        correct_dev_predictions = 0
        dev_batches = create_batches(dev_data, batch_size)

        with torch.no_grad():
            for batch in dev_batches:
                input_sequences = [item[0] for item in batch]
                labels = torch.stack([item[1] for item in batch])
                input_sequences = torch.nn.utils.rnn.pad_sequence(input_sequences, batch_first=True, padding_value=0)  # Pad sequences

                output = model(input_sequences)
                correct_dev_predictions += (torch.argmax(output, dim=1) == labels).sum().item()

        dev_accuracy = correct_dev_predictions / len(dev_data) * 100
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_train_loss:.4f}, "
              f"Train Accuracy: {train_accuracy:.2f}%, Dev Accuracy: {dev_accuracy:.2f}%")

    return model


#####################
# MODELS FOR PART 2 #
#####################

class LanguageModel(object):
    def get_log_prob_single(self, next_char, context):
        raise Exception("Only implemented in subclasses")

    def get_log_prob_sequence(self, next_chars, context):
        raise Exception("Only implemented in subclasses")


class UniformLanguageModel(LanguageModel):
    def __init__(self, voc_size):
        self.voc_size = voc_size

    def get_log_prob_single(self, next_char, context):
        return np.log(1.0 / self.voc_size)

    def get_log_prob_sequence(self, next_chars, context):
        return np.log(1.0 / self.voc_size) * len(next_chars)


class RNNLanguageModel(LanguageModel, nn.Module): 
    def __init__(self, vocab_size, embedding_dim, hidden_dim, vocab_index):
        super(RNNLanguageModel, self).__init__()  # Initialize nn.Module
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.RNN(embedding_dim, hidden_dim, batch_first=True)
        self.output_layer = nn.Linear(hidden_dim, vocab_size)
        self.vocab_index = vocab_index

    def forward(self, input_seq):
        embeds = self.embedding(input_seq)
        rnn_out, _ = self.rnn(embeds)
        logits = self.output_layer(rnn_out)
        return logits

    def get_log_prob_single(self, next_char, context):
        context_idx = torch.tensor([self.vocab_index.index_of(c) for c in context], dtype=torch.long).unsqueeze(0)
        output = self.forward(context_idx)
        next_char_idx = self.vocab_index.index_of(next_char)
        log_prob = F.log_softmax(output[0, -1], dim=-1)[next_char_idx]
        return log_prob.item()

    def get_log_prob_sequence(self, next_chars, context):
        log_prob_sum = 0.0
        full_context = context + next_chars
        for i in range(len(next_chars)):
            log_prob_sum += self.get_log_prob_single(full_context[len(context) + i], full_context[:len(context) + i])
        return log_prob_sum


def train_lm(args, train_text, dev_text, vocab_index):
    vocab_size = len(vocab_index)
    embedding_dim = 32
    hidden_dim = 64
    model = RNNLanguageModel(vocab_size, embedding_dim, hidden_dim, vocab_index)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

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
            output = model(batch_inputs.unsqueeze(0))  # Add batch dimension
            loss = criterion(output.view(-1, vocab_size), batch_targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss / len(dataloader)}")

    model.eval()
    dev_log_prob = model.get_log_prob_sequence(dev_text, context=" ")
    dev_perplexity = np.exp(-dev_log_prob / len(dev_text))
    print(f"Development set perplexity: {dev_perplexity}")

    return model


