import numpy as np
import collections
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
from torch.utils.data import DataLoader, TensorDataset

# For testing purposes only
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
        raise Exception("Only implemented in subclasses")

    def get_log_prob_sequence(self, next_chars, context):
        raise Exception("Only implemented in subclasses")


# Uniform language model assigns equal probability to all characters in the vocabulary
class UniformLanguageModel(LanguageModel):
    # Initialize the language model with the vocabulary size
    def __init__(self, voc_size):
        self.voc_size = voc_size

    # Return the log probability of a single character given the context
    def get_log_prob_single(self, next_char, context):
        return np.log(1.0 / self.voc_size)

    # Return the log probability of a sequence of characters given the context
    def get_log_prob_sequence(self, next_chars, context):
        return np.log(1.0 / self.voc_size) * len(next_chars)

# RNN language model
class RNNLanguageModel(LanguageModel, nn.Module): 
    # Initialize the RNN language model with the vocabulary size, embedding dimension, hidden dimension, and vocabulary index
    def __init__(self, vocab_size, embedding_dim, hidden_dim, vocab_index, sos_token="<sos>"):
        super(RNNLanguageModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.RNN(embedding_dim, hidden_dim, batch_first=True)
        self.output_layer = nn.Linear(hidden_dim, vocab_size)
        self.vocab_index = vocab_index
        self.sos_token = sos_token  # Store the SOS token

    # Forward pass of the RNN language model
    def forward(self, input_seq):
        embeds = self.embedding(input_seq)
        rnn_out, _ = self.rnn(embeds)
        logits = self.output_layer(rnn_out)
        return logits

    # Return the log probability of a single character given the context
    def get_log_prob_single(self, next_char, context):
        context_idx = torch.tensor([self.vocab_index.index_of(c) for c in context], dtype=torch.long).unsqueeze(0)
        output = self.forward(context_idx)
        next_char_idx = self.vocab_index.index_of(next_char)
        # Get the log probability of the next character
        log_prob = F.log_softmax(output[0, -1], dim=-1)[next_char_idx]
        return log_prob.item()

    # Return the log probability of a sequence of characters given the context
    def get_log_prob_sequence(self, next_chars, context):
        log_prob_sum = 0.0
        full_context = context + next_chars
        # Calculate the log probability of each character in the sequence
        for i in range(len(next_chars)):
            log_prob_sum += self.get_log_prob_single(full_context[len(context) + i], full_context[:len(context) + i])
        return log_prob_sum

# Train the language model using the training text and evaluate on the development text
def train_lm(args, train_text, dev_text, vocab_index, test_text=None):
    # Parameters for the RNN language model
    vocab_size = len(vocab_index)
    embedding_dim = 32
    hidden_dim = 64
    model = RNNLanguageModel(vocab_size, embedding_dim, hidden_dim, vocab_index, sos_token="<sos>")  # Add SOS token
    # Optimizer used for training the model to minimize the loss
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    # Loss function used to calculate the loss between the predicted and target values
    criterion = nn.CrossEntropyLoss()

    # Prepare training data
    # Convert the training text to indices using the vocabulary index
    train_indices = [vocab_index.index_of("<sos>")] + [vocab_index.index_of(c) for c in train_text]  # Add <sos> at the beginning
    train_inputs = torch.tensor(train_indices[:-1], dtype=torch.long)
    train_targets = torch.tensor(train_indices[1:], dtype=torch.long)

    # Prepare validation data
    # Convert the development text to indices using the vocabulary index
    dev_indices = [vocab_index.index_of("<sos>")] + [vocab_index.index_of(c) for c in dev_text]
    dev_inputs = torch.tensor(dev_indices[:-1], dtype=torch.long)
    dev_targets = torch.tensor(dev_indices[1:], dtype=torch.long)

    num_epochs = 10
    batch_size = 32
    train_dataset = TensorDataset(train_inputs, train_targets)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Lists to store loss and perplexity values
    train_loss_values = []
    dev_loss_values = []
    dev_perplexities = []

    # Training the model
    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0
        # Iterate over the training data
        for batch_inputs, batch_targets in train_dataloader:
            optimizer.zero_grad()
            outputs = model(batch_inputs.unsqueeze(0))
            loss = criterion(outputs.view(-1, vocab_size), batch_targets)
            loss.backward()
            # Step is called to update the weights
            optimizer.step()
            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_dataloader)
        train_loss_values.append(avg_train_loss)
        print(f"Epoch {epoch + 1}/{num_epochs}, Training Loss: {avg_train_loss:.4f}")

        # Evaluate on validation set
        model.eval()
        with torch.no_grad():
            # Unsqueeze to add batch dimension
            outputs = model(dev_inputs.unsqueeze(0))
            dev_loss = criterion(outputs.view(-1, vocab_size), dev_targets)
            dev_loss_values.append(dev_loss.item())

            # Calculate perplexity
            dev_log_prob = model.get_log_prob_sequence(dev_text, context="<sos>")  # Start with SOS token
            dev_perplexity = np.exp(-dev_log_prob / len(dev_text))
            dev_perplexities.append(dev_perplexity)
        print(f"Epoch {epoch + 1}/{num_epochs}, Validation Loss: {dev_loss.item():.4f}, Perplexity: {dev_perplexity:.4f}")

    # Plotting the loss values
    plt.figure(figsize=(8, 5))
    epochs = range(1, num_epochs + 1)
    plt.plot(epochs, train_loss_values, marker='o', linestyle='-', color='b', label='Training Loss')
    plt.plot(epochs, dev_loss_values, marker='o', linestyle='-', color='r', label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Over Epochs')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Plotting the perplexity values
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, dev_perplexities, marker='o', linestyle='-', color='g', label='Validation Perplexity')
    plt.xlabel('Epoch')
    plt.ylabel('Perplexity')
    plt.title('Validation Perplexity Over Epochs')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Testing the model if test_text is provided
    if test_text is not None:
        test_indices = [vocab_index.index_of("<sos>")] + [vocab_index.index_of(c) for c in test_text]  # Add SOS token
        test_inputs = torch.tensor(test_indices[:-1], dtype=torch.long)
        test_targets = torch.tensor(test_indices[1:], dtype=torch.long)

        model.eval()

        # Calculate loss and perplexity on the test set
        with torch.no_grad():
            # Unsqueeze to add batch dimension
            outputs = model(test_inputs.unsqueeze(0))
            # Calculate loss
            test_loss = criterion(outputs.view(-1, vocab_size), test_targets)
            test_log_prob = model.get_log_prob_sequence(test_text, context="<sos>")  # Start with SOS token
            # Calculate perplexity
            test_perplexity = np.exp(-test_log_prob / len(test_text))
        print(f"Test Loss: {test_loss.item():.4f}, Test Perplexity: {test_perplexity:.4f}")
    else:
        print("No test data provided. Skipping testing phase.")

    return model