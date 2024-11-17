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


# For rnn classifier
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
        # Convert input sequence to tensor
        with torch.no_grad():
            input_indices = [self.vocab_index.index_of(char) for char in input_sequence]
            input_tensor = torch.tensor(input_indices, dtype=torch.long).unsqueeze(0)  # Add batch dimension
            output = self.forward(input_tensor)
            return torch.argmax(output, dim=1).item()


def train_frequency_based_classifier(cons_exs, vowel_exs):
    # Count the occurrences of each letter after the space for consonants and vowels
    consonant_counts = collections.Counter()
    vowel_counts = collections.Counter()
    for ex in cons_exs:
        consonant_counts[ex[-1]] += 1
    for ex in vowel_exs:
        vowel_counts[ex[-1]] += 1
    return FrequencyBasedClassifier(consonant_counts, vowel_counts)


def train_rnn_classifier(args, train_cons_exs, train_vowel_exs, dev_cons_exs, dev_vowel_exs, vocab_index):
    # Define hyperparameters/parameters
    embedding_size = 40
    hidden_size = 25
    output_size = 2
    batch_size = 16
    learning_rate = 0.001
    epochs = 10

    print("Embedding Size:", embedding_size)
    print("Hidden Size:", hidden_size)

    # Initialising the model,loss functions as well as the optimiser. 
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
    
    train_losses = []
    train_accuracies = []
    dev_accuracies = []

    # Training the model
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct_train_predictions = 0

        # Create mini-batches for training
        train_batches = create_batches(train_data, batch_size)

        for batch in train_batches:
            # Extract sequences and labels
            input_sequences = [item[0] for item in batch]  
            labels = torch.stack([item[1] for item in batch])  
            # Pad sequences to have the same length
            input_sequences = torch.nn.utils.rnn.pad_sequence(input_sequences, batch_first=True, padding_value=0)  # Pad sequences

            # Forward pass
            # Zero the gradients used to update the weights
            optimizer.zero_grad()
            output = model(input_sequences)
            loss = criterion(output, labels)

            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  
            optimizer.step()

            # Calculate loss and accuracy
            total_loss += loss.item()
            correct_train_predictions += (torch.argmax(output, dim=1) == labels).sum().item()

        avg_train_loss = total_loss / len(train_batches)
        train_accuracy = correct_train_predictions / len(train_data) * 100

        train_losses.append(avg_train_loss)
        train_accuracies.append(train_accuracy)

        # Evaluate on dev set
        model.eval()
        correct_dev_predictions = 0
        dev_batches = create_batches(dev_data, batch_size)

        # Calculate accuracy on dev set
        with torch.no_grad():
            for batch in dev_batches:
                input_sequences = [item[0] for item in batch]
                labels = torch.stack([item[1] for item in batch])
                input_sequences = torch.nn.utils.rnn.pad_sequence(input_sequences, batch_first=True, padding_value=0)  # Pad sequences

                output = model(input_sequences)
                correct_dev_predictions += (torch.argmax(output, dim=1) == labels).sum().item()

        dev_accuracy = correct_dev_predictions / len(dev_data) * 100
        dev_accuracies.append(dev_accuracy)
        
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_train_loss:.4f}, "
              f"Train Accuracy: {train_accuracy:.2f}%, Dev Accuracy: {dev_accuracy:.2f}%")
        

    # Plot training loss
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, epochs + 1), train_losses, label="Training Loss", marker="o")
    plt.title("Training Loss Over Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.legend()
    plt.show()

    # Plot training and dev accuracy
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, epochs + 1), train_accuracies, label="Training Accuracy", marker="o")
    plt.plot(range(1, epochs + 1), dev_accuracies, label="Dev Accuracy", marker="o")
    plt.title("Accuracy Over Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy (%)")
    plt.grid(True)
    plt.legend()
    plt.show()

    return model


# 
""" QUESTION 2 """
# def train_rnn_classifier(
#     args, train_cons_exs, train_vowel_exs, dev_cons_exs, dev_vowel_exs, vocab_index, max_context_length=20
# ):
#     embedding_size = 40
#     hidden_size = 25
#     output_size = 2
#     batch_size = 32
#     learning_rate = 0.001
#     epochs = 15

#     print("Embedding Size:", embedding_size)
#     print("Hidden Size:", hidden_size)

#     # To store accuracy for each context length
#     context_results = {}

#     for context_length in range(1, max_context_length + 1):
#         print(f"\nEvaluating context length: {context_length}")
        
#         model = RNNClassifier(vocab_index, embedding_size, hidden_size, output_size)
#         criterion = nn.CrossEntropyLoss()
#         optimizer = optim.Adam(model.parameters(), lr=learning_rate)

#         # Prepare training and dev data with truncated context
#         train_data = [
#             (
#                 torch.tensor([vocab_index.index_of(char) for char in ex[:context_length]], dtype=torch.long),
#                 torch.tensor(0),
#             )
#             for ex in train_cons_exs
#         ] + [
#             (
#                 torch.tensor([vocab_index.index_of(char) for char in ex[:context_length]], dtype=torch.long),
#                 torch.tensor(1),
#             )
#             for ex in train_vowel_exs
#         ]

#         dev_data = [
#             (
#                 torch.tensor([vocab_index.index_of(char) for char in ex[:context_length]], dtype=torch.long),
#                 torch.tensor(0),
#             )
#             for ex in dev_cons_exs
#         ] + [
#             (
#                 torch.tensor([vocab_index.index_of(char) for char in ex[:context_length]], dtype=torch.long),
#                 torch.tensor(1),
#             )
#             for ex in dev_vowel_exs
#         ]

#         # Function to create mini-batches
#         def create_batches(data, batch_size):
#             random.shuffle(data)  # Shuffle data for each epoch
#             return [data[i : i + batch_size] for i in range(0, len(data), batch_size)]

#         for epoch in range(epochs):
#             model.train()
#             total_loss = 0
#             correct_train_predictions = 0

#             # Create mini-batches for training
#             train_batches = create_batches(train_data, batch_size)

#             for batch in train_batches:
#                 input_sequences = [item[0] for item in batch]  # Extract sequences
#                 labels = torch.stack([item[1] for item in batch])  # Extract labels and stack them
#                 input_sequences = torch.nn.utils.rnn.pad_sequence(
#                     input_sequences, batch_first=True, padding_value=0
#                 )  # Pad sequences

#                 optimizer.zero_grad()
#                 output = model(input_sequences)
#                 loss = criterion(output, labels)
#                 loss.backward()
#                 torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient clipping
#                 optimizer.step()

#                 total_loss += loss.item()
#                 correct_train_predictions += (torch.argmax(output, dim=1) == labels).sum().item()

#         train_accuracy = correct_train_predictions / len(train_data) * 100

#         # Evaluate on dev set
#         model.eval()
#         correct_dev_predictions = 0
#         dev_batches = create_batches(dev_data, batch_size)

#         with torch.no_grad():
#             for batch in dev_batches:
#                 input_sequences = [item[0] for item in batch]
#                 labels = torch.stack([item[1] for item in batch])
#                 input_sequences = torch.nn.utils.rnn.pad_sequence(
#                     input_sequences, batch_first=True, padding_value=0
#                 )  # Pad sequences

#                 output = model(input_sequences)
#                 correct_dev_predictions += (torch.argmax(output, dim=1) == labels).sum().item()

#         dev_accuracy = correct_dev_predictions / len(dev_data) * 100
#         print(
#             f"Context Length: {context_length}, Train Accuracy: {train_accuracy:.2f}%, Dev Accuracy: {dev_accuracy:.2f}%"
#         )

#         # Store results for the context length
#         context_results[context_length] = {
#             "train_accuracy": train_accuracy,
#             "dev_accuracy": dev_accuracy,
#         }

#     # Plot the results
#     context_lengths = list(context_results.keys())
#     train_accuracies = [context_results[cl]["train_accuracy"] for cl in context_lengths]
#     dev_accuracies = [context_results[cl]["dev_accuracy"] for cl in context_lengths]

#     plt.figure(figsize=(10, 6))
#     plt.plot(context_lengths, train_accuracies, label="Train Accuracy", marker="o")
#     plt.plot(context_lengths, dev_accuracies, label="Dev Accuracy", marker="o")
#     plt.title("Accuracy vs. Context Length")
#     plt.xlabel("Context Length")
#     plt.ylabel("Accuracy (%)")
#     plt.legend()
#     plt.grid(True)
#     plt.show()

#     # Return the trained model (for the last context length) and context results
#     return model, context_results




#####################
# MODELS FOR PART 2 #
#####################

# Define the special start-of-sequence token
# SOS_TOKEN = "<s>"

# Update your VocabularyIndex class to include the SOS token
# class VocabularyIndex:
#     def __init__(self, chars):
#         self.chars = [SOS_TOKEN] + sorted(set(chars))  # Add SOS token at the beginning
#         self.char2idx = {char: idx for idx, char in enumerate(self.chars)}
#         self.idx2char = {idx: char for idx, char in enumerate(self.chars)}

#     def index_of(self, char):
#         return self.char2idx[char]

#     def char_at(self, idx):
#         return self.idx2char[idx]

#     def __len__(self):
#         return len(self.chars)


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
    def __init__(self, vocab_size, embedding_dim, hidden_dim, vocab_index):
        # Initialize the parent classes
        super(RNNLanguageModel, self).__init__()  # Initialize nn.Module
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.RNN(embedding_dim, hidden_dim, batch_first=True)
        self.output_layer = nn.Linear(hidden_dim, vocab_size)
        self.vocab_index = vocab_index

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


# def train_lm(args, train_text, dev_text, vocab_index):
#     vocab_size = len(vocab_index)
#     embedding_dim = 32
#     hidden_dim = 64
#     model = RNNLanguageModel(vocab_size, embedding_dim, hidden_dim, vocab_index)
#     optimizer = optim.Adam(model.parameters(), lr=0.001)
#     criterion = nn.CrossEntropyLoss()

#     train_indices = [vocab_index.index_of(c) for c in train_text]
#     inputs = torch.tensor(train_indices[:-1], dtype=torch.long)
#     targets = torch.tensor(train_indices[1:], dtype=torch.long)

#     num_epochs = 8
#     batch_size = 32
#     dataset = TensorDataset(inputs, targets)
#     dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

#     model.train()
#     for epoch in range(num_epochs):
#         total_loss = 0
#         for batch_inputs, batch_targets in dataloader:
#             optimizer.zero_grad()
#             output = model(batch_inputs.unsqueeze(0))  # Add batch dimension
#             loss = criterion(output.view(-1, vocab_size), batch_targets)
#             loss.backward()
#             optimizer.step()
#             total_loss += loss.item()
#         print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss / len(dataloader)}")

#     model.eval()
#     dev_log_prob = model.get_log_prob_sequence(dev_text, context=" ")
#     dev_perplexity = np.exp(-dev_log_prob / len(dev_text))
#     print(f"Development set perplexity: {dev_perplexity}")

#     return model


# Train the language model using the training text and evaluate on the development text
def train_lm(args, train_text, dev_text, vocab_index, test_text=None):
    # Parameters for the RNN language model
    vocab_size = len(vocab_index)
    embedding_dim = 32
    hidden_dim = 64
    model = RNNLanguageModel(vocab_size, embedding_dim, hidden_dim, vocab_index)
    # Optimizer used for training the model to minimize the loss
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    # Loss function used to calculate the loss between the predicted and target values
    criterion = nn.CrossEntropyLoss()

    # Prepare training data
    # Convert the training text to indices using the vocabulary index
    train_indices = [vocab_index.index_of(c) for c in train_text]
    train_inputs = torch.tensor(train_indices[:-1], dtype=torch.long)
    train_targets = torch.tensor(train_indices[1:], dtype=torch.long)

    # Prepare validation data
    # Convert the development text to indices using the vocabulary index
    dev_indices = [vocab_index.index_of(c) for c in dev_text]
    dev_inputs = torch.tensor(dev_indices[:-1], dtype=torch.long)
    dev_targets = torch.tensor(dev_indices[1:], dtype=torch.long)

    num_epochs = 8
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
            dev_log_prob = model.get_log_prob_sequence(dev_text, context=" ")
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
        test_indices = [vocab_index.index_of(c) for c in test_text]
        test_inputs = torch.tensor(test_indices[:-1], dtype=torch.long)
        test_targets = torch.tensor(test_indices[1:], dtype=torch.long)

        model.eval()

        # Calculate loss and perplexity on the test set
        with torch.no_grad():
            # Unsqueeze to add batch dimension
            outputs = model(test_inputs.unsqueeze(0))
            # Calculate loss
            test_loss = criterion(outputs.view(-1, vocab_size), test_targets)
            test_log_prob = model.get_log_prob_sequence(test_text, context=" ")
            # Calculate perplexity
            test_perplexity = np.exp(-test_log_prob / len(test_text))
        print(f"Test Loss: {test_loss.item():.4f}, Test Perplexity: {test_perplexity:.4f}")
    else:
        print("No test data provided. Skipping testing phase.")

    return model

