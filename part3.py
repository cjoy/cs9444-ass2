import numpy as np
import torch
import torch.nn as tnn
import torch.nn.functional as F
import torch.optim as topti
from torchtext import data
from torchtext.vocab import GloVe
from imdb_dataloader import IMDB
import re


# Class for creating the neural network.
class Network(tnn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.dropout_prob = 0.5
        self.embedding_dim = 50
        self.hidden_dim = 164
        self.lstm = tnn.LSTM(self.embedding_dim, self.hidden_dim, batch_first=True, bias=True, dropout=self.dropout_prob, num_layers=2, bidirectional=True)
        self.fc = tnn.Linear(self.hidden_dim*2, 1)
        self.dropout = tnn.Dropout(p = self.dropout_prob)

    def forward(self, input, length):
        batchSize, _, _ = input.size()
        lstm_out,(hn,cn) = self.lstm(input)
        hidden = self.dropout(torch.cat((hn[-2,:,:], hn[-1,:,:]), dim=1))
        out = self.fc(hidden.squeeze(0)).view(batchSize, -1)[:, -1]
        return out


class PreProcessing():
    def pre(x):
        """Called after tokenization"""
        return x

    def post(batch, vocab):
        """Called after numericalization but prior to vectorization"""
        return batch
    
    def tokenizer(text):
        string = text.replace('<br />', ' ')
        string = "".join([ c if c.isalnum() else " " for c in string ])
        return string.split()

    text_field = data.Field(lower=True, tokenize=tokenizer, include_lengths=True, batch_first=True, preprocessing=pre, postprocessing=post)


def rand_del(words, prob):
    result = []
    if len(words) == 1:
        result = words
    else:
        for i in words:
            if np.random.uniform(0, 1) > prob:
                result.append(i)
        if len(result) == 0:
            result.append(words[np.random.randint(0, len(words))])
    result = " ".join(result)
    return result

def rand_swap(words, n):
    result = words.copy()

    for i in range(n):
        r1, r2, j = np.random.randint(0, len(words)), 0, 0
        for j in range(3):
            r2 = np.random.randint(0, len(words))
            if r1 != r2:
                result[r1], result[r2] = result[r2], result[r1]
                break
    result = " ".join(result)
    return result

def aug_sentence(text, label, text_field, label_field):
    augmented = []
    fields = [('text', text_field), ('label', label_field)]
    for i in range(2):
        rd = rand_del(text, 0.1)
        rs = rand_swap(text, max(1, int(0.1 * len(text))))
        rd_example = data.Example.fromlist([rd, label], fields)
        rs_example = data.Example.fromlist([rs, label], fields)
        augmented.extend([rd_example, rs_example])
    return augmented
        

def lossFunc():
    """
    Define a loss function appropriate for the above networks that will
    add a sigmoid to the output and calculate the binary cross-entropy.
    """
    return tnn.BCEWithLogitsLoss()


def main():
    # Use a GPU if available, as it should be faster.
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Using device: " + str(device))

    # Load the training dataset, and create a data loader to generate a batch.
    textField = PreProcessing.text_field
    labelField = data.Field(sequential=False)

    train, dev = IMDB.splits(textField, labelField, train="train", validation="dev")

    # Data augmentation
    train.examples.extend(dev.examples)
    aug_examples = []
    for i in train.examples:
        aug_examples.extend(aug_sentence(i.text, i.label, textField, labelField))
    train.examples.extend(aug_examples)

    textField.build_vocab(train, dev, vectors=GloVe(name="6B", dim=50))
    labelField.build_vocab(train, dev)

    trainLoader, testLoader = data.BucketIterator.splits((train, dev), shuffle=True, batch_size=64,
                                                         sort_key=lambda x: len(x.text), sort_within_batch=True)

    net = Network().to(device)
    criterion =lossFunc()
    optimiser = topti.Adam(net.parameters(), lr=0.0003)  # Minimise the loss using the Adam algorithm.

    for epoch in range(10):
        running_loss = 0

        for i, batch in enumerate(trainLoader):
            # Get a batch and potentially send it to GPU memory.
            inputs, length, labels = textField.vocab.vectors[batch.text[0]].to(device), batch.text[1].to(
                device), batch.label.type(torch.FloatTensor).to(device)

            labels -= 1

            # PyTorch calculates gradients by accumulating contributions to them (useful for
            # RNNs).  Hence we must manually set them to zero before calculating them.
            optimiser.zero_grad()

            # Forward pass through the network.
            output = net(inputs, length)

            loss = criterion(output, labels)

            # Calculate gradients.
            loss.backward()

            # Minimise the loss according to the gradient.
            optimiser.step()

            running_loss += loss.item()

            if i % 32 == 31:
                print("Epoch: %2d, Batch: %4d, Loss: %.3f" % (epoch + 1, i + 1, running_loss / 32))
                running_loss = 0

    num_correct = 0

    # Save model
    torch.save(net.state_dict(), "./model.pth")
    # Overide as CPU model
    cpu_device = torch.device('cpu')
    cpu_net = Network().to(cpu_device)
    cpu_net.load_state_dict(torch.load('model.pth', map_location=torch.device(cpu_device)))
    torch.save(cpu_net.state_dict(), './model.pth')
    print("Saved model")


    # Evaluate network on the test dataset.  We aren't calculating gradients, so disable autograd to speed up
    # computations and reduce memory usage.
    with torch.no_grad():
        for batch in testLoader:
            # Get a batch and potentially send it to GPU memory.
            inputs, length, labels = textField.vocab.vectors[batch.text[0]].to(device), batch.text[1].to(
                device), batch.label.type(torch.FloatTensor).to(device)

            labels -= 1

            # Get predictions
            outputs = torch.sigmoid(net(inputs, length))
            predicted = torch.round(outputs)

            num_correct += torch.sum(labels == predicted).item()

    accuracy = 100 * num_correct / len(dev)

    print(f"Classification accuracy: {accuracy}")

if __name__ == '__main__':
    main()
