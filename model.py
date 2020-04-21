import numpy as np
import torch
from torch import nn


# Long Short Term Memory Network
class LSTM(nn.Module):

    def __init__(self, layer_dim, hidden_dim, vocab_size, embedding_dim,
                 output_dim, dropout_proba=0.2):
        """
        Initalise the model with
        :param layer_dim: Number of recurrent layers
        :param hidden_dim: Size of features in hidden state `h`
        :param vocab_size: Size of the vocabulary containing unique words
        :param embedding_dim: Embedding length of word embeddings in the input
        :param dropout_proba: Probability of dropout
        :param output_dim: Number of classes to predict
        """
        super(LSTM, self).__init__()
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim

        self.lstm = nn.LSTM(embedding_dim, hidden_dim, layer_dim,
                            bidirectional=False, dropout=dropout_proba, 
                            batch_first=True)
        self.dropout = nn.Dropout(dropout_proba)

        self.fc1 = nn.Linear(hidden_dim, 32)
        self.fc2 = nn.Linear(32, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        Forward pass of model on a given input and hidden state
        :param x: Text embedding input
        :return: Output from network
        """
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.layer_dim, x.size(0),
                         self.hidden_dim).requires_grad_().cuda()

        # Initialize cell state with zeros
        c0 = torch.zeros(self.layer_dim, x.size(0),
                         self.hidden_dim).requires_grad_().cuda()

        x = self.word_embeddings(x)

        # Forward propagate LSTM
        out, (hn, cn) = self.lstm(x, (h0, c0))

        # Reshape such that output can be fed into a fully connected layer
        out = self.fc1(out[:, -1, :])

        out = self.dropout(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        return out

    def predict(self, dataloader):
        """
        Makes prediction across all samples contained within a dataloader
        :param dataloader:
        :return: predictions and ground truth label
        """
        y_true = np.array([])
        y_preds = np.array([])
        for _, batch in enumerate(dataloader):
            y_true = np.append(y_true, batch.label.cpu().numpy())

            y_pred = self.forward(batch.text).argmax(dim=1).cpu().numpy()
            y_preds = np.append(y_preds, y_pred)

        return y_preds, y_true
