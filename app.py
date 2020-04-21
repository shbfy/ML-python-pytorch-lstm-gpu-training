# coding=utf-8
import json
import os
import sys

import jsonpickle
import torch
import torch.onnx as torch_onnx
import torchtext
from sklearn.metrics import f1_score, accuracy_score, classification_report
from torch import nn
from torch import optim
from torchtext import data
from tqdm import tqdm

from model import LSTM

# Ensure result is reproducible
RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)
torch.backends.cudnn.deterministic = True

# Define hyper parameters
LEARNING_RATE = 0.001
BATCH_SIZE = 8
MAX_VOCAB_SIZE = 10000
N_CLASSES = 2
EMBEDDING_DIM = 300
N_LAYERS = 1

# Step 1: Set up target metrics for evaluating training

# Define a target loss metric to aim for
target_accuracy = 0.7

# instantiate classifier and scaler

# Step 2: Perform training for model
TEXT = data.Field(lower=True, batch_first=True)
LABEL = data.LabelField(dtype=torch.long, batch_first=True, sequential=False)
train, val = torchtext.datasets.IMDB.splits(TEXT, LABEL)

# build the GloVe embeddings
TEXT.build_vocab(
    train, vectors=torchtext.vocab.GloVe(name='6B', dim=EMBEDDING_DIM),
    max_size=MAX_VOCAB_SIZE)

LABEL.build_vocab(train)

train_iter, val_iter = data.BucketIterator.splits((train, val),
                                                  batch_size=BATCH_SIZE,
                                                  device='cuda')
train_iter.repeat = False
val_iter.repeat = False

# Instantiate the classifier
net = LSTM(layer_dim=N_LAYERS, hidden_dim=100, vocab_size=len(TEXT.vocab),
           embedding_dim=EMBEDDING_DIM, output_dim=N_CLASSES,
           dropout_proba=0.2).cuda()

# Define loss function and optimiser
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE, eps=1e-08)

# train model
epochs = 5

for epoch in range(epochs):
    train_loss = 0
    val_loss = 0

    net.train()
    train_correct = 0
    for batch in tqdm(train_iter):
        optimizer.zero_grad()
        text, target = batch.text, batch.label
        output = net(text)

        # calculate loss
        loss = criterion(output, target)

        # backpropagation, compute gradients
        loss.backward()

        # apply gradients
        optimizer.step()

        train_loss += loss.data.item()
        y_pred = output.argmax(dim=1, keepdim=True)
        train_correct += y_pred.eq(target.view_as(y_pred)).sum().item()

    train_loss /= len(train_iter)
    train_accuracy = 100 * train_correct / len(train_iter.dataset)

    net.eval()
    val_correct = 0
    for batch in val_iter:
        text, target = batch.text, batch.label
        output = net(text)
        loss = criterion(output, target)
        val_loss += loss.data.item()
        y_pred = output.argmax(dim=1, keepdim=True)
        val_correct += y_pred.eq(target.view_as(y_pred)).sum().item()

    val_loss /= len(val_iter)
    val_accuracy = 100 * val_correct / len(val_iter.dataset)

    print(f"Epoch {epoch + 1} :: Train/Loss {round(train_loss, 3)} :: "
          "Train/Accuracy {round(train_accuracy, 3)}")
    print(f"Epoch {epoch + 1} :: Val/Loss {round(val_loss, 3)} :: "
          "Val/Accuracy {round(val_accuracy, 3)}")

# Step 3: Evaluate the quality of the trained model
# Only persist the model if we have passed our desired threshold
if val_accuracy / 100 < target_accuracy:
    sys.exit('Training failed to meet threshold')

# Step 4: Persist the trained model in ONNX format in the local file system
# along with any significant metrics

# persist preprocessing steps
with open('processor.json', 'w') as f:
    f.write(jsonpickle.dumps(TEXT))

# persist model
rand_text, _ = next(iter(train_iter))
torch_onnx.export(net,
                  rand_text[[0]],  # Keep batch size to 1
                  'model.onnx',
                  verbose=False,
                  input_names=['input1'],
                  output_names=['output1'],
                  dynamic_axes={'input1': {0: 'batch'},
                                'output1': {0: 'batch'}})

# calculate set of quality metrics
y_pred, y_true = net.predict(val_iter)

f1_metric = f1_score(y_true, y_pred)
accuracy_metric = accuracy_score(y_true, y_pred)
print(classification_report(y_true, y_pred))

# write metrics
if not os.path.exists("metrics"):
    os.mkdir("metrics")

with open("metrics/f1.metric", "w+") as f:
    json.dump(f1_metric, f)
with open("metrics/accuracy.metric", "w+") as f:
    json.dump(accuracy_metric, f)
