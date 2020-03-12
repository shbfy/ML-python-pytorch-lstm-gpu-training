# ML-python-pytorch-lstm-gpu-training

Quickstart project for executing an IMDB classifier using the PyTorch framework on a GPU.

This quickstart trains the model and persists as in ONNX format. The service runtime will then serve the model on localhost where the user can then send GET requests to perform inference.

### Scalar recordings

Average loss and accuracy metrics are captured at the end of each training epoch for the train/val datasets. These can be inspected in the Scalar graph.
