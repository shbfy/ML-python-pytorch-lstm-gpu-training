# Using py.test framework
import os

import jsonpickle
import onnx
import onnxruntime as ort
import torchtext
from torch import nn


def test_metrics_export():
    assert os.path.exists("metrics/accuracy.metric")
    assert os.path.exists("metrics/f1.metric")


def test_model_export():
    assert os.path.exists("model.onnx")


def test_torchtext_exists():
    assert os.path.exists('processor.json')


def test_onnx_loads():
    model = onnx.load("model.onnx")
    assert model is not None


def test_torchtext_loads():
    with open('processor.json') as f:
        TEXT = jsonpickle.loads(f.read())

    assert isinstance(TEXT, torchtext.data.Field)


def test_export_makes_prediction():
    with open('processor.json') as f:
        TEXT = jsonpickle.loads(f.read())

    # Load the ONNX model
    model = onnx.load("model.onnx")

    # Check that the IR is well formed
    onnx.checker.check_model(model)

    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess = ort.InferenceSession('model.onnx', so)

    # Capture metadata on model
    input_name = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[0].name
    input_len = model.graph.input[0].type.tensor_type.shape.dim[1].dim_value

    doc = "That was a really great movie!"
    processed = TEXT.process(([TEXT.preprocess(doc)])).cuda()
    padded = nn.ConstantPad1d(
        (0, input_len - processed.shape[1]), 0)(processed)
    result = sess.run([output_name], {input_name: padded.numpy()})

    assert type(result) == list
