# Using py.test framework
import os
import shutil

# run training script
import app  # noqa


def test_metrics_export():
    assert os.path.exists("metrics/tensorboard")

    # clean up test artifacts
    shutil.rmtree("metrics")


def test_model_export():
    assert os.path.exists("model.onnx")

    # clean up test artifacts
    os.remove("model.onnx")
