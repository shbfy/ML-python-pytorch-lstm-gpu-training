import os
import shutil


def pytest_sessionstart(session):
    print("\nPre-session setup... "
         	"This may take a while as a new model is training.")
    import app  # noqa


def pytest_sessionfinish(session, exitstatus):
    print("\nPost-session teardown... Removing test artifacts.")

    # clean up artifacts
    shutil.rmtree("metrics")
    shutil.rmtree(".data")
    shutil.rmtree(".vector_cache")

    os.remove("processor.json")
    os.remove("model.onnx")
