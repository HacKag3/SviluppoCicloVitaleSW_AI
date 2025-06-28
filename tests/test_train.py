import os
import pytest
from src.train_fer2013 import train_fer2013
from src.evaluate_fer2013 import evaluate_fer2013

FER2013_MODELPATH : str = "./results/fer2013_model.pth"

@pytest.mark.skipif(not os.path.exists("./data/fer2013/"), reason="Dataset non disponibile")
def test_fer2013_training():
    train_fer2013(1, FER2013_MODELPATH)
    assert os.path.exists(FER2013_MODELPATH), "Model file not found after training."
    assert os.path.getsize(FER2013_MODELPATH)>0, "Model file is empty."

    import torch
    from src.train_fer2013 import FER2013
    fer = FER2013(FER2013_MODELPATH)

    xin = torch.randn(1, 1, 48, 48)
    fer.model.eval()
    with torch.no_grad():
        output = fer.model(xin)
    assert output.shape==(1,7), "Model output shape in incorrect."

    os.remove(FER2013_MODELPATH)
    if os.path.exists("confusion_matrix.png"):
        os.remove("confusion_matrix.png")
    print("Test passed: Model trained and saved successfully.")

@pytest.mark.skipif(not os.path.exists("./data/fer2013/"), reason="Dataset non disponibile")
def test_fer2013_evaluation():
    train_fer2013(1, FER2013_MODELPATH)
    evaluate_fer2013(FER2013_MODELPATH)
    os.remove(FER2013_MODELPATH)
    os.remove("confusion_matrix.png")
