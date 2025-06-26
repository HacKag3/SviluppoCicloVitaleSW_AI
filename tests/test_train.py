import os
from src.train_fer2013 import train_fer2013, evaluate_fer2013

FER2013_MODELPATH : str = "test_fer2013_model.pth"

def test_fer2013_training():
    train_fer2013(1, FER2013_MODELPATH)
    assert os.path.exists(FER2013_MODELPATH), "Model file not found after training."
    assert os.path.getsize(FER2013_MODELPATH)>0, "Model file is empty."

    import torch
    from src.train_fer2013 import SimpleCNN
    model = SimpleCNN()
    model.load_state_dict(torch.load(FER2013_MODELPATH))

    xin = torch.randn(1, 1, 48, 48)
    model.eval()
    with torch.no_grad():
        output = model(xin)
    assert output.shape==(1,7), "Model output shape in incorrect."

    os.remove(FER2013_MODELPATH)
    os.remove("confusion_matrix.png")
    print("Test passed: Model trained and saved successfully.")

def test_fer2013_evaluation():
    train_fer2013(1, FER2013_MODELPATH)
    evaluate_fer2013(FER2013_MODELPATH)
    os.remove(FER2013_MODELPATH)
    os.remove("confusion_matrix.png")
