import os
from src.FER2013 import FER2013

DATASET_FOLDER = "./data/fer2013/"
FER2013_MODELPATH : str = "./persistent_data/fer2013_model.pth"
FER2013_MODELPATH_SAVE : str = "./results/fer2013_model.pth"

def train_fer2013(epochs: int = 2, model_path: str=FER2013_MODELPATH, save_path: str=FER2013_MODELPATH_SAVE):
    if not os.path.exists(model_path):
        fer = FER2013()
    else:
        fer = FER2013(model_path)
    fer.train(pathToData=DATASET_FOLDER+"train", epochs=epochs)
    fer.save_state_dict(path=save_path)

if __name__=="__main__":
    train_fer2013()