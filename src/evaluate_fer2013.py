import os
from src.FER2013 import FER2013

DATASET_FOLDER = "./data/fer2013/"
FER2013_MODELPATH : str = "./persistent_data/fer2013_model.pth"
FER2013_MODELPATH_SAVE : str = "./results/fer2013_model.pth"

def evaluate_fer2013(model_path: str=FER2013_MODELPATH):
    if not os.path.exists(model_path):
        fer = FER2013()
    else:
        fer = FER2013(model_path)
    fer.evaluate(pathToData=DATASET_FOLDER+"test")

if __name__=="__main__":
    evaluate_fer2013()