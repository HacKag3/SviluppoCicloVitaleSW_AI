import os
from src.FER2013 import FER2013

FER2013_MODELPATH : str = "./results/fer2013_model.pth"
IMAGE_PATH : str = './data/fer2013/inference'
INFERENCE_PATH : str = './results/inference'

def inference_fer2013(model_path: str=FER2013_MODELPATH, image_path: str=IMAGE_PATH, output_path: str=INFERENCE_PATH):
    if not os.path.exists(model_path):
        raise FileNotFoundError("Necessario un modello pre-addestrato esistente")
    
    fer = FER2013(model_path)
    for filename in os.listdir(image_path):
        if filename.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
            fer.inference_singleImg(pathToImg=os.path.join(image_path, filename), output_dir=output_path)

if __name__=="__main__":
    inference_fer2013()