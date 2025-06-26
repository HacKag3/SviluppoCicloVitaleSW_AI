import os
from src.eda import eda

def test_eda():
    eda()
    os.remove("class_distribution.png")
    os.remove("samples_images.png")
    print("EDA test passed: EDA function executed successfully.")