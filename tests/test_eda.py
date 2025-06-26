import os
import pytest
from src.eda import eda

@pytest.mark.skipif(not os.path.exists("./data/fer2013/"), reason="Dataset non disponibile")
def test_eda():
    eda()
    os.remove("class_distribution.png")
    os.remove("samples_images.png")
    print("EDA test passed: EDA function executed successfully.")