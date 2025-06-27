from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

DATASET_FOLDER = "./data/fer2013/"

EMOTION_MAP = {     # Mappa delle emozioni (opzionale)
    0: "Angry", 
    1: "Disgust", 
    2: "Fear", 
    3: "Happy",
    4: "Sad", 
    5: "Surprise", 
    6: "Neutral"
}

def eda():
    '''
    EDA (analisi esplorativa dei dati)
    '''
    transform = transforms.Compose([transforms.Grayscale(num_output_channels=1), transforms.ToTensor()])
    train_set = ImageFolder(root=DATASET_FOLDER+'train', transform=transform)
    test_set = ImageFolder(root=DATASET_FOLDER+'test', transform=transform)
    train_loader = DataLoader(train_set, batch_size=64, shuffle=True)

    print("EDA Reports:")
    print("- Train data size: ", len(train_set))
    print("- Test data size: ", len(test_set))
    # Controllo sulla distribuzione delle classi
    labels = np.array(train_set.targets)
    unique, counts = np.unique(labels, return_counts=True)
    class_dist = dict(zip(unique, counts))
    print("- Class distribution: ", end="")
    for class_idx, count in sorted(class_dist.items()):
        print(f"[{EMOTION_MAP[int(class_idx)]}: {count}]", end=" ")
    print()
    # Grafico sulla distribuzione delle classi
    plt.figure(figsize=(10,4))
    sns.barplot(x=[EMOTION_MAP[k] for k in class_dist.keys()], y=list(class_dist.values()))
    plt.title("Class distribution")
    plt.xlabel("Emotion")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig("class_distribution.png")
    plt.close()
    # Mostra delle immagini di esempio
    examples = enumerate(train_loader)
    _, (examples_data, examples_targets) = next(examples)
    plt.figure(figsize=(10,4))
    for i in range(10):
        plt.subplot(2,5,i+1)
        plt.imshow(examples_data[i][0], cmap="gray", interpolation="none")
        plt.title(f"Label: {EMOTION_MAP[examples_targets[i].item()]}")
        plt.axis("off")
    plt.tight_layout()
    plt.savefig("samples_images.png")
    plt.close()

if __name__=="__main__":
    eda()