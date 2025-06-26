import os
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torch
import time
from tqdm import tqdm
import torch.optim as optim
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

DATASET_FOLDER = "./data/fer2013/"
FER2013_MODELPATH : str = "fer2013_model.pth"
TOTAL_CLASS_NUM = 7
EMOTION_MAP = {     # Mappa delle emozioni (opzionale)
    0: "Angry", 
    1: "Disgust", 
    2: "Fear", 
    3: "Happy",
    4: "Sad", 
    5: "Surprise", 
    6: "Neutral"
}

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 64, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64*10*10, 64),
            nn.ReLU(),
            nn.Linear(64, TOTAL_CLASS_NUM)
        )
    
    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x

def train_fer2013(epochs: int, save_path: str=FER2013_MODELPATH):
    '''
    Train del modello su CPU
    '''
    transform = transforms.Compose([transforms.Grayscale(num_output_channels=1), transforms.Resize((48, 48)), transforms.ToTensor()])
    train_set = ImageFolder(root=DATASET_FOLDER+'train', transform=transform)
    train_loader = DataLoader(train_set, batch_size=64, shuffle=True)

    model = SimpleCNN()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    start = time.time()
    for epoch in tqdm(range(epochs)):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoca [{epoch+1}/{epochs}], Loss: {running_loss/len(train_loader):.4f}")
    print(f"Training time: {time.time()-start:.2f} seconds")

    torch.save(model.state_dict(), save_path)

def evaluate_fer2013(model_path: str=FER2013_MODELPATH):
    '''
    Evaluate del modello
    '''
    transform = transforms.Compose([transforms.Grayscale(num_output_channels=1), transforms.Resize((48, 48)), transforms.ToTensor()])
    test_set = ImageFolder(root=DATASET_FOLDER+'test', transform=transform)
    test_loader = DataLoader(test_set, batch_size=1000, shuffle=False)

    assert os.path.exists(model_path), f"Model file {model_path} non esistente."

    model = SimpleCNN()
    model.load_state_dict(torch.load(model_path))

    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted==labels).sum().item()
            all_preds.extend(predicted.numpy())
            all_labels.extend(labels.numpy())
    
    accuracy = correct/total
    print(f"Test sull'accuratezza: {accuracy:.4f}")

    # Classification Report
    print("Classification Report: ", classification_report(all_labels, all_preds))
    # Plot Confusion Matrix
    conf_matrix = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8,6))
    class_names = [EMOTION_MAP[i] for i in range(TOTAL_CLASS_NUM)]
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Purples', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Emozione Ottenuta")
    plt.ylabel("Emozione Prevista")
    plt.title("Confusion Matrix")
    plt.savefig("confusion_matrix.png")
    plt.show()

if __name__=="__main__":
    epoche = int(input("Quante epoche: "))
    train_fer2013(epoche)
    evaluate_fer2013()