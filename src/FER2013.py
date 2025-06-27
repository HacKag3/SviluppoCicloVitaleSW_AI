
import time, os
import torch
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.io import read_image
from torchvision.transforms.functional import to_pil_image
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.optim as optim
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

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

class FER2013():
    def __init__(self, model_path : str = None):
        self.model = SimpleCNN()
        self.transform = self._get_transforms()
        if model_path is not None:
            self.load(model_path)

    def _get_transforms(self):
        return transforms.Compose([
            transforms.Grayscale(num_output_channels=1), 
            transforms.Resize((48, 48)), 
            transforms.ToTensor()
        ])

    def load(self, model_path: str):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file {model_path} not found")
        # Opzione 1: Caricamento completo
        try:
            obj = torch.load(model_path)
            if isinstance(obj, dict):
                self.model.load_state_dict(obj)
                print(f"State dict loaded from {model_path}")
            else:
                self.model = obj
                print(f"Full model loaded from {model_path}")
        except Exception as e:
            raise RuntimeError(f"Error loading model: {str(e)}")
    
    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.model, path)
        print(f"Model saved to {path}")

    def save_state_dict(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.model.state_dict(), path)
        print(f"Model saved to {path}")
    
    def train(self, pathToData : str, epochs : int = 2, batch_size : int = 64):
        train_set = ImageFolder(root=pathToData, transform=self.transform)
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
        
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()

        start = time.time()
        for epoch in range(epochs):
            self.model.train()
            running_loss = 0.0
            for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
                optimizer.zero_grad()
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
        
        print(f"Training time: {time.time()-start:.2f} seconds")
        print(f"Loss: {running_loss/len(train_loader):.4f}")
    
    def evaluate(self, pathToData : str, batch_size : int = 1000, verbose : bool = True):
        test_set = ImageFolder(root=pathToData, transform=self.transform)
        test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

        self.model.eval()
        
        correct = 0
        total = 0
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for images, labels in test_loader:
                outputs = self.model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted==labels).sum().item()
                all_preds.extend(predicted.numpy())
                all_labels.extend(labels.numpy())

        accuracy = correct/total

        if verbose:
            print(f"Test sull'accuratezza: {accuracy:.4f}")
            # Classification Report
            print("Classification Report: ", classification_report(all_labels, all_preds))
            # Plot Confusion Matrix
            conf_matrix = confusion_matrix(all_labels, all_preds)
            plt.figure(figsize=(8,6))
            sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Purples', xticklabels=EMOTION_MAP.values(), yticklabels=EMOTION_MAP.values())
            plt.xlabel("Emozione Ottenuta")
            plt.ylabel("Emozione Prevista")
            plt.title("Confusion Matrix")
            plt.savefig("./results/confusion_matrix.png")
            plt.close()
            print(f"Salvata la matrice di confusione in ./results/confusion_matrix.png")
    
    def inference_singleImg(self, pathToImg : str, output_dir: str):
        self.model.eval()
        image = read_image(pathToImg)
        image_tensor = self.transform(to_pil_image(image)).unsqueeze(0)
        
        with torch.no_grad():
            output = self.model(image_tensor)
            pred = torch.argmax(output, dim=1).item()
            predicted_class = EMOTION_MAP[pred]
        
        save_dir = os.path.join(output_dir, predicted_class)
        os.makedirs(save_dir, exist_ok=True)

        filename = os.path.basename(pathToImg)
        save_path = os.path.join(save_dir, filename)
        save_image(image_tensor.squeeze(0), save_path)
        print(f"Inferenza completata: {predicted_class} -> {save_path}")

