import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
import pickle
import numpy as np
import pandas as pd

# Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load dataset
df = pd.read_csv("input.csv")
df['subject'].fillna('Unknown subject', inplace=True)

# Preprocess data
ticket_subjects = df['subject'].values
agent_groups = df['agent_group'].values

# Convert text to TF-IDF features
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(ticket_subjects).toarray()
with open("vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

# Convert labels to numerical values
encoder = LabelEncoder()
y = encoder.fit_transform(agent_groups)
with open("encoder.pkl", "wb") as f:
    pickle.dump(encoder, f)

# Convert labels to one-hot encoding
y_one_hot = np.eye(len(encoder.classes_))[y]

# Convert to PyTorch tensors
X_tensor = torch.tensor(X, dtype=torch.float32, device=device)
y_tensor = torch.tensor(y_one_hot, dtype=torch.float32, device=device)

# Create dataset and dataloaders
dataset = TensorDataset(X_tensor, y_tensor)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=10, shuffle=False)

# Define the Neural Network
class TicketClassifier(nn.Module):
    def __init__(self, input_size, output_size):
        super(TicketClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_size)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.softmax(self.fc3(x), dim=1)
        return x

# Initialize model
model = TicketClassifier(X.shape[1], len(encoder.classes_)).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
def train_model(model, train_loader, val_loader, epochs=50):
    best_val_loss = float('inf')
    patience = 3
    patience_counter = 0
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            output = model(X_batch)
            loss = criterion(output, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        val_loss = evaluate_model(model, val_loader)
        print(f"Epoch {epoch+1}, Training Loss: {total_loss/len(train_loader):.4f}, Validation Loss: {val_loss:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), "ticket_classification_model.pth")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

def evaluate_model(model, val_loader):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            output = model(X_batch)
            loss = criterion(output, y_batch)
            total_loss += loss.item()
    return total_loss / len(val_loader)

# Train the model
train_model(model, train_loader, val_loader)

# Prediction function
def predict_agent_group(subject_line):
    model.load_state_dict(torch.load("ticket_classification_model.pth"))
    model.to(device)
    model.eval()
    subject_features = vectorizer.transform([subject_line]).toarray()
    subject_tensor = torch.tensor(subject_features, dtype=torch.float32, device=device)
    with torch.no_grad():
        prediction = model(subject_tensor)
        predicted_class = torch.argmax(prediction).item()
    return encoder.inverse_transform([predicted_class])[0]

# Example predictions
new_subject_lines = [
    "Put your subject's for testing here!",
    "Lie by lie, we build the company and they feast"
]

for subject in new_subject_lines:
    print(f"Subject: '{subject}' -> Assigned to: {predict_agent_group(subject)}")
