import torch
import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder

# Define the same model architecture as used during training
class TicketClassifier(torch.nn.Module):
    def __init__(self, input_size, num_classes):
        super(TicketClassifier, self).__init__()
        self.fc1 = torch.nn.Linear(input_size, 128)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(128, 64)
        self.fc3 = torch.nn.Linear(64, num_classes)
        self.softmax = torch.nn.Softmax(dim=1)  # Softmax for classification

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.softmax(self.fc3(x))
        return x

# Load the saved vectorizer
with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# Load the label encoder
with open("encoder.pkl", "rb") as f:
    encoder = pickle.load(f)

# Load the trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TicketClassifier(input_size=3242, num_classes=len(encoder.classes_))
model.load_state_dict(torch.load("ticket_classification_model.pth", map_location=device))
model.to(device)
model.eval()

# Function for inference
def predict_agent_group(subject_line):
    # Convert text to numerical features using the saved vectorizer
    subject_features = vectorizer.transform([subject_line]).toarray()
    
    # Convert to torch tensor and move to correct device
    subject_tensor = torch.tensor(subject_features, dtype=torch.float32).to(device)

    # Get prediction
    with torch.no_grad():
        prediction = model(subject_tensor)

    # Get the agent group with the highest probability
    predicted_class = torch.argmax(prediction, dim=1).item()

    # Decode the numerical label back to agent group name
    predicted_agent_group = encoder.inverse_transform([predicted_class])[0]
    return predicted_agent_group

# Example usage
if __name__ == "__main__":
    new_subjects = [
        "System outage affecting all users",
        "Password reset request from user",
        "Unable to access invoice for last month",
        "Website homepage is down"
    ]

    for subject in new_subjects:
        print(f"Subject: '{subject}' -> Assigned to: {predict_agent_group(subject)}")
