import torch
import torch.nn as torch_nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from dataset_builder import GestureSequenceDataset
import os

class GestureLSTM(torch_nn.Module):
    def __init__(self, input_size=132, hidden_size=128, num_layers=2, num_classes=5):
        super(GestureLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = torch_nn.LSTM(
            input_size, 
            hidden_size, 
            num_layers, 
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0
        )
        
        # Taking only the last hidden state of the sequence
        self.fc = torch_nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        # x is of shape (batch, seq_len, features)
        # We output from the final time step
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        out, _ = self.lstm(x, (h0, c0))
        # out[:, -1, :] corresponds to the context from the last sequence step
        out = self.fc(out[:, -1, :])
        return out

def train_model(epochs=10, sequence_length=30, batch_size=64, learning_rate=0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    try:
        files = [
            "dataset_clap.csv", 
            "dataset_disco.csv", 
            "dataset_hello.csv", 
            "dataset_wakanda.csv", 
            "dataset_zombie.csv"
        ]
        
        print("Loading dataset...")
        full_dataset = GestureSequenceDataset(files, sequence_length=sequence_length)
        print(f"Dataset successfully loaded. Length: {len(full_dataset)}")
                
        # Split into 80% train, 20% validation
        train_size = int(0.8 * len(full_dataset))
        val_size = len(full_dataset) - train_size
        train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        model = GestureLSTM(num_classes=5).to(device)
        criterion = torch_nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
        best_val_loss = float('inf')
        print("Starting training loop...")
        
        # NOTE Debug
        # Check an item from train_loader
        bx, by = next(iter(train_loader))
        print(f"Batch x: {bx.shape}, Batch y: {by.shape}")
        
        for epoch in range(epochs):
            model.train()
            running_loss = 0.0
            correct = 0
            total = 0
            
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
            train_acc = 100 * correct / total
            
            # Validation
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    
                    val_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()
            
            val_acc = 100 * val_correct / val_total
            avg_val_loss = val_loss / len(val_loader)
            
            print(f"Epoch [{epoch+1}/{epochs}] - "
                  f"Loss: {running_loss/len(train_loader):.4f}, Acc: {train_acc:.2f}% | "
                  f"Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.2f}%")
                  
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(model.state_dict(), "gesture_lstm_model.pth")
                
        print("Training finished! Best model saved to gesture_lstm_model.pth")
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    train_model(epochs=15, sequence_length=30)
