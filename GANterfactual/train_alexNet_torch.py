import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision.datasets import ImageFolder
from torchvision.tranforms import Grayscale, ToTensor, Resize, Compose
from torch.utils.data import DataLoader
import wandb
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize wandb run
wandb.init(project="AlexNet_GANterfactual")

class AdaptedAlexNet(nn.Module):
    def __init__(self, input_dim, num_labels):
        super(AdaptedAlexNet, self).__init__()
        self.input_dim = input_dim
        self.num_labels = num_labels
        #convolutional layers
        self.conv1 = nn.Conv2d(1, 96, kernel_size=(11,11), stride=(4,4), padding = 'valid')
        self.conv2 = nn.Conv2d(in_channels=96, out_channels=256, kernel_size=(11, 11), stride=(1, 1), padding='valid')
        self.conv3 = nn.Conv2d(in_channels=256, out_channels=384, kernel_size=(3, 3), stride=(1, 1), padding='valid')
        self.conv4 = nn.Conv2d(in_channels=384, out_channels=384, kernel_size=(3, 3), stride=(1, 1), padding='valid')
        self.conv5 = nn.Conv2d(in_channels=384, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding='valid')

        #max pooling
        self.maxpool = nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding='valid')

        #activation
        self.relu = nn.ReLU()

        #normalization
        self.batchnorm1 = nn.BatchNorm2d(96)
        self.batchnorm2 = nn.BatchNorm2d(256)
        self.batchnorm3 = nn.BatchNorm2d(384)
        self.batchnorm4 = nn.BatchNorm2d(384)
        self.batchnorm5 = nn.BatchNorm2d(256)
        self.batchnorm6 = nn.BatchNorm1d(4096)
        self.batchnorm7 = nn.BatchNorm1d(1000)

        #dropout
        self.dropout = nn.Dropout(0.4)

        #fully connected
        self.fc1 = nn.Linear(4096, 4096)
        self.fc2 = nn.Linear(4096, 1000)
        self.fc3 = nn.Linear(1000, self.num_labels)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.batchnorm1(x)

        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.batchnorm2(x)

        x = self.conv3(x)
        x = self.relu(x)
        x = self.batchnorm3(x)

        x = self.conv4(x)
        x = self.relu(x)
        x = self.batchnorm4(x)

        x = self.conv5(x)
        x = self.relu(x)
        x = self.batchnorm5(x)

        #flatten to 1d tensor
        x = torch.flatten(x, 1)

        #dense layers
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.batchnorm6(x)

        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.batchnorm6(x)

        x = self.fc3(x)
        x = F.softmax(x, dim=1)

        return x

model = AdaptedAlexNet().to(device)
# Track model architecture
wandb.watch(model)
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=1e-2)

transforms = Compose([
    Resize((512, 512)),
    Grayscale(num_output_channels=1),
    ToTensor(),
])

#dataloader
train_data = ImageFolder(
    root="../data/train",  # Replace with your train data path
    transform=transforms
)

val_data = ImageFolder(
    root="../data/validation",  # Replace with your validation data path
    transform=transforms
)

train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
val_loader = DataLoader(val_data, batch_size=32, shuffle=True)
epochs = 10

wandb.config = {
    "learning_rate": 1e-2,  # Set your hyperparameters here
    "batch_size": 32,
    "epochs": epochs,
}

#training loop
for epoch in tqdm(range(epochs)):
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        # Validation loop
        model.eval()  # Set model to evaluation mode
        with torch.no_grad():  # Disable gradients for validation
            val_running_loss = 0.0
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs)
                val_loss = criterion(outputs, labels)

                val_running_loss += val_loss.item()
        
        # Log training and validation loss
        wandb.log({"loss": running_loss / len(train_loader), "val_loss": val_running_loss / len(val_loader)})
        print('[%d] train loss: %.3f, val loss: %.3f' %
            (epoch + 1, running_loss / len(train_loader), val_running_loss / len(val_loader)))

        model.train()  # Switch back to training mode

# Save your trained model
torch.save(model.state_dict(), "model.pt")