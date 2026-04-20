import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.conv1 = nn.Conv2d(3, 32, 3)
        self.conv2 = nn.Conv2d(32, 64, 3)

        self.pool = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(64  * 6 * 6, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):

        # Conv1 : 32 * 32 -> 30 * 30
        # Pool :  30 * 30 -> 15 * 15
        # Conv2 : 15 * 15 -> 13 * 13 
        # Pool : 13 * 13 -> 6 * 6
        # Final : 64 * 6 * 6
        x = self.pool(F.relu(self.conv1(x)))           
        x = self.pool(F.relu(self.conv2(x)))           

        # Flatten
        x = x.view(-1, 64 * 6 * 6)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x