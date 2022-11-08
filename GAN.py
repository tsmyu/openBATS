import torch

in_features = 1 * 28 * 28

class Discriminator(torch.nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.fc1 = torch.nn.Linear(in_features, 384, bias=False)
        self.fc2 = torch.nn.Linear(384, 128, bias=False)
        self.fc3 = torch.nn.Linear(128, 1, bias=False)
        self.relu = torch.nn.LeakyReLU()
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))

        return x

zsize = 100

class Generator(torch.nn.Module):
    def __init__(self, zsize, in_features):
        super().__init__()
        self.fc1 = torch.nn.Linear(zsize, 256, bias=False)
        self.fc2 = torch.nn.Linear(256, 512, bias=False)
        self.fc3 = torch.nn.Linear(512, in_features, bias=False)
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = torch.tanh(self.fc3(x))

        return x