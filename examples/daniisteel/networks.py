import torch.nn as nn
import torch.nn.functional as F


class MLPNetwork(nn.Module):
    def __init__(self, n_features, n_output):
        super().__init__()
        self.fc1 = nn.Linear(in_features=n_features, out_features=100)
        self.fc2 = nn.Linear(in_features=100, out_features=50)
        self.fc3 = nn.Linear(in_features=50, out_features=50)
        self.fc4 = nn.Linear(in_features=50, out_features=50)
        self.fc5 = nn.Linear(in_features=50, out_features=10)
        self.out = nn.Linear(in_features=10, out_features=n_output)

    def forward(self, x):
        x = x.flatten(start_dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = self.out(x)
        return x