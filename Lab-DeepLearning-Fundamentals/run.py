import torch
import torch.nn as nn
import torch.optim as optim

X = torch.tensor([
    [5.0], [10.0], [10.0], [5.0], [10.0],
    [5.0], [10.0], [10.0], [5.0], [10.0],
    [5.0], [10.0], [10.0], [5.0], [10.0],
    [5.0], [10.0], [10.0], [5.0], [10.0]
], dtype=torch.float32)

y = torch.tensor([
    [30.5], [63.0], [67.0], [29.0], [62.0],
    [30.5], [63.0], [67.0], [29.0], [62.0],
    [30.5], [63.0], [67.0], [29.0], [62.0],
    [30.5], [63.0], [67.0], [29.0], [62.0]
], dtype=torch.float32)

class LinearRegressionModel(nn.Module):
    def __init__(self):
        super(LinearRegressionModel, self).__init__()
        self.fc1 = nn.Linear(1, 5)
        self.fc2 = nn.Linear(5, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
model = LinearRegressionModel()
criterion = nn.MSELoss()    
optimizer = optim.SGD(model.parameters(), lr=0.01)

for epoch in range(1000):
    optimizer.zero_grad()
    outputs = model(X)
    loss = criterion(outputs, y)
    loss.backward()
    optimizer.step()

    if (epoch+1) % 100 == 0:
        print(f'Epoch [{epoch+1}/1000], Loss: {loss.item()}')