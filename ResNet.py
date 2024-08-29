# ResNet
 
import torch
from torch import nn, optim
from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.nn.functional as F
 
batch_size = 64
transforms = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081))])
 
train_dataset = datasets.MNIST(root='../dataset/mnist/', train=True, download=True, transform=transforms)
train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
 
test_dataset = datasets.MNIST(root='../dataset/mnist', train=True, download=True, transform=transforms)
test_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
 
 
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.channels = channels
 
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
 
    def forward(self, x):
        y = F.relu(self.conv1(x))
        y = self.conv2(y)
        return F.relu(x + y)
 
 
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5)
        self.mp = nn.MaxPool2d(2)
 
        self.rbloch1 = ResidualBlock(16)
        self.rbloch2 = ResidualBlock(32)
 
        self.fc = nn.Linear(512, 10)
 
    def forward(self, x):
        in_size = x.size(0)
        x = self.mp(F.relu(self.conv1(x)))
        x = self.rbloch1(x)
        x = self.mp(F.relu(self.conv2(x)))
        x = self.rbloch2(x)
        x = x.view(in_size, -1)
        x = self.fc(x)
        return x
 
 
model = Net()
 
criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
 
 
def train(epoch):
    running_loss = 0.0
    for batch_idx, data in enumerate(train_loader, 0):
        inputs, target = data
        optimizer.zero_grad()
 
        outputs = model(inputs)
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()
 
        running_loss += loss.item()
        if batch_idx % 300 == 299:
            print('[%d, %5d] loss: %.3f' % (epoch + 1, batch_size + 1, running_loss / 300))
            running_loss = 0.0
 
 
def test():
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            outputs = model(images)
            _, predicted = torch.max(outputs.data, dim=1)
 
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
 
        print('accuracy on test set: %d %% ' % (100 * correct / total))
 
 
if __name__ == '__main__':
    for epoch in range(10):
        train(epoch)
        test()
 