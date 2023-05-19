# import torch
# import torchvision
# from torchvision import datasets, transforms

# class Net(torch.nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()
#         self.fc1 = torch.nn.Linear(784, 128)
#         self.fc2 = torch.nn.Linear(128, 64)
#         self.fc3 = torch.nn.Linear(64, 10)

#     def forward(self, x):
#         x = torch.relu(self.fc1(x))
#         x = torch.relu(self.fc2(x))
#         x = self.fc3(x)
#         return x

# net = Net()

# train_loader = torch.utils.data.DataLoader(
#     datasets.MNIST('./data', train=True, download=True,
#                    transform=transforms.Compose([
#                        transforms.ToTensor(),
#                        transforms.Normalize((0.1307,), (0.3081,))
#                    ])),
#     batch_size=64, shuffle=True)


# criterion = torch.nn.CrossEntropyLoss()
# optimizer = torch.optim.SGD(net.parameters(), lr=0.01)

# for epoch in range(10):
#     for i, (inputs, labels) in enumerate(train_loader, 0):
#         inputs, labels = inputs.reshape(-1, 784), labels

#         optimizer.zero_grad()

#         outputs = net(inputs)
#         loss = criterion(outputs, labels)
#         loss.backward()
#         optimizer.step()

#         if i % 100 == 99:
#             print('[%d, %5d] loss: %.3f' %
#                   (epoch + 1, i + 1, loss.item()))

# import torch

# def compute_expression(x):
#     y = 2 * x**3 + 3 * x**2 + 4 * x + 5
#     return y

# x = torch.tensor(2.0)
# y = compute_expression(x)
# print(y)

# import torch
# import torchvision
# from torchvision import datasets, transforms
# import matplotlib.pyplot as plt

# transform = transforms.compose(
#     [transforms.ToTensor(),
#      transforms.Normalize((0.5,), (0.5))])

# trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)

# image, lavel = trainset[0]

# plt.imshow(image.reshape(28, 28), cmap='gray')
# plt.show()

# class Net(torch.nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()
#         self.conv1 = torch.nn.Conv2d(1, 6, 5)
#         self.pool = torch.nn.MaxPool2d(2, 2)
#         self.conv2 = torch.nn.Conv2d(6, 16, 5)
#         self.fc1 = torch.nn.Linear(16 * 4 * 4, 120)
#         self.fc2 = torch.nn.Linear(120, 84)
#         self.fc3 = torch.nn.Linear(84, 10)

    # def forward(self, x):
    #     x = self.pool(torch.nn.functional.relu(self.conv1(x)))
    #     x = self.pool(torch.nn.functional.relu(self.conv2(2)))
    #     x = x.view(-1, 16 * 4 * 4)
    #     x = torch.nn.functional.relu(self.fc1(x))
    #     x = torch.nn.functional.relu(self.fc2(x))
    #     x = self.fc3(x)
    #     return x

# net = Net()

# PATH = './mnist_net.pth'
# net.load_state_dict(torch.load(PATH))

# image_batch = image.unsqueeze(0)
# result = net(image_batch)
# predicted_label = result.argmax().item()

# print(predicted_label)

# import torch
# from torch import nn
# from torch import optim
# from torchvision import datasets, transforms

# # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # # tensorをGPUに移す
# # data = torch.tensor([1.0, 2.0])
# # data = data.to(device)

# # # CPU上のテンソルを作成
# # x = torch.randn(3, 3)

# # # GPUに移動
# # device = torch.device("cuda:0")
# # x = x.to(device)


# transform = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize((0.1307,), (0.3081,))
# ])

# train_dataset = datasets.MNIST(root='./mnist_data', train=True, transform=transform, download=True)

# train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
# test_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=64, shuffle=False)

# class Net(nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()
#         self.fc1 = nn.Linear(784, 128)
#         self.fc2 = nn.Linear(128, 10)

#     def forward(self, x):
#         x = x.view(-1, 784)
#         x = torch.relu(self.fc1(x))
#         x = self.fc2(x)
#         return x

# model = Net()

# criterion = nn.CrossEntropyLoss()
# optimizer = optim.SGD(model.parameters(), lr=0.01)

# for epoch in range(5):
#     for i, (images, labels) in enumerate(train_loader):
#         outputs = model(images)
#         loss = criterion(outputs, labels)

#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()

#         print(f'Epoch {epoch+1}, Loss: {loss.item()}')
# import torch
# torch.cuda.is_available()

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# print(device)
# print(torch.cuda.device_count())
# print(torch.cuda.get_device_name(0))
# if torch.cuda.is_available():
#     print(f'GPUが利用可能です。現在のデバイスは {torch.cuda.current_device()} で、その名前は {torch.cuda.get_device_name()} です。')
# else:
#     print('GPUは利用できません。')

# カスタムデータセットの作成

# import os
# import torch
# from torch.utils.data import Dataset, DataLoader
# from torchvision.transforms import ToTensor
# from PIL import Image

# # 画像のパスとラベルを含むリストを定義
# image_paths = ['images/bear.jpg', 'images/cake.jpg', 'images/suribachi.jpg']
# labels = [0, 1, 0]
# transform = ToTensor()

# class CustomImageDataset(Dataset):
#     def __init__(self, image_paths, labels, transform=None):
#         self.image_paths = image_paths
#         self.labels = labels
#         self.transform = transform

#     def __len__(self):
#         return len(self.image_paths)
    
#     def __getitem__(self, idx):
#         img_path = os.path.join(os.getcwd(), self.image_paths[idx])
#         img = Image.open(img_path)
#         if self.transform:
#             img = self.transform(img)
#         return img, self.labels[idx]
        
# dataset = CustomImageDataset(image_paths, labels, transform=transform)
# print(dataset)

# dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

# for inputs, labels in dataloader:
#     print(inputs.shape, labels)

#複数のGPU

# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torchvision import datasets, transforms
# from torch.utils.data import DataLoader

# class SimpleCNN(nn.Module):
#     def __init__(self):
#         super(SimpleCNN, self).__init__()
#         self.conv1 = nn.Conv2d(3, 6, 5)
#         self.pool = nn.MaxPool2d(2, 2)
#         self.conv2 = nn.Conv2d(6, 16, 5)
#         self.fc1 = nn.Linear(16 * 5 * 5, 120)
#         self.fc2 = nn.Linear(120, 84)
#         self.fc3 = nn.Linear(84,10)

#     def forward(self, x):
#         x = self.pool(self.relu(self.conv1(x)))
#         x = self.pool(self.relu(self.conv2(x)))
#         x = x.view(-1, 16 * 5 * 5)
#         x = self.relu(self.fc1(x))
#         x = self.relu(self.fc2(x))
#         x = self.fc3(x)
#         return x

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model = SimpleCNN()
# if torch.cuda.device_count() > 1:
#     print("lets use", torch.cuda.device_count(), "GPU!")
#     model = nn.DataParallel(model)
# model.to(device)