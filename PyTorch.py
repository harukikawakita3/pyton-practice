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

#         def forward(self, x):
#             x = self.pool(torch.nn.functional.relu(self.conv1(x)))
#             x = self.pool(torch.nn.functional.relu(self.conv2(2)))
#             x = x.view(-1, 16 * 4 * 4)
#             x = torch.nn.functional.relu(self.fc1(x))
#             x = torch.nn.functional.relu(self.fc2(x))
#             x = self.fc3(x)
#             return x

# net = Net()

# PATH = './mnist_net.pth'
# net.load_state_dict(torch.load(PATH))

# image_batch = image.unsqueeze(0)
# result = net(image_batch)
# predicted_label = result.argmax().item()

# print(predicted_label)

