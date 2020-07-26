import torch
import torchvision
import torch.optim as optim
import torch.nn as nn
import torchvision.transforms as transforms
from model import studentNet,teacherNet

transform = transforms.Compose(
    [transforms.ToTensor()])

trainset = torchvision.datasets.MNIST(root='Model_Compression/knowledgeDistillation/example/data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.MNIST(root='Model_Compression/knowledgeDistillation/example/data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False, num_workers=2)

device =  torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
Loss = nn.CrossEntropyLoss()

# Train Student Net Without Teacher
print("Start training studeng net alone")
student = studentNet().to(device)
optimizer = optim.SGD(student.parameters(), lr=0.001, momentum=0.9)

for epoch in range(2):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = student(inputs)
        loss = Loss(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

print('Student Finished Training')

# Eval Student
correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = studentNet(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the student network without on the 10000 test images: %d %%' % (
    100 * correct / total))

