import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import os
import time

class studentNet(nn.Module):
    def __init__(self):
        super(studentNet, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 800)
        self.fc2 = nn.Linear(800, 100)
        self.fc3 = nn.Linear(100, 10)
        

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class teacherNet(nn.Module):
    def __init__(self):
        super(teacherNet,self).__init__()
        #1*1*28*28
        self.conv1 = nn.Conv2d(1, 10, 5) 
        self.conv2 = nn.Conv2d(10, 20, 3) 
        self.fc1 = nn.Linear(20 * 10 * 10, 500)
        self.fc2 = nn.Linear(500, 10)
        
    def forward(self, x):
        in_size = x.size(0)
        out= self.conv1(x) # 1* 10 * 24 *24
        out = F.relu(out)
        out = F.max_pool2d(out, 2, 2) # 1* 10 * 12 * 12
        out = self.conv2(out) # 1* 20 * 10 * 10
        out = F.relu(out)
        out = out.view(in_size, -1) # 1 * 2000
        out = self.fc1(out) # 1 * 500
        out = F.relu(out)
        out = self.fc2(out) # 1 * 10
        return out

if __name__=="__main__":
    student = studentNet()
    teacher = teacherNet()

    print('# student parameters:', sum(param.numel() for param in student.parameters()))
    print('# teacher parameters:', sum(param.numel() for param in teacher.parameters()))
    