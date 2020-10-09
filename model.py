import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchvision
import torchvision.models as models
import torchvision.transforms as transforms

device = 'cuda' if torch.cuda.is_available() else 'cpu'

if device == 'cuda':
    torch.cuda.set_device(0)
    print(torch.cuda.current_device())


dataset = torchvision.datasets.ImageFolder(root="data/raw/",
                                           transform=transforms.Compose([
                                               transforms.ToTensor(),
                                               transforms.Normalize((0.5, 0.5, 0.5),
                                                                    (0.5, 0.5, 0.5)),
                                           ]))

dataloader = torch.utils.data.DataLoader(dataset,
                                         batch_size=4,
                                         shuffle=True,
                                         num_workers=16)

model = models.resnet18(pretrained=False)
model.fc = nn.Linear(512, len(dataset.classes), bias=True)
model.to(device)

print(model)
