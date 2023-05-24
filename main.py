import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch import autograd
from torchmeta.datasets import CIFARFS, Omniglot, MiniImagenet
from torchmeta.transforms import Categorical, ClassSplitter
from torchmeta.utils.data import BatchMetaDataLoader
from torchvision.transforms import transforms
from maml_parallel import MAML
import torchvision
from functools import partial
import torch.optim as optim
import wandb

import os
os.environ["CUDA_VISIBLE_DEVICES"]="5"
device = 'cuda' if torch.cuda.is_available() else 'cpu'

run = wandb.init(
#   mode="disabled",
  project="meta-kd",
  dir="./",
  name = 'resnet18-cifar-5-3',
  tags=["omniglot", "resnet18", "maml", "functional"],
)

if __name__ == '__main__':
    test = False
    n_way = 5 
    shots = 5
    batch_size = 24
    epochs = 200
    num_adaption_steps = 1
    seed = 42

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    transform = transforms.Compose([
        transforms.Resize((32,32)),
        transforms.ToTensor(),
    ])

    # Load data
    train_dataset = CIFARFS("data/", num_classes_per_task=n_way, meta_train=True, transform=transform, target_transform=Categorical(num_classes=n_way), download=False)
    val_dataset = CIFARFS("data/", num_classes_per_task=n_way, meta_val=True, transform=transform, target_transform=Categorical(num_classes=n_way), download=False)
    test_dataset = CIFARFS("data/", num_classes_per_task=n_way, meta_test=True, transform=transform, target_transform=Categorical(num_classes=n_way), download=False)

    train_dataset = ClassSplitter(train_dataset, shuffle=True, num_train_per_class=shots, num_test_per_class=shots)
    val_dataset = ClassSplitter(val_dataset, shuffle=True, num_train_per_class=shots, num_test_per_class=shots)
    test_dataset = ClassSplitter(test_dataset, shuffle=True, num_train_per_class=shots, num_test_per_class=shots)

    train_loader = BatchMetaDataLoader(train_dataset, batch_size=batch_size, num_workers=4)
    val_loader = BatchMetaDataLoader(val_dataset, batch_size=batch_size, num_workers=4)
    test_loader = BatchMetaDataLoader(test_dataset, batch_size=batch_size, num_workers=4)

    net = torchvision.models.resnet18(norm_layer=partial(nn.BatchNorm2d, track_running_stats=False))
    # net.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False) # Since omniglot is grayscale
    net.fc = nn.Linear(net.fc.in_features, n_way)
    net.to(device)
    net.train()

    maml = MAML()
    meta_opt = optim.Adam(net.parameters(), lr=1e-3)

    for epoch in range(epochs):
        maml.train(train_loader, net, device, meta_opt, epoch, num_adaption_steps)
        maml.test(val_loader, net, device, epoch, num_adaption_steps)

    maml.test(test_loader, net, device, epoch, num_adaption_steps)

    torch.save(net.state_dict(), 'trained/resnet18-omniglot-5-3.pth')
