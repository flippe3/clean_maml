import torch
from torchvision.transforms import transforms
from maml_parallel import MAML
import torchvision
import torch.optim as optim
import torch.nn as nn
from functools import partial
import wandb
import copy

import utils
from models.cnn import CNN
from models.mlp import MLP

import os
os.environ["CUDA_VISIBLE_DEVICES"]="5"
device = 'cuda' if torch.cuda.is_available() else 'cpu'

run = wandb.init(
#   mode="disabled",
  project="thesis",
  dir="./",
  name = '3xresnet18-CIFAR-5-5-1',
#   tags=["omniglot", "resnet18", "maml", "functional"],
)

wandb.run.log_code(".")

if __name__ == '__main__':
    iterations = 15000 
    n_way = 5 
    shots = 5
    batch_size = 32
    num_adaption_steps = 1
    num_test_adaption_steps = 1
    seed = 42
    greyscale = True
    inner_lr = 0.01 # 0.1 for omniglot, 1e-3 for miniimagenet
    meta_lr = 0.01

    utils.set_seed(seed)

    transform = transforms.Compose([
        transforms.Resize((28,28)),
        transforms.ToTensor(),
    ])

    train_loader, val_loader, test_loader = utils.load_data(name="Omniglot",
                                                            shots=shots,
                                                            n_ways=n_way,
                                                            batch_size=batch_size,
                                                            root='../data',
                                                            num_workers=4,
                                                            train_transform=transform,
                                                            test_transform=transform)
    

    m1= torchvision.models.resnet18(norm_layer=partial(nn.BatchNorm2d, track_running_stats=False))
    m1.fc = nn.Linear(m1.fc.in_features, n_way)
    m1.to(device)

    maml = MAML()

    m1_meta_opt = optim.Adam(m1.parameters(), lr=1e-3)

    train_loader = iter(train_loader)

    for epoch in range(iterations):
        m1_loss, m1_acc = maml.train(train_loader, m1, device, m1_meta_opt, epoch, num_adaption_steps, inner_lr)

        wandb.log({"m1_train_loss": m1_loss, "m1_train_acc": m1_acc})

        if epoch % 1000 == 0:
            m1_loss, m1_acc = maml.test(val_loader, m1, device, num_adaption_steps, inner_lr)
            
            wandb.log({"m1_val_loss": m1_loss, "m1_val_acc": m1_acc})

    m1_test_loss, m1_test_acc = maml.test(test_loader, m1, device, epoch, num_adaption_steps)

    wandb.log({"m1_test_loss": m1_test_loss, "m1_test_acc": m1_test_acc}) 

    print(f"M1 Test Loss: {m1_test_loss:.4f} | Test Acc: {m1_test_acc:.4f}")

    torch.save(m1.state_dict(), 'trained/m1.pt')
