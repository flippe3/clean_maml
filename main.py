import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch import autograd
from torchmeta.datasets import CIFARFS, Omniglot, MiniImagenet
from torchmeta.transforms import Categorical, ClassSplitter
from torchmeta.utils.data import BatchMetaDataLoader
from torchvision.transforms import transforms
from maml import MAML

if __name__ == '__main__':
    test = False
    num_way = 5 
    shots = 1
    train_max_samples = 5000
    val_max_samples = 100
    num_inner_steps = 1
    inner_lr = 0.4
    learn_inner_lrs = False
    outer_lr = 0.001

    maml = MAML(
        num_way,
        num_inner_steps,
        inner_lr,
        learn_inner_lrs,
        outer_lr,
        train_max_samples,
        val_max_samples
    )

    transform = transforms.Compose([
        transforms.Resize((32,32)),
        # transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
    ])

    train_dataset = MiniImagenet("data/", num_classes_per_task=num_way, meta_train=True, transform=transform, target_transform=Categorical(num_classes=num_way), download=True)
    val_dataset = MiniImagenet("data/", num_classes_per_task=num_way, meta_val=True, transform=transform, target_transform=Categorical(num_classes=num_way), download=True)
    test_dataset = MiniImagenet("data/", num_classes_per_task=num_way, meta_test=True, transform=transform, target_transform=Categorical(num_classes=num_way), download=True)

    train_dataset = ClassSplitter(train_dataset, shuffle=True, num_train_per_class=1, num_test_per_class=15)
    val_dataset = ClassSplitter(val_dataset, shuffle=True, num_train_per_class=1, num_test_per_class=15)
    test_dataset = ClassSplitter(test_dataset, shuffle=True, num_train_per_class=1, num_test_per_class=15)

    train_loader = BatchMetaDataLoader(train_dataset, batch_size=32, num_workers=4)
    val_loader = BatchMetaDataLoader(val_dataset, batch_size=32, num_workers=4)
    test_loader = BatchMetaDataLoader(test_dataset, batch_size=32, num_workers=4)

    if test == False:
        maml.train(train_loader,val_loader)
    else:
        maml.test(test_loader)

