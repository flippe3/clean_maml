import torch
import os
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torchvision import models
from torchmeta.datasets import CIFARFS, Omniglot
from torchmeta.transforms import Categorical, ClassSplitter
from torchmeta.utils.data import BatchMetaDataLoader
from torchvision.transforms import transforms
from models import model_dict                
from maml import MAML

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using: {device}")

train_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    # transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
])
test_transform = transforms.Compose([
    transforms.Resize(256),
    # transforms.Resize((32,32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
])           

CLASSES = 2

train_dataset = CIFARFS("data/", num_classes_per_task=CLASSES, meta_train=True, transform=train_transform, target_transform=Categorical(num_classes=CLASSES), download=True)
val_dataset = CIFARFS("data/", num_classes_per_task=CLASSES, meta_val=True, transform=test_transform, target_transform=Categorical(num_classes=CLASSES), download=True)
test_dataset = CIFARFS("data/", num_classes_per_task=CLASSES, meta_test=True, transform=test_transform, target_transform=Categorical(num_classes=CLASSES), download=True)

train_dataset = ClassSplitter(train_dataset, shuffle=True, num_train_per_class=16, num_test_per_class=16)
val_dataset = ClassSplitter(val_dataset, shuffle=True, num_train_per_class=16, num_test_per_class=16)
test_dataset = ClassSplitter(test_dataset, shuffle=True, num_train_per_class=16, num_test_per_class=16)

train_loader = BatchMetaDataLoader(train_dataset, batch_size=16, num_workers=4)
val_loader = BatchMetaDataLoader(val_dataset, batch_size=16, num_workers=4)
test_loader = BatchMetaDataLoader(test_dataset, batch_size=16, num_workers=4)


# model = model_dict['resnet110'](num_classes=3)
# model = model.cuda()
# model.fc = torch.nn.Linear(model.fc.in_features, 2)
model = models.resnet18(num_classes=CLASSES)
model = model.cuda()

meta_optim = torch.optim.Adam(model.parameters(), lr=2e-4)

maml = MAML(model, meta_optimizer=meta_optim, device=device, epochs=3, max_batches=100, tasks=16, num_adaption_steps=10)

maml.outer_loop(train_loader, val_loader, test_loader)  