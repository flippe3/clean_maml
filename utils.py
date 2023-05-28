import torch
import torchmeta
import numpy as np

def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def load_data(name, shots, n_ways, batch_size, train_transform, test_transform, root, num_workers=4):
    train_dataset = torchmeta.datasets.__dict__[name](
        root,
        num_classes_per_task=n_ways,
        meta_train=True,
        download=True,
        transform=train_transform, 
        target_transform=torchmeta.transforms.Categorical(num_classes=n_ways),
    )

    val_dataset = torchmeta.datasets.__dict__[name](
        root,
        num_classes_per_task=n_ways,
        meta_val=True,
        download=True,
        transform=test_transform, 
        target_transform=torchmeta.transforms.Categorical(num_classes=n_ways),
    )

    test_dataset = torchmeta.datasets.__dict__[name](
        root,
        num_classes_per_task=n_ways,
        meta_test=True,
        download=True,
        transform=test_transform, 
        target_transform=torchmeta.transforms.Categorical(num_classes=n_ways),
    )

    train_dataset = torchmeta.transforms.ClassSplitter(train_dataset, shuffle=True, num_train_per_class=shots, num_test_per_class=shots)
    val_dataset = torchmeta.transforms.ClassSplitter(val_dataset, shuffle=True, num_train_per_class=shots, num_test_per_class=shots)
    test_dataset = torchmeta.transforms.ClassSplitter(test_dataset, shuffle=True, num_train_per_class=shots, num_test_per_class=shots)

    train_dataloader = torchmeta.utils.data.BatchMetaDataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_dataloader = torchmeta.utils.data.BatchMetaDataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_dataloader = torchmeta.utils.data.BatchMetaDataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    return train_dataloader, val_dataloader, test_dataloader