from support.omniglot_loaders import OmniglotNShot
from torch.func import vmap, grad, functional_call
import torch.optim as optim
import torch.nn.functional as F
import torchvision
from torch import nn
import torch
import matplotlib.pyplot as plt
import argparse
import time
import functools
import torchvision
from functools import partial

import pandas as pd
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
plt.style.use('bmh')

# Trains a model for n_inner_iter using the support and returns a loss
# using the query.
def loss_for_task(net, n_inner_iter, x_spt, y_spt, x_qry, y_qry):
    params = dict(net.named_parameters())
    buffers = dict(net.named_buffers())
    querysz = x_qry.size(0)

    def compute_loss(new_params, buffers, x, y):
        logits = functional_call(net, (new_params, buffers), x)
        loss = F.cross_entropy(logits, y)
        return loss

    new_params = params
    for _ in range(n_inner_iter):
        grads = grad(compute_loss)(new_params, buffers, x_spt, y_spt)
        new_params = {k: new_params[k] - g * 1e-1 for k, g, in grads.items()}

    # The final set of adapted parameters will induce some
    # final loss and accuracy on the query dataset.
    # These will be used to update the model's meta-parameters.
    qry_logits = functional_call(net, (new_params, buffers), x_qry)
    qry_loss = F.cross_entropy(qry_logits, y_qry)
    qry_acc = (qry_logits.argmax(dim=1) == y_qry).sum() / querysz

    return qry_loss, qry_acc


def train(db, net, device, meta_opt, epoch):
    params = dict(net.named_parameters())
    buffers = dict(net.named_buffers())
    n_train_iter = db.x_train.shape[0] // db.batchsz

    for batch_idx in range(n_train_iter):
        start_time = time.time()
        # Sample a batch of support and query images and labels.
        x_spt, y_spt, x_qry, y_qry = db.next()

        task_num, setsz, c_, h, w = x_spt.size()

        n_inner_iter = 1
        meta_opt.zero_grad()

        # In parallel, trains one model per task. There is a support (x, y)
        # for each task and a query (x, y) for each task.
        compute_loss_for_task = functools.partial(loss_for_task, net, n_inner_iter)
        qry_losses, qry_accs = vmap(compute_loss_for_task)(x_spt, y_spt, x_qry, y_qry)

        # Compute the maml loss by summing together the returned losses.
        qry_losses.sum().backward()

        meta_opt.step()
        qry_losses = qry_losses.detach().sum() / task_num
        qry_accs = 100. * qry_accs.sum() / task_num
        i = epoch + float(batch_idx) / n_train_iter
        iter_time = time.time() - start_time
        if batch_idx % 4 == 0:
            print(
                f'[Epoch {i:.2f}] Train Loss: {qry_losses:.2f} | Acc: {qry_accs:.2f} | Time: {iter_time:.2f}'
            )

def test(db, net, device, epoch):
    # Crucially in our testing procedure here, we do *not* fine-tune
    # the model during testing for simplicity.
    # Most research papers using MAML for this task do an extra
    # stage of fine-tuning here that should be added if you are
    # adapting this code for research.
    params = dict(net.named_parameters())
    buffers = dict(net.named_buffers())
    n_test_iter = db.x_test.shape[0] // db.batchsz

    qry_losses = []
    qry_accs = []

    for batch_idx in range(n_test_iter):
        x_spt, y_spt, x_qry, y_qry = db.next('test')
        task_num, setsz, c_, h, w = x_spt.size()

        # TODO: Maybe pull this out into a separate module so it
        # doesn't have to be duplicated between `train` and `test`?
        n_inner_iter = 1

        for i in range(task_num):
            new_params = params
            for _ in range(n_inner_iter):
                spt_logits = functional_call(net, (new_params, buffers), x_spt[i])
                spt_loss = F.cross_entropy(spt_logits, y_spt[i])
                grads = torch.autograd.grad(spt_loss, new_params.values())
                new_params = {k: new_params[k] - g * 1e-1 for k, g, in zip(new_params, grads)}

            # The query loss and acc induced by these parameters.
            qry_logits = functional_call(net, (new_params, buffers), x_qry[i]).detach()
            qry_loss = F.cross_entropy(
                qry_logits, y_qry[i], reduction='none')
            qry_losses.append(qry_loss.detach())
            qry_accs.append(
                (qry_logits.argmax(dim=1) == y_qry[i]).detach())

    qry_losses = torch.cat(qry_losses).mean().item()
    qry_accs = 100. * torch.cat(qry_accs).float().mean().item()
    print(f'[Epoch {epoch+1:.2f}] Test Loss: {qry_losses:.2f} | Acc: {qry_accs:.2f}')


if __name__ == '__main__':
    epochs = 15
    batch_size = 32 
    n_way, k_spt, k_qry = 5, 5, 15
    seed = 42

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    # Set up the Omniglot loader.
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    db = OmniglotNShot(
        '/tmp/omniglot-data',
        batchsz=batch_size,
        n_way=n_way,
        k_shot=k_spt,
        k_query=k_qry,
        imgsz=28,
        device=device,
    )

    net = torchvision.models.resnet18(norm_layer=partial(nn.BatchNorm2d, track_running_stats=False))
    net.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False) # Since omniglot is grayscale
    net.fc = nn.Linear(net.fc.in_features, n_way)
    net.to(device)
    net.train()

    meta_opt = optim.Adam(net.parameters(), lr=1e-3)

    for epoch in range(epochs):
        train(db, net, device, meta_opt, epoch)
        test(db, net, device, epoch)