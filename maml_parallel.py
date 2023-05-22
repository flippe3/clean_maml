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
import wandb

import pandas as pd
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
plt.style.use('bmh')

import os
os.environ["CUDA_VISIBLE_DEVICES"]="5"

class MAML:
    # Trains a model for n_inner_iter using the support and returns a loss
    # using the query.
    def loss_for_task(self, net, n_inner_iter, x_spt, y_spt, x_qry, y_qry):
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

        qry_logits = functional_call(net, (new_params, buffers), x_qry)
        qry_loss = F.cross_entropy(qry_logits, y_qry)
        qry_acc = (qry_logits.argmax(dim=1) == y_qry).sum() / querysz

        return qry_loss, qry_acc


    def train(self, db, net, device, meta_opt, epoch, num_adaption_steps):
        params = dict(net.named_parameters())
        buffers = dict(net.named_buffers())

        max_batches = 40
        db = iter(db)
        
        for batch_idx in range(max_batches):
            start_time = time.time()

            batch = next(db)
            x_spt, y_spt = batch['train']
            x_qry, y_qry = batch['test']
            x_spt, y_spt = x_spt.to(device), y_spt.to(device)
            x_qry, y_qry = x_qry.to(device), y_qry.to(device)

            task_num, setsz, c_, h, w = x_spt.size()

            meta_opt.zero_grad()

            # In parallel, trains one model per task. There is a support (x, y)
            # for each task and a query (x, y) for each task.
            compute_loss_for_task = functools.partial(self.loss_for_task, net, num_adaption_steps)
            qry_losses, qry_accs = vmap(compute_loss_for_task)(x_spt, y_spt, x_qry, y_qry)

            # Compute the maml loss by summing together the returned losses.
            qry_losses.sum().backward()

            meta_opt.step()
            qry_losses = qry_losses.detach().sum() / task_num
            qry_accs = 100. * qry_accs.sum() / task_num
            i = epoch + float(batch_idx) / max_batches
            iter_time = time.time() - start_time
            if batch_idx % 4 == 0:
                print(f'[Epoch {i:.2f}] Train Loss: {qry_losses:.2f} | Acc: {qry_accs:.2f} | Time: {iter_time:.2f}')
                wandb.log({'loss': qry_losses, 'acc': qry_accs, 'epoch': i})

    def test(self, db, net, device, epoch, num_adaption_steps):
        params = dict(net.named_parameters())
        buffers = dict(net.named_buffers())
        max_batches = 10
        db = iter(db) 

        qry_losses = []
        qry_accs = []

        for batch_idx in range(max_batches):
            batch = next(db)
            x_spt, y_spt = batch['train']
            x_qry, y_qry = batch['test']
            x_spt, y_spt = x_spt.to(device), y_spt.to(device)
            x_qry, y_qry = x_qry.to(device), y_qry.to(device)

            task_num, setsz, c_, h, w = x_spt.size()

            # TODO: Maybe pull this out into a separate module so it
            # doesn't have to be duplicated between `train` and `test`?

            compute_loss_for_task = functools.partial(self.loss_for_task, net, num_adaption_steps)
            qry_loss, qry_acc = vmap(compute_loss_for_task)(x_spt, y_spt, x_qry, y_qry)

            qry_losses.append(qry_loss)
            qry_accs.append(qry_acc)

            # for i in range(task_num):
            #     new_params = params
            #     for _ in range(num_adaption_steps):
            #         spt_logits = functional_call(net, (new_params, buffers), x_spt[i])
            #         spt_loss = F.cross_entropy(spt_logits, y_spt[i])
            #         grads = torch.autograd.grad(spt_loss, new_params.values())
            #         new_params = {k: new_params[k] - g * 1e-1 for k, g, in zip(new_params, grads)}

            #     # The query loss and acc induced by these parameters.
            #     qry_logits = functional_call(net, (new_params, buffers), x_qry[i]).detach()
            #     qry_loss = F.cross_entropy(qry_logits, y_qry[i], reduction='none')
            #     qry_losses.append(qry_loss.detach())
            #     qry_accs.append((qry_logits.argmax(dim=1) == y_qry[i]).detach())

        qry_losses = torch.cat(qry_losses).mean().item()
        qry_accs = 100. * torch.cat(qry_accs).float().mean().item()
        print(f'[Epoch {epoch+1:.2f}] Test Loss: {qry_losses:.2f} | Acc: {qry_accs:.2f}')
        wandb.log({'test_loss': qry_losses, 'test_acc': qry_accs})