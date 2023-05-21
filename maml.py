"""Implementation of model-agnostic meta-learning for Omniglot."""

import argparse
import os

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch import autograd
from torchmeta.datasets import CIFARFS, Omniglot
from torchmeta.transforms import Categorical, ClassSplitter
from torchmeta.utils.data import BatchMetaDataLoader
from torchvision.transforms import transforms
from util import score
from models.ConvNet import ConvModel
# from models.ResNet import ResNet

NUM_INPUT_CHANNELS = 1
NUM_HIDDEN_CHANNELS = 64
KERNEL_SIZE = 3
NUM_CONV_LAYERS = 4
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
LOG_INTERVAL = 10
VAL_INTERVAL = LOG_INTERVAL * 5
NUM_TEST_TASKS = 600

class MAML:
    def __init__(
            self,
            num_outputs,
            num_inner_steps,
            inner_lr,
            learn_inner_lrs,
            outer_lr,
            train_max_samples,
            val_max_samples
    ):
        self.model = ConvModel(num_outputs)
        self._meta_parameters = self.model.meta_parameters
        self._num_inner_steps = num_inner_steps
        self._inner_lrs = {
            k: torch.tensor(inner_lr, requires_grad=learn_inner_lrs)
            for k in self._meta_parameters.keys()
        }
        self._outer_lr = outer_lr

        self._optimizer = torch.optim.Adam(
            list(self._meta_parameters.values()) +
            list(self._inner_lrs.values()),
            lr=self._outer_lr
        )

        self.train_max_samples = train_max_samples
        self.val_max_samples = val_max_samples

    def _inner_loop(self, images, labels, train):
        accuracies = []
        parameters = {
            k: torch.clone(v)
            for k, v in self._meta_parameters.items()
        }
        for _ in range(self._num_inner_steps):
            y_support = self.model.forward(images, parameters)
            inner_loss = F.cross_entropy(y_support, labels)

            accuracies.append(score(y_support, labels))
            grads_list = autograd.grad(inner_loss, parameters.values(), create_graph=train)

            for (i, n) in enumerate(parameters.keys()):
                parameters[n] = parameters[n] - self._inner_lrs[n] * grads_list[i]

        y_support = self.model.forward(images, parameters)
        accuracies.append(score(y_support, labels))

        return parameters, accuracies

    def _outer_step(self, task_batch, train):
        outer_loss_batch = []
        accuracies_support_batch = []
        accuracy_query_batch = []

        batch_images_support, batch_labels_support = task_batch['train']
        batch_images_query, batch_labels_query = task_batch['test']
        for task in range(len(task_batch['train'][0])):
            images_support, labels_support = batch_images_support[task], batch_labels_support[task]
            images_query, labels_query = batch_images_query[task], batch_labels_query[task]

            images_support, labels_support = images_support.to(DEVICE), labels_support.to(DEVICE)
            images_query, labels_query = images_query.to(DEVICE), labels_query.to(DEVICE)

            adapted_parameters, accuracy_support = self._inner_loop(images_support, labels_support, train=train)
            y_query = self.model.forward(images_query, adapted_parameters)

            # Compute classification losses
            outer_loss = F.cross_entropy(y_query, labels_query)
            outer_loss_batch.append(outer_loss)

            accuracies_support_batch.append(accuracy_support)
            accuracy_query = score(y_query, labels_query)
            accuracy_query_batch.append(accuracy_query)

        outer_loss = torch.mean(torch.stack(outer_loss_batch))
        accuracies_support = np.mean(accuracies_support_batch,axis=0)
        accuracy_query = np.mean(accuracy_query_batch)
        return outer_loss, accuracies_support, accuracy_query

    def train(self, dataloader_train, dataloader_val):
        for i_step, task_batch in enumerate(dataloader_train):
            self._optimizer.zero_grad()
            outer_loss, accuracies_support, accuracy_query = (
                self._outer_step(task_batch, train=True)
            )
            outer_loss.backward()
            self._optimizer.step()

            if i_step % VAL_INTERVAL == 0:
                losses = []
                accuracies_pre_adapt_support = []
                accuracies_post_adapt_support = []
                accuracies_post_adapt_query = []
                for i, val_task_batch in enumerate(dataloader_val):
                    outer_loss, accuracies_support, accuracy_query = (
                        self._outer_step(val_task_batch, train=False)
                    )
                    losses.append(outer_loss.item())
                    accuracies_pre_adapt_support.append(accuracies_support[0])
                    accuracies_post_adapt_support.append(accuracies_support[-1])
                    accuracies_post_adapt_query.append(accuracy_query)

                    if i == self.val_max_samples:
                        break
                    
                loss = np.mean(losses)
                accuracy_pre_adapt_support = np.mean(accuracies_pre_adapt_support)
                accuracy_post_adapt_support = np.mean(accuracies_post_adapt_support)
                accuracy_post_adapt_query = np.mean(accuracies_post_adapt_query)

                print(f"Loss: {loss}, acc_pre_adp_sup: {accuracy_pre_adapt_support}, acc_post_adp_sup: {accuracy_post_adapt_support}, acc_post_adp_query: {accuracy_post_adapt_query}") 
            if i_step == self.train_max_samples:
                break


    def test(self, dataloader_test):
        accuracies = []
        for i, task_batch in enumerate(dataloader_test):
            _, _, accuracy_query = self._outer_step(task_batch, train=False)
            accuracies.append(accuracy_query)
            if i == NUM_TEST_TASKS:
                break
        mean = np.mean(accuracies)
        print("Acc:", mean)
