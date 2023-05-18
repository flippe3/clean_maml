import copy 
import torch
from tqdm import tqdm
import torch.nn.functional as F
from collections import OrderedDict


class MAML:
    def __init__(self, model, meta_optimizer, device, epochs, max_batches, tasks, num_adaption_steps):
        self.tasks = tasks
        self.num_adaption_steps = num_adaption_steps
        self.meta_model = model
        self.device = device
        self.meta_optimizer = meta_optimizer
        self.epochs = epochs
        self.max_batches = max_batches
    
    def inner_loop(self, train_data, train_labels, test_data, test_labels):
        model = copy.deepcopy(self.meta_model)
        model.to(self.device)
        optim = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9, weight_decay=5e-4)

        train_data, train_labels = train_data.to(self.device), train_labels.to(self.device)
        test_data, test_labels = test_data.to(self.device), test_labels.to(self.device)

        # Adapt model on query data
        for i in range(self.num_adaption_steps):
            outputs = model(train_data)
            loss = F.cross_entropy(outputs, train_labels)
            loss.backward()
            optim.step()
            optim.zero_grad()

        # Compute gradients on test data
        test_outputs = model(test_data)
        test_loss = F.cross_entropy(test_outputs, test_labels)
        grads = torch.autograd.grad(test_loss, model.parameters(), create_graph=True)

        # Acc
        _, preds = torch.max(test_outputs, 1)
        total_correct = (preds == test_labels).sum().item()
        acc = total_correct / len(test_labels)

        meta_params = list(self.meta_model.parameters())
        updated_params = list(model.parameters())

        gradients = []
        for i, params in enumerate(model.parameters()):
            gradients.append(grads[i])


        # updated_params = OrderedDict(model.named_parameters())
        # name_list, tensor_list = zip(*updated_params.items())
        # for name, param, grad in zip(name_list, tensor_list, grads):
        #     updated_params[name] = param - 0.5 * grad

        return gradients, acc, loss

    
    def outer_loop(self, train_loader, val_loader, test_loader):
        for epoch in range(self.epochs):
            for j, batch in enumerate(train_loader):
                train_data, train_labels = batch['train']
                test_data, test_labels = batch['test']
                
                gradients = []
                accs = []
                losses = []
                for task in range(self.tasks):
                    if task == 0:
                        grads, acc, loss = self.inner_loop(train_data[task], train_labels[task], test_data[task], test_labels[task])
                        for i in range(len(grads)):
                            gradients.append(grads[i])
                        accs.append(acc)
                        losses.append(loss)
                    else:
                        grads, acc, loss = self.inner_loop(train_data[task], train_labels[task], test_data[task], test_labels[task])
                        for i in range(len(grads)):
                            gradients[i] += grads[i]
                        accs.append(acc)
                        losses.append(loss)
                
                # print(f"Epoch: {epoch}, Batch: {j}, Accuracy: {(sum(accs)/len(accs)):.3f} Loss: {(sum(losses)/len(losses)):.4f}")

                # Average gradients
                avg_gradients = []
                for i, grad in enumerate(gradients):
                    avg_gradients.append(grad / self.tasks)

                # Update meta model
                for i, params in enumerate(self.meta_model.parameters()):
                    params.grad = avg_gradients[i]

                self.meta_optimizer.step()
                self.meta_optimizer.zero_grad()
                
                if j == self.max_batches:
                    break   


