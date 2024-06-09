import os
import torch
import pickle
from tqdm import tqdm
import torch.nn as nn
from model import CNNModel

class Client():
    def __init__(self, client_id, client_dataloader, client_logs):
        self.client_id = client_id
        self.train_loader = client_dataloader
        self.client_logs = client_logs

    def __load_global_model(self, device='cpu'):
        global_model_state = torch.load('./model_cache/global_model_state.pkl', map_location=device)
        model = CNNModel().to(device)
        model.load_state_dict(global_model_state)
        return model

    def __train(self, model, device, n_client_epochs=1):
        model.train()

        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters())

        print(f'Training Client: {self.client_id}')
        for i in tqdm(range(n_client_epochs)): # Batch_size: 1 -> FedSGD, b -> FedAvg
            train_loss = 0
            train_correct = 0
            # ReTHINK: Its like batch gradient descent for FedSGD.
            optimizer.zero_grad() # for FedSGD. We will be using aggregated gradient across all dataset.
            for iter, (img, target) in enumerate(tqdm(self.train_loader)):
                img, target = img.to(device), target.to(device)
                output = model(img)

                loss = loss_fn(output, target)
                loss.backward()

                pred = torch.argmax(output, axis=-1)

                train_loss += loss.item()
                train_correct += (pred == target).sum().item()

                if n_client_epochs > 1: # For FedAvg
                    optimizer.step()
                    optimizer.zero_grad()

            client_loss = train_loss/len(self.train_loader.dataset)
            client_accuracy = train_correct * 100/len(self.train_loader.dataset)
            print(f'Client id: {self.client_id}, \
                    Client Epoch : {i} \
                    Loss: , {client_loss:.4f} \
                    Accuracy: {client_accuracy:.4f}')
            
            if self.client_id not in self.client_logs:
                self.client_logs[self.client_id] = {
                    'loss': [client_loss],
                    'accuracy': [client_accuracy]
                }
            else:
                self.client_logs[self.client_id]['loss'].append(client_loss)
                self.client_logs[self.client_id]['accuracy'].append(client_accuracy)

        if n_client_epochs == 1:
            grads = {'n_samples': len(self.train_loader.dataset), 'named_grads': {}}
            for name, param in model.named_parameters():
                grads['named_grads'][name] = param.grad
            return grads
        else:
            weights = {'n_samples': len(self.train_loader.dataset), 'named_params': {}}
            for name, param in model.named_parameters():
                weights['named_params'][name] = param
            return weights



    def run(self, device='cpu', n_client_epochs=1):
        model = self.__load_global_model(device)
        if n_client_epochs == 1:
            grads   = self.__train(model=model, device=device, n_client_epochs=n_client_epochs)
            client_location = f'./model_cache/client_grad_{self.client_id}.pkl'
            torch.save(grads, client_location)
        else:
            weights = self.__train(model=model, device=device, n_client_epochs=n_client_epochs)
            client_location = f'./model_cache/client_weights_{self.client_id}.pkl'
            torch.save(weights, client_location)

        # if os.path.exists(client_location):
        #     os.remove(client_location)
        
        return self.client_logs
    
