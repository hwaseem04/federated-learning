import os
import data
import torch
import pickle
from model import CNNModel
from data import CustomDataset # Dont remove this. You will get error while loading
from clients import Client
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.optim import SGD 

def server_aggregate_weights(n_clients, device): # for FedAvg

    model_info = []
    for c in range(n_clients):
        model_info.append(torch.load(f'./model_cache/client_weights_{c}.pkl', map_location=device))

    total_weights = {}
    n_total_samples = 0
    # acc_weights = summation[ (number_samples_per_client/total_samples) * model_state_per_client]
    for info in model_info:
        n_samples = info['n_samples']
        for name, param in info['named_params'].items():
            if name not in total_weights:
                total_weights[name] = param * n_samples
            else:
                total_weights[name] += (param * n_samples) 
        n_total_samples += n_samples


    weights = {}
    for name, param in total_weights.items():
        weights[name] = param / n_total_samples

    return weights

def server_aggregate_gradients(n_clients, device): # For FedSGD
    grads_info = []
    for c in range(n_clients):
        grads_info.append(torch.load(f'./model_cache/client_grad_{c}.pkl', map_location=device))

    total_grads = {}
    n_total_samples = 0
    # acc_gradient = summation[ (number_samples_per_client/total_samples) *  gradient_per_client]
    for info in grads_info:
        n_samples = info['n_samples']
        for name, grad_value in info['named_grads'].items():
            if name not in total_grads:
                total_grads[name] = grad_value * n_samples
            else:
                total_grads[name] += (grad_value * n_samples) 
        n_total_samples += n_samples

    gradients = {}
    for name, grad_value in total_grads.items():
        gradients[name] = grad_value/ n_total_samples
        # print(gradients[name].shape, gradients[name].dtype)
    # print()
    return gradients

def server_step(model, optimizer, gradients=None, weights=None):
    model.train()
    optimizer.zero_grad()

    if gradients is not None:
        for name, parameter in model.named_parameters():
            parameter.grad = gradients[name]
        optimizer.step()
    if weights is not None:
        for name, parameter in model.named_parameters():
            parameter = weights[name]
    torch.save(model.state_dict(), './model_cache/global_model_state.pkl')

def evaluate_global_model(device, batch_size):
    with open(f'./client_dataset/test_data.pkl', 'rb') as f:
        test_dataset = pickle.load(f)
    dataloader = DataLoader(test_dataset, batch_size=batch_size)
    model = CNNModel().to(device)
    global_model_state = torch.load('./model_cache/global_model_state.pkl', map_location=device)
    model.load_state_dict(global_model_state)

    train_loss = 0
    train_correct = 0

    loss_fn = torch.nn.CrossEntropyLoss()

    with torch.no_grad():
        for img, target in tqdm(dataloader):
            img, target = img.to(device), target.to(device)
            output = model(img)
            loss = loss_fn(output, target)
            pred = torch.argmax(output, axis=-1)

            train_loss += (loss.item() * len(img))
            train_correct += (pred == target).sum().item()

    loss = train_loss/len(dataloader.dataset)
    accuracy = train_correct * 100/len(dataloader.dataset)
    print(f'======== TEST EVALUATION: \
                Loss: {loss}, \
                Accuracy: {accuracy} =========')
    return loss, accuracy


def fed(n_clients, n_server_epochs, batch_size, device,  n_client_epochs, algorithm):
    # Initialize clients
    ## Creating dataset for clients
    data.main()

    # Initialize server: Store initial random weights so that clients can use
    model = CNNModel().to(device)
    torch.save(model.state_dict(), './model_cache/global_model_state.pkl')
    optimizer = SGD(model.parameters(), lr=0.001)

    client_logs = {}

    ## Server communication rounds
    for epoch in range(n_server_epochs):
        # Training of all clients
        # Ideally clients should be running parallely. For simplicity lets make them to run sequentially

        for client_id in range(n_clients):
            with open(f'./client_dataset/client{client_id}_data.pkl', 'rb') as f:
                loaded_client_dataset = pickle.load(f)
            dataloader = DataLoader(loaded_client_dataset, batch_size=batch_size, shuffle=True)
            c = Client(client_id=client_id, client_dataloader=dataloader, client_logs=client_logs)

            # Client iteration
            client_logs = c.run(device=device, n_client_epochs=n_client_epochs)


        # Server aggregation and Update global parameter and store(share) the global weight
        if algorithm == 'fedsgd':
            gradients = server_aggregate_gradients(n_clients, device=device)
            server_step(model, optimizer, gradients=gradients)
        elif algorithm == 'fedavg':
            weights = server_aggregate_weights(n_clients, device=device)
            server_step(model, optimizer, weights=weights)
        
        loss, accuracy = evaluate_global_model(device, batch_size)
        if 'test' not in client_logs:
            client_logs['test'] = {'loss': [loss], 'accuracy': [accuracy]}
        else:
            client_logs['test']['loss'].append(loss)
            client_logs['test']['accuracy'].append(accuracy)

        with open(f'logs_{algorithm}.pkl', 'wb') as f:
            pickle.dump(client_logs, f)
        print('='*67, f'Epoch: {epoch+1} done', '='*68)
        print('='*150)
            
# Initialize server and client parameters
n_clients = 10
n_server_epochs = 10
n_client_epochs = 1
batch_size = 1

if n_client_epochs == 1:
    algorithm= 'fedsgd'
else:
    algorithm = 'fedavg'

device = torch.device('mps')

fed(n_clients,
    n_server_epochs, 
    batch_size, 
    device=device, 
    n_client_epochs=n_client_epochs,
    algorithm=algorithm)

