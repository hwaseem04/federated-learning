
import os
import pickle
from torchvision import transforms, datasets
from torch.utils.data import Dataset

transform = transforms.Compose([
    transforms.Normalize((0.5,), (0.5,))
])

class CustomDataset(Dataset):
    def __init__(self, data, client_id=None, transform=None):
        self.transform= transform
        self.indices = []
        self.data = data
        if client_id is not None:
            for i, (_, target) in enumerate(data):
                if target == client_id:
                    self.indices.append(i)
        else:
            self.indices = [i for i in range(len(data))]
        
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        actual_index = self.indices[idx]
        image = self.transform(self.data[actual_index][0])
        label = self.data[actual_index][1]
        
        return image, label

def main():
    image_path = './'
    mnist_data = datasets.MNIST(root = image_path,
                                            train = True,
                                            transform = transforms.ToTensor(),
                                            download = True)
    mnist_test_data = datasets.MNIST(root = image_path, 
                                                train = False,
                                                transform = transforms.ToTensor(),
                                                download = True)
    client_dataset_directory = './client_dataset'
    if os.path.exists(client_dataset_directory):
        print('===== Dataset Already Exists =====')
    else:
        print('===== Creating dataset =====')
        # Creating train dataset
        os.makedirs(client_dataset_directory)
        for i in range(10):
            dataset_client = CustomDataset(mnist_data, client_id=i, transform=transform)
            with open(os.path.join(client_dataset_directory, f'client{i}_data.pkl'), 'wb') as f:
                pickle.dump(dataset_client, f)
                print(f'Client {i} data ready!' )
        print('Train Datasets created !')

        # Creating test dataset
        test_dataset = CustomDataset(mnist_test_data, transform=transform)
        with open(os.path.join(client_dataset_directory, f'test_data.pkl'), 'wb') as f:
            pickle.dump(test_dataset, f)
            print(f'Test data ready!' )
        print('===== Required Datasets generated =====')

    with open(os.path.join(client_dataset_directory, f'dataset_class.pkl'), 'wb') as f:
        pickle.dump(CustomDataset, f)
   
    
