from torch.utils.data import DataLoader
from torch.utils.data.dataset import Subset
from torchvision import datasets
from torchvision import transforms
import torch

def get_dataloader(type='cifar10', data_root='../data', batch_size=16):
    '''
    Return: train, valid and test data loader. Each loader contains (data, label) pairs
    '''
    if type is 'cifar10':
        train_indices = torch.arange(0, 48000)
        valid_indices = torch.arange(48000, 50000)
        train_transform = transforms.Compose([transforms.Resize((64, 64)),
                                      # transforms.RandomCrop((64, 64)),
                                      transforms.ToTensor()])
        test_transform = transforms.Compose([transforms.Resize((64, 64)),
                                     # transforms.CenterCrop((64, 64)),
                                     transforms.ToTensor()])
        train_and_valid = datasets.CIFAR10(root=data_root, train=True, transform=train_transform, download=False)
        train_dataset = Subset(train_and_valid, train_indices)
        valid_dataset = Subset(train_and_valid, valid_indices)
        test_dataset = datasets.CIFAR10(root=data_root, train=False, transform=train_transform, download=False)
        train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
        valid_loader = DataLoader(dataset=valid_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
        return train_loader, valid_loader, test_loader