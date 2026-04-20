from torchvision import datasets, transforms
from torch.utils.data import Subset
import numpy as np

def load_datasets(num_clients):
    transform = transforms.Compose([transforms.ToTensor()])

    trainset = datasets.CIFAR10(
        root="./data",
        train=True,
        download=True,
        transform=transform 
    )
    
    testset = datasets.CIFAR10(
        root="./data",
        train=False,
        download=True,
        transform=transform
    )

    # ---- Split trainset ----
    train_size = len(trainset)
    train_indices = np.arange(train_size)
    np.random.shuffle(train_indices)

    train_split_size = train_size // num_clients

    # ---- Split testset (RIÊNG) ----
    test_size = len(testset)
    test_indices = np.arange(test_size)
    np.random.shuffle(test_indices)

    test_split_size = test_size // num_clients

    client_trainsets = []
    client_testsets = []

    for i in range(num_clients):
        train_idx = train_indices[i*train_split_size : (i+1)*train_split_size]
        test_idx = test_indices[i*test_split_size : (i+1)*test_split_size]

        train_subset = Subset(trainset, train_idx)
        test_subset = Subset(testset, test_idx)

        client_trainsets.append(train_subset)
        client_testsets.append(test_subset)

    return client_trainsets, client_testsets