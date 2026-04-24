from torchvision import datasets, transforms
from torch.utils.data import Subset
import numpy as np
import torch
import random

SEED = 42
np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


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


def load_datasets_label_skew(num_clients):

    transform = transforms.Compose([
        transforms.ToTensor()
    ])

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

    targets_train = np.array(trainset.targets)
    targets_test = np.array(testset.targets)

    client_trainsets=[]
    client_testsets=[]

    # mỗi client nhận 2 labels
    labels_per_client=2

    for i in range(num_clients):

        labels=[
            (2*i)%10,
            (2*i+1)%10
        ]

        train_idx=np.where(
            np.isin(targets_train,labels)
        )[0]

        test_idx=np.where(
            np.isin(targets_test,labels)
        )[0]

        client_trainsets.append(
            Subset(trainset,train_idx)
        )

        client_testsets.append(
            Subset(testset,test_idx)
        )

    return client_trainsets,client_testsets

def load_datasets_dirichlet(num_clients, alpha=0.5):
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

    y_train=np.array(trainset.targets)
    y_test=np.array(testset.targets)

    num_classes=10

    client_train_indices=[[] for _ in range(num_clients)]
    client_test_indices=[[] for _ in range(num_clients)]

    for c in range(num_classes):

        idx=np.where(y_train==c)[0]
        np.random.shuffle(idx)

        proportions=np.random.dirichlet(
            alpha*np.ones(num_clients)
        )

        split_points=(
            np.cumsum(proportions)*len(idx)
        ).astype(int)[:-1]

        split_idx=np.split(idx,split_points)

        for i in range(num_clients):
            client_train_indices[i].extend(
                split_idx[i]
            )


    # -------- TEST SPLIT --------
    for c in range(num_classes):

        idx=np.where(y_test==c)[0]
        np.random.shuffle(idx)

        proportions=np.random.dirichlet(
            alpha*np.ones(num_clients)
        )

        split_points=(
            np.cumsum(proportions)*len(idx)
        ).astype(int)[:-1]

        split_idx=np.split(idx,split_points)

        for i in range(num_clients):
            client_test_indices[i].extend(
                split_idx[i]
            )


    client_trainsets=[]
    client_testsets=[]

    for i in range(num_clients):

        client_trainsets.append(
            Subset(
                trainset,
                client_train_indices[i]
            )
        )

        client_testsets.append(
            Subset(
                testset,
                client_test_indices[i]
            )
        )


    return (client_trainsets,client_testsets)