from torch.utils.data import DataLoader

import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10, CIFAR100


def cifar_loader(num_classes, batch_size):
    """Cifar Data Loader"""

    # Define Transform for Image Processing #
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Prepare Dataset #
    if num_classes == 10:
        train_cifar = CIFAR10(root='./data/', train=True, transform=train_transform, download=True)
        val_cifar = CIFAR10(root='./data/', train=False, transform=val_transform, download=True)
        test_cifar = CIFAR10(root='./data/', train=False, transform=val_transform, download=True)

    elif num_classes == 100:
        train_cifar = CIFAR100(root='./data/', train=True, transform=train_transform, download=True)
        val_cifar = CIFAR100(root='./data/', train=False, transform=val_transform, download=True)
        test_cifar = CIFAR100(root='./data/', train=False, transform=val_transform, download=True)

    else:
        raise NotImplementedError

    # Prepare Data Loader #
    train_loader = DataLoader(dataset=train_cifar, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(dataset=val_cifar, batch_size=batch_size, shuffle=False, drop_last=True)
    test_loader = DataLoader(dataset=test_cifar, batch_size=batch_size, shuffle=False, drop_last=True)

    return train_loader, val_loader, test_loader