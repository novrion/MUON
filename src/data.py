import torch
import torchvision


CIFAR_MEAN = torch.tensor((0.4914, 0.4822, 0.4465))
CIFAR_STD = torch.tensor((0.2470, 0.2435, 0.2616))


def get_data_transforms():
    train_transform = torchvision.transforms.Compose([
        torchvision.transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
        torchvision.transforms.RandomPerspective(distortion_scale=0.5, p=0.5),
        torchvision.transforms.RandomHorizontalFlip(p=0.5),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])

    test_transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])

    return train_transform, test_transform


def get_cifar(batch_size: int, path: str = "./data/cifar_10"):
    train_transform, test_transform = get_data_transforms()

    # Download datasets
    trainset = torchvision.datasets.CIFAR10(
        root=path,
        train=True,
        download=True,
        transform=train_transform
    )
    testset = torchvision.datasets.CIFAR10(
        root=path,
        train=False,
        download=True,
        transform=test_transform
    )

    # Put datasets into loaders
    trainloader = torch.utils.data.DataLoader(
        trainset,
        batch_size=batch_size,
        shuffle=True
    )
    testloader = torch.utils.data.DataLoader(
        testset,
        batch_size=batch_size,
        shuffle=False
    )

    return trainloader, testloader
