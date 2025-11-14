import torch
import torchvision


def get_cifar(batch_size: int, path: str = "./data/cifar_10"):
    """Returns dataloader for train 50k images with certain batch size
    and testloader of 10k images. The output classes are:
    'plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'.
    Downloads into / retrives data from path. 
    """

    transform = torchvision.transforms.Compose(
        [torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    # --- Download Train and Test Dataset --- 
    trainset = torchvision.datasets.CIFAR10(root=path, train=True, download=True, transform = transform)
    testset = torchvision.datasets.CIFAR10(root=path, train=False, download=True, transform = transform)

    # --- Put into loaders ---
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)    
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False,)
    return trainloader, testloader


if __name__ == "__main__":
    trainloader, testloader = get_cifar(batch_size=4)
    images, labels = next(iter(trainloader))
    print(f"images shape: {images.shape}") # [batch size, 3, 32, 32]
    print(f"labels shape: {labels.shape}") # [batch size]

    # --- Visualize images ---
    import numpy as np
    import matplotlib.pyplot as plt
    images = images / 2 + 0.5 # Unnormalize 
    img_grid = torchvision.utils.make_grid(images)
    npimg = img_grid.numpy()
    plt.figure(figsize=(8,4))
    plt.imshow(np.transpose(npimg, (1, 2, 0))) # From [C, H, W] to [H, W, C] 
    plt.show()


# TODO: Possibly add transforms to images
