import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_dataloaders(split_dir, batch_size=64, num_workers=0):
    """
    Applies transformations and returns DataLoaders for train, validation, and test splits.
    
    Args:
        split_dir (str): The root directory containing 'train', 'val', and 'test' subfolders.
        batch_size (int): Number of images per batch.
        num_workers (int): Subprocesses to use for data loading.
        
    Returns:
        trainloader, valloader, testloader, classes
    """
    # Define transformations (Resize, ToTensor, Normalize)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Instantiate datasets
    train_dataset = datasets.ImageFolder(
        root=os.path.join(split_dir, "train"),
        transform=transform
    )
    
    val_dataset = datasets.ImageFolder(
        root=os.path.join(split_dir, "val"),
        transform=transform
    )
    
    test_dataset = datasets.ImageFolder(
        root=os.path.join(split_dir, "test"),
        transform=transform
    )

    # Create DataLoaders
    trainloader = DataLoader(train_dataset, batch_size=batch_size,
                             shuffle=True, num_workers=num_workers, pin_memory=True)
    valloader   = DataLoader(val_dataset, batch_size=batch_size,
                             shuffle=False, num_workers=num_workers, pin_memory=True)
    testloader  = DataLoader(test_dataset, batch_size=batch_size,
                             shuffle=False, num_workers=num_workers, pin_memory=True)

    return trainloader, valloader, testloader, train_dataset.classes