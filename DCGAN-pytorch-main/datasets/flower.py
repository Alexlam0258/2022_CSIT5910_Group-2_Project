import torch
import torchvision

from configs.parameters import dataroot, image_size, batch_size, workers

dataset = torchvision.datasets.ImageFolder(root=dataroot,
                           transform=torchvision.transforms.Compose([
                               torchvision.transforms.Resize(image_size),
                               torchvision.transforms.CenterCrop(image_size),
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]))

flower_dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=workers 
)
