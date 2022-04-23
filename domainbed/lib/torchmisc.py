import torch


def dataloader(dataset, batch_size=4, num_workers=0, valid=True, gpu=True):
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=not valid,
        num_workers=int(num_workers),
        drop_last=not valid,
        pin_memory=gpu,
    )
