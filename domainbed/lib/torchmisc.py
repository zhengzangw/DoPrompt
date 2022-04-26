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


class GradientReverse(torch.autograd.Function):
    scale = 1.0

    @staticmethod
    def forward(ctx, x):
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return GradientReverse.scale * grad_output.neg()


def grad_reverse(x, scale=1.0):
    GradientReverse.scale = scale
    return GradientReverse.apply(x)
