import math
import matplotlib.pyplot as plt
from torchvision import transforms
import torch


def display_batch_iterative(data, fig=None, axes=None, denormalize=False, mean=None, std=None):
    """Updates images in a single window dynamically without opening multiple tabs.

    Args:
        data: Tensor of shape (batch_size, 3, H, W).
        fig: (optional) Matplotlib figure to reuse.
        axes: (optional) Matplotlib axes to reuse.
        denormalize (bool, optional): If True, denormalize the image tensor. Defaults to False.
        mean (tuple, optional): Mean for denormalization.  Required if denormalize is True.
        std (tuple, optional): Std for denormalization. Required if denormalize is True.

    Returns:
        fig, axes: The figure and axes for reuse.
    """
    if not isinstance(data, torch.Tensor):
        raise TypeError("Input data must be a torch.Tensor")
    if data.ndim != 4:
        raise ValueError(
            "Input data must be a 4D tensor (batch_size, 3, H, W)")

    batch_size = data.shape[0]
    max_per_row = 10
    ncols = min(max_per_row, batch_size)
    nrows = math.ceil(batch_size / max_per_row)

    if fig is None or axes is None:
        plt.ion()  # Enable interactive mode
        fig, axes = plt.subplots(
            nrows, ncols, figsize=(ncols * 1.5, nrows * 1.5))

        if nrows == 1:
            axes = [axes]  # make it a list with one row
        if ncols == 1:
            axes = [[ax] for ax in axes]

    try:
        axes = axes.tolist()
    except:
        pass

    for idx in range(batch_size):
        row = idx // max_per_row
        col = idx % max_per_row
        ax = axes[row][col]

        # Denormalize if required
        if denormalize:
            if mean is None or std is None:
                raise ValueError(
                    "Mean and std must be provided for denormalization.")
            img = data[idx].cpu() * torch.tensor(std).view(3, 1, 1) + \
                torch.tensor(mean).view(3, 1, 1)
            # Ensure values are in the [0, 1] range
            img = torch.clamp(img, 0, 1)
        else:
            img = data[idx].cpu()

        pil_img = transforms.ToPILImage()(img)  # Convert tensor to PIL Image
        ax.imshow(pil_img)
        ax.axis('off')

    # Hide any unused axes if total_images doesn't fill the grid completely
    for idx in range(batch_size, nrows * ncols):
        row = idx // max_per_row
        col = idx % max_per_row
        axes[row][col].axis('off')

    fig.canvas.draw()  # Redraw the figure
    fig.canvas.flush_events()  # Process UI events

    return fig, axes  # Return the figure and axes for reuse


def display_batch(data, denormalize=False, mean=None, std=None):
    """Displays the target images in a grid.

    Args:
        data: Tensor of shape (batch_size, 3, H, W).
        denormalize (bool, optional): If True, denormalize the image tensor. Defaults to False.
        mean (tuple, optional): Mean for denormalization.  Required if denormalize is True.
        std (tuple, optional): Std for denormalization. Required if denormalize is True.
    """
    if not isinstance(data, torch.Tensor):
        raise TypeError("Input data must be a torch.Tensor")
    if data.ndim != 4:
        raise ValueError(
            "Input data must be a 4D tensor (batch_size, 3, H, W)")

    batch_size = data.shape[0]
    max_per_row = 10
    ncols = min(max_per_row, batch_size)
    nrows = math.ceil(batch_size / max_per_row)
    plt.figure(figsize=(ncols * 1.5, nrows * 1.5))

    for idx in range(batch_size):
        plt.subplot(nrows, ncols, idx + 1)

        # Denormalize if required
        if denormalize:
            if mean is None or std is None:
                raise ValueError(
                    "Mean and std must be provided for denormalization.")
            img = data[idx].cpu() * torch.tensor(std).view(3, 1, 1) + \
                torch.tensor(mean).view(3, 1, 1)
            # Ensure values are in the [0,1] range
            img = torch.clamp(img, 0, 1)
        else:
            img = data[idx].cpu()

        pil_img = transforms.ToPILImage()(img)
        plt.imshow(pil_img)
        plt.axis('off')

    plt.tight_layout()
    plt.show()
