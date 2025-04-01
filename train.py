"""Implement the .train function."""

import torch

from collections import defaultdict

# ------ Local imports ------ #
from scheduler import GradualWarmupScheduler


def train(model, loss_fn, trainloader, validloader,
          optimizer='adam', scheduler='linear',
          epochs=120, lr=0.1, weight_decay=5e-4,
          validate=100, warmup=False,
          setup=dict(dtype=torch.float, device=torch.device('cpu'))):
    """Run the main interface. Train a network with specifications from the Strategy object."""
    stats = defaultdict(list)
    scheduler_typ = scheduler
    optimizer, scheduler = set_optimizer(
        model, optimizer, lr, weight_decay, scheduler_typ, warmup)

    for epoch in range(epochs):
        model.train()
        step(model, loss_fn, trainloader, optimizer,
             scheduler, scheduler_typ, setup, stats)

        if epoch % validate == 0 or epoch == (epochs - 1):
            model.eval()
            validate(model, loss_fn, validloader, setup, stats)
            # Print information about loss and accuracy
            print_status(epoch, loss_fn, optimizer, stats)

    return stats


def step(model, loss_fn, dataloader, optimizer, scheduler, scheduler_typ, setup, stats):
    """Step through one epoch."""
    epoch_loss, epoch_metric = 0, 0
    for batch, (inputs, targets) in enumerate(dataloader):
        # Prep Mini-Batch
        optimizer.zero_grad()

        # Transfer to GPU
        inputs = inputs.to(**setup)
        targets = targets.to(device=setup['device'], non_blocking=False)

        # Get loss
        outputs = model(inputs)
        loss, _, _ = loss_fn(outputs, targets)

        epoch_loss += loss.item()

        loss.backward()
        optimizer.step()

        metric, name, _ = loss_fn.metric(outputs, targets)
        epoch_metric += metric.item()

        if scheduler_typ == 'cyclic':
            scheduler.step()
    if scheduler_typ == 'linear':
        scheduler.step()

    stats['train_losses'].append(epoch_loss / (batch + 1))
    stats['train_' + name].append(epoch_metric / (batch + 1))


def validate(model, loss_fn, dataloader, setup, stats):
    """Validate model effectiveness of val dataset."""
    epoch_loss, epoch_metric = 0, 0
    with torch.no_grad():
        for batch, (inputs, targets) in enumerate(dataloader):
            # Transfer to GPU
            inputs = inputs.to(**setup)
            targets = targets.to(device=setup['device'], non_blocking=False)

            # Get loss and metric
            outputs = model(inputs)
            loss, _, _ = loss_fn(outputs, targets)
            metric, name, _ = loss_fn.metric(outputs, targets)

            epoch_loss += loss.item()
            epoch_metric += metric.item()

    stats['valid_losses'].append(epoch_loss / (batch + 1))
    stats['valid_' + name].append(epoch_metric / (batch + 1))


def set_optimizer(model, optimizer, lr, weight_decay, scheduler, warmup):
    """Build model optimizer and scheduler.

    The linear scheduler drops the learning rate in intervals.
    # Example: epochs=160 leads to drops at 60, 100, 140.
    """
    if optimizer == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9,
                                    weight_decay=weight_decay, nesterov=True)
    elif optimizer == 'AdamW':
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=lr, weight_decay=weight_decay)

    if scheduler == 'linear':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                         milestones=[120 // 2.667, 120 // 1.6,
                                                                     120 // 1.142], gamma=0.1)
        # Scheduler is fixed to 120 epochs so that calls with fewer epochs are equal in lr drops.

    if warmup:
        scheduler = GradualWarmupScheduler(
            optimizer, multiplier=10, total_epoch=10, after_scheduler=scheduler)

    return optimizer, scheduler


def print_status(epoch, loss_fn, optimizer, stats):
    """Print basic console printout every validation epochs."""
    current_lr = optimizer.param_groups[0]['lr']
    name, format = loss_fn.metric()
    print(f'Epoch: {epoch}| lr: {current_lr:.4f} | '
          f'Train loss is {stats["train_losses"][-1]:6.4f}, Train {name}: {stats["train_" + name][-1]:{format}} | '
          f'Val loss is {stats["valid_losses"][-1]:6.4f}, Val {name}: {stats["valid_" + name][-1]:{format}} |')
