import time
import torch
import numpy as np

# ------ Local imports ------ #
import consts
from data import load_data, num_classes
from model import construct_model
from plot import display_batch
from reconstructor import GradientReconstructor
from options import options
from utils import system_startup


def extract_ground_truth(dataloader, args, setup):
    if args.batch_size == 1:
        target_id = np.random.randint(len(dataloader.dataset))
        gt_data, gt_label = dataloader.dataset[target_id]
        gt_data, gt_label = (
            gt_data.unsqueeze(0).to(**setup),
            torch.as_tensor((gt_label,), device=setup["device"]),
        )
        data_shape = (3, gt_data.shape[2], gt_data.shape[3])
    else:
        gt_data, gt_label = [], []
        target_id = np.random.randint(len(dataloader.dataset))
        while len(gt_label) < args.batch_size:
            data, label = dataloader.dataset[target_id]
            target_id += 1
            if label not in gt_label:
                gt_label.append(torch.as_tensor(
                    (label,), device=setup["device"]))
                gt_data.append(data.to(**setup))
        gt_data = torch.stack(gt_data)
        gt_label = torch.cat(gt_label)
        data_shape = (3, gt_data.shape[2], gt_data.shape[3])
    return gt_data, gt_label, data_shape


if __name__ == "__main__":
    args = options().parse_args()

    setup = system_startup()
    start_time = time.time()

    loss_fn, trainloader, validloader = load_data(
        args.dataset, batch_size=args.batch_size, data_path=args.data_path)

    # print some data
    print(f"Training set size: {len(trainloader.dataset)}")
    print(f"Validation set size: {len(validloader.dataset)}")

    dm = torch.as_tensor(
        getattr(consts, f"{args.dataset.lower()}_mean"), **setup)[:, None, None]
    ds = torch.as_tensor(
        getattr(consts, f"{args.dataset.lower()}_std"), **setup)[:, None, None]

    num_classes = num_classes(args.dataset)

    model, model_seed = construct_model(
        args.model, num_classes, num_channels=3)
    model.to(**setup)
    model.eval()

    # Sanity check: run the model.
    # training_stats = defaultdict(list)
    # validate(model, loss_fn, validloader, setup, training_stats)
    # name, fmt = loss_fn.metric()
    # print(f'Val loss is {training_stats["valid_losses"][-1]:6.4f},\
    #       Val {name}: {training_stats["valid_" + name][-1]:{fmt}}.')

    # Extracting original data and labels.
    gt_data, gt_label, shape = extract_ground_truth(validloader, args, setup)

    print(f"Ground truth data shape: {gt_data.shape}")
    print(f"Ground truth label shape: {gt_label.shape}")

    display_batch(gt_data, denormalize=True, mean=dm, std=ds)

    # Run reconstruction.
    model.zero_grad()
    target_loss, _, _ = loss_fn(model(gt_data), gt_label)
    gt_gradient = torch.autograd.grad(target_loss, model.parameters())
    gt_gradient = [grad.detach() for grad in gt_gradient]
    gt_gradnorm = torch.stack([g.norm() for g in gt_gradient]).mean()

    print(f"Full gradient norm is {gt_gradnorm:e}.")

    gt_label = gt_label if not args.reconstruct_label else None

    reconstructor = GradientReconstructor(model, mean_std=(dm, ds), batch_size=args.batch_size, optimizer=args.optimizer,
                                          max_iter=args.max_iter, lr_decay=args.lr_decay, cost_fn=args.cost_fn,
                                          idlg=args.idlg)
    data, stats = reconstructor.reconstruct(
        gt_gradient, gt_label, trials=args.trials, shape=shape)

    display_batch(data, denormalize=True, mean=dm, std=ds)

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time:.2f} seconds.")
