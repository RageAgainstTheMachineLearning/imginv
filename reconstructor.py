from collections import defaultdict
import time
import torch


class GradientReconstructor():
    """Instantiate a reconstruction algorithm."""

    def __init__(self, model, mean_std=(0.0, 1.0), batch_size=1, optimizer='adam', max_iter=100,
                 lr_decay=False, cost_fn='sim', idlg=True):
        """Initialize with algorithm setup."""
        self.model = model
        self.setup = dict(device=next(model.parameters()).device,
                          dtype=next(model.parameters()).dtype)

        self.optimizer = optimizer
        self.max_iter = max_iter
        self.mean_std = mean_std
        self.batch_size = batch_size
        self.lr_decay = lr_decay
        self.cost_fn = cost_fn
        self.idlg = idlg

        self.loss_fn = torch.nn.CrossEntropyLoss(reduction='mean')

    def reconstruct(self, gt_gradient, gt_label=None, trials=1, shape=(3, 32, 32)):
        """Reconstruct data from gradient."""
        start_time = time.time()
        stats = defaultdict(list)

        x = self._init_data(trials, shape)
        scores = torch.zeros(trials)

        if gt_label is None:
            if self.batch_size == 1 and self.idlg:
                # iDLG trick:
                last_weight_min = torch.argmin(
                    torch.sum(gt_gradient[-2], dim=-1), dim=-1)
                gt_label = last_weight_min.detach().reshape((1,)).requires_grad_(False)
            else:
                def loss_fn(pred, labels):
                    labels = torch.nn.functional.softmax(labels, dim=-1)
                    return torch.mean(torch.sum(- labels * torch.nn.functional.log_softmax(pred, dim=-1), 1))
                self.loss_fn = loss_fn
        else:
            assert gt_label.shape[0] == self.batch_size

        try:
            for trial in range(trials):
                x_trial, y_trial = self._run_trial(
                    x[trial], gt_label, gt_gradient)
                x[trial] = x_trial

                scores[trial] = self._score_trial(
                    x_trial, y_trial, gt_gradient)
        except KeyboardInterrupt:
            print('Trial procedure manually interruped.')
            pass

        print('Choosing optimal result ...')
        # guard against NaN/-Inf scores?
        scores = scores[torch.isfinite(scores)]
        optimal_index = torch.argmin(scores)
        print(f'Optimal result score: {scores[optimal_index]:2.4f}')
        stats['score'] = scores[optimal_index].item()
        x_optimal = x[optimal_index]

        print(f'Total time: {time.time()-start_time}.')
        return x_optimal.detach(), stats

    def _init_data(self, trials, shape):
        return torch.randn((trials, self.batch_size, *shape), **self.setup)

    def _run_trial(self, x_trial, y_trial, gt_gradient):

        x_trial.requires_grad = True

        if y_trial is None:
            out = self.model(x_trial)
            y_trial = torch.randn(out.shape[1]).to(
                **self.setup).requires_grad_(True)

            if self.optimizer == 'adam':
                optimizer = torch.optim.Adam([x_trial, y_trial], lr=0.1)
            elif self.optimizer == 'sgd':
                optimizer = torch.optim.SGD(
                    [x_trial, y_trial], lr=0.01, momentum=0.9, nesterov=True)
            elif self.optimizer == 'LBFGS':
                optimizer = torch.optim.LBFGS(
                    [x_trial, y_trial], lr=1, line_search_fn='strong_wolfe')
            else:
                raise ValueError()
        else:
            if self.optimizer == 'adam':
                optimizer = torch.optim.Adam([x_trial], lr=0.1)
            elif self.optimizer == 'sgd':
                optimizer = torch.optim.SGD(
                    [x_trial], lr=0.01, momentum=0.9, nesterov=True)
            elif self.optimizer == 'LBFGS':
                optimizer = torch.optim.LBFGS(
                    [x_trial], lr=1, line_search_fn='strong_wolfe')
            else:
                raise ValueError()

        max_iter = self.max_iter

        dm, ds = self.mean_std

        if self.lr_decay:
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                             milestones=[max_iter // 2.667, max_iter // 1.6,
                                                                         max_iter // 1.142], gamma=0.1)
        try:
            for iteration in range(max_iter):
                closure = self._gradient_closure(
                    optimizer, x_trial, y_trial, gt_gradient)
                rec_loss = optimizer.step(closure)
                if self.lr_decay:
                    scheduler.step()

                with torch.no_grad():
                    # Project to image space.
                    x_trial.data = torch.max(
                        torch.min(x_trial, (1 - dm) / ds), -dm / ds)

                    if (iteration + 1 == max_iter) or iteration % 10 == 0:
                        print(
                            f'It: {iteration}. Rec. loss: {rec_loss.item():2.8f}.')
        except KeyboardInterrupt:
            print(f'Recovery interrupted manually in iteration {iteration}!')
            pass

        return x_trial.detach(), y_trial

    def _gradient_closure(self, optimizer, x_trial, y_trial, gt_gradient):

        def closure():
            optimizer.zero_grad()
            self.model.zero_grad()

            loss = self.loss_fn(self.model(x_trial), y_trial)
            gradient = torch.autograd.grad(
                loss, self.model.parameters(), create_graph=True)
            rec_loss = self._reconstruction_costs(
                gradient, gt_gradient)
            rec_loss.backward()
            return rec_loss
        return closure

    def _score_trial(self, x_trial, y_trial, gt_gradient, ):
        self.model.zero_grad()
        x_trial.grad = None

        loss = self.loss_fn(self.model(x_trial), y_trial)
        gradient = torch.autograd.grad(
            loss, self.model.parameters(), create_graph=False)

        return self._reconstruction_costs(gradient, gt_gradient)

    def _reconstruction_costs(self, grad, gt_grad):
        """Input gradient is given data."""
        indices = torch.arange(len(gt_grad))  # Default indices.
        weights = gt_grad[0].new_ones(len(gt_grad))  # Same weight.

        pnorm = [0, 0]
        costs = 0
        for i in indices:
            if self.cost_fn == 'l2':
                costs += ((grad[i] - gt_grad[i]).pow(2)).sum() * weights[i]
            elif self.cost_fn == 'l1':
                costs += ((grad[i] - gt_grad[i]).abs()).sum() * weights[i]
            elif self.cost_fn == 'max':
                costs += ((grad[i] - gt_grad[i]).abs()).max() * weights[i]
            elif self.cost_fn == 'sim':
                costs -= (grad[i] * gt_grad[i]).sum() * weights[i]
                pnorm[0] += grad[i].pow(2).sum() * weights[i]
                pnorm[1] += gt_grad[i].pow(2).sum() * weights[i]
            elif self.cost_fn == 'simlocal':
                costs += 1 - torch.nn.functional.cosine_similarity(grad[i].flatten(),
                                                                   gt_grad[i].flatten(
                ),
                    0, 1e-10) * weights[i]
        if self.cost_fn == 'sim':
            costs = 1 + costs / pnorm[0].sqrt() / pnorm[1].sqrt()

        return costs
