import matplotlib.pyplot as plt
import numpy as np
import math
import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
from torch.autograd import Variable

from torchvision import datasets
from torchvision.transforms import transforms

from mine.models.gan import GAN

from mine.datasets import FunctionDataset, MultivariateNormalDataset
from mine.models.layers import ConcatLayer, CustomSequential

import pytorch_lightning as pl
from pytorch_lightning import Trainer
import mine.utils.helpers as utils

from tqdm.auto import tqdm

torch.autograd.set_detect_anomaly(True)

EPS = 1e-6

device = 'cuda' if torch.cuda.is_available() else 'cpu'
datadir = "/scratch/network/ls1546/mine-pytorch/data/"

class EMALoss(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, running_ema):
        ctx.save_for_backward(input, running_ema)
        input_log_sum_exp = input.exp().mean().log()

        return input_log_sum_exp

    @staticmethod
    def backward(ctx, grad_output):
        input, running_mean = ctx.saved_tensors
        # print("input nan/inf: ", torch.sum(torch.isfinite(input)))
        # print("run mean: ", running_mean)
        # print("grad_output: ", grad_output)
        grad = grad_output * input.exp().detach() / \
            (running_mean + EPS) / input.shape[0]
        return grad, None


def ema(mu, alpha, past_ema):
    return alpha * mu + (1.0 - alpha) * past_ema


def ema_loss(x, running_mean, alpha):
    t_exp = torch.sum(torch.exp(x), 0)/x.shape[0].detach()
    # print(running_mean)
    if running_mean == 0:
        running_mean = t_exp
    else:
        running_mean = ema(t_exp, alpha, running_mean.item())
    
    t_log = EMALoss.apply(x, running_mean)

    # Recalculate ema

    return t_log, running_mean


class Mine(nn.Module):
    def __init__(self, T, loss='mine', alpha=0.01, method=None, device=device):
        super().__init__()
        self.running_mean = 0
        self.loss = loss
        self.alpha = alpha
        self.method = method
        self.device = device

        if method == 'concat':
            if isinstance(T, nn.Sequential):
                self.T = CustomSequential(ConcatLayer(), *T)
            else:
                self.T = CustomSequential(ConcatLayer(), T)
        else:
            self.T = T.to(device)

    def forward(self, x, z, z_marg=None):
        if z_marg is None:
            z_marg = z[torch.randperm(x.shape[0])]

        x = x.to(device)
        z = z.to(device)
        z_marg = z_marg.to(device)

        t = self.T(x, z).mean()
        t_marg = self.T(x, z_marg)

        if self.loss in ['mine']:
            second_term, self.running_mean = ema_loss(
                t_marg, self.running_mean, self.alpha)
        elif self.loss in ['fdiv']:
            second_term = torch.exp(t_marg - 1).mean()
        elif self.loss in ['mine_biased']:
            second_term = torch.logsumexp(
                t_marg, 0) - math.log(t_marg.shape[0])

        return -t + second_term

    def mi(self, x, z, z_marg=None):
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()
        if isinstance(z, np.ndarray):
            z = torch.from_numpy(z).float()
            
        with torch.no_grad():
            mi = -self.forward(x, z, z_marg)
        return mi

    def optimize(self, X, Y, epochs, batch_size, lam, name, opt=None):
        threshold = 10.0
        best_loss = np.inf
        count = 0
        batch_num = 0

        if opt is None:
            opt = torch.optim.AdamW(self.parameters(), lr=1e-4, weight_decay=lam)
        loss_list = []
        for epoch in (pbar := tqdm(range(1, epochs + 1))):
            mu_mi = 0
            for x, y in utils.batch(X, Y, batch_size):
                if batch_num == 0:
                    try:
                        x, y = x.to(self.device), y.to(self.device)
                        # print(x.shape)
                        opt.zero_grad()
                        loss = self.forward(x, y)
                        loss_list.append(loss)
                        loss.backward()
                        opt.step()

                        mu_mi -= loss.item()
                    except:
                        print("NAN/INF")
                        # np.save(f"{datadir}{name}_batchx{batch_num}.npy", x.detach().cpu().numpy())
                        # np.save(f"{datadir}{name}_batchy{batch_num}.npy", y.detach().cpu().numpy())
                        continue
                else:
                    x, y = x.to(self.device), y.to(self.device)
                    # print(x.shape)
                    opt.zero_grad()
                    loss = self.forward(x, y)
                    loss_list.append(loss)
                    loss.backward()
                    opt.step()

                    mu_mi -= loss.item()
                batch_num += 1

            pbar.set_description(f"epoch: {epoch}, mu_mi: {mu_mi:4f}")
            curr_loss = loss_list[epoch].detach().cpu().numpy()

            # checkpoint the model if loss below threshold, save best model and avg checkpointed model
            if curr_loss < threshold:
                if curr_loss < best_loss:
                    best_loss = curr_loss
                    torch.save(self.T, f"{datadir}{name}_best_mine.pth")
                    np.save(f"{datadir}{name}_best_loss.npy", best_loss)
                
                torch.save(self.T, f"{datadir}{name}_ckpt_mine{count}.pth")
                count += 1

        final_mi = self.mi(X, Y)
        print(f"Final MI: {final_mi}")

        # Average the weights of the checkpointed models
        for i in range(count):
            model = torch.load(f"{datadir}{name}_ckpt_mine{i}.pth")
            weights = model.fc1x.weight.detach().cpu().numpy()[0]
            if i == 0:
                avg_weights = weights
            avg_weights = avg_weights + weights
        avg_weights = avg_weights / count
        # model.fc1x.weight = nn.Parameter(torch.tensor(avg_weights))
        np.save(f"{datadir}avg_weights_{name}.npy", avg_weights)

        return final_mi, loss_list


class T(nn.Module):
    def __init__(self, x_dim, z_dim):
        super().__init__()
        self.layers = CustomSequential(ConcatLayer(), nn.Linear(x_dim + z_dim, 400),
                                       nn.ReLU(),
                                       nn.Linear(400, 400),
                                       nn.ReLU(),
                                       nn.Linear(400, 400),
                                       nn.ReLU(),
                                       nn.Linear(400, 1))

    def forward(self, x, z):
        return self.layers(x, z)


class MutualInformationEstimator(pl.LightningModule):
    def __init__(self, x_dim, z_dim, loss='mine', **kwargs):
        super().__init__()
        self.x_dim = x_dim
        self.T = CustomSequential(ConcatLayer(), nn.Linear(x_dim + z_dim, 100), nn.ReLU(),
                                  nn.Linear(100, 100), nn.ReLU(), nn.Linear(100, 1))

        self.energy_loss = Mine(self.T, loss=loss, alpha=kwargs['alpha'])

        self.kwargs = kwargs

        self.train_loader = kwargs.get('train_loader')
        self.test_loader = kwargs.get('test_loader')

    def forward(self, x, z):
        if self.on_gpu:
            x = x.cuda()
            z = z.cuda()

        return self.energy_loss(x, z)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.kwargs['lr'], weight_decay=lam)

    def training_step(self, batch, batch_idx):

        x, z = batch

        if self.on_gpu:
            x = x.cuda()
            z = z.cuda()

        loss = self.energy_loss(x, z)
        mi = -loss
        tensorboard_logs = {'loss': loss, 'mi': mi}
        tqdm_dict = {'loss_tqdm': loss, 'mi': mi}

        return {
            **tensorboard_logs, 'log': tensorboard_logs, 'progress_bar': tqdm_dict
        }

    def test_step(self, batch, batch_idx):
        x, z = batch
        loss = self.energy_loss(x, z)

        return {
            'test_loss': loss, 'test_mi': -loss
        }

    def test_end(self, outputs):
        avg_mi = torch.stack([x['test_mi']
                              for x in outputs]).mean().detach().cpu().numpy()
        tensorboard_logs = {'test_mi': avg_mi}

        self.avg_test_mi = avg_mi
        return {'avg_test_mi': avg_mi, 'log': tensorboard_logs}

    def train_dataloader(self):
        if self.train_loader:
            return self.train_loader

        train_loader = torch.utils.data.DataLoader(
            FunctionDataset(self.kwargs['N'], self.x_dim,
                            self.kwargs['sigma'], self.kwargs['f']),
            batch_size=self.kwargs['batch_size'], shuffle=True)
        return train_loader

    def test_dataloader(self):
        if self.test_loader:
            return self.train_loader

        test_loader = torch.utils.data.DataLoader(
            FunctionDataset(self.kwargs['N'], self.x_dim,
                            self.kwargs['sigma'], self.kwargs['f']),
            batch_size=self.kwargs['batch_size'], shuffle=True)
        return test_loader


def build_dist(rho):
    mu = torch.tensor([0.0, 0.0])
    cov = torch.tensor([[1, rho], [rho, 1]])
    dist = MultivariateNormal(mu, cov)
    return dist


def function_experiment():
    N = 3000
    lr = 1e-4
    batch_size = 256
    epochs = 200

    def f1(x): return x
    def f2(x): return x**3
    def f3(x): return torch.sin(x)
    sigmas = torch.linspace(0, 0.9, 10)
    fs = [f1, f2, f3]
    dim = 2

    res = []
    for sigma in sigmas:
        for ix, f in enumerate(fs):
            print(f"Experiment: {ix + 1}, Sigma: {sigma}...")

            kwargs = {
                'N': N,
                'sigma': sigma,
                'f': f,
                'lr': lr,
                'batch_size': batch_size
            }

            model = MutualInformationEstimator(
                dim, dim, loss='mine', **kwargs).to(device)
            trainer = Trainer(max_epochs=epochs,
                              early_stop_callback=False, gpus=1)
            trainer.fit(model)
            trainer.test()

            # Append result
            res.append([ix, sigma, model.avg_test_mi])

    res = np.array(res)
    Z = res[:, -1].reshape((len(sigmas), len(fs))).T
    plt.figure()
    plt.imshow(Z, cmap='Blues')
    plt.show()


def rho_experiment():
    dim = 20
    N = 3000
    lr = 1e-3
    epochs = 100
    batch_size = 128

    x_dim = dim
    z_dim = dim

    steps = 20
    rhos = np.linspace(-0.99, 0.99, steps)
    res = []

    # Rho Experiment
    for rho in rhos:
        train_loader = torch.utils.data.DataLoader(
            MultivariateNormalDataset(N, dim, rho), batch_size=batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(
            MultivariateNormalDataset(N, dim, rho), batch_size=batch_size, shuffle=True)

        true_mi = train_loader.dataset.true_mi

        kwargs = {
            'lr': lr,
            'batch_size': batch_size,
            'train_loader': train_loader,
            'test_loader': test_loader,
            'alpha': 1.0
        }

        model = MutualInformationEstimator(
            dim, dim, loss='mine_biased', **kwargs).to(device)
        trainer = Trainer(max_epochs=epochs, early_stop_callback=False, gpus=1)
        trainer.fit(model)
        trainer.test()

        print("True_mi {}".format(true_mi))
        print("MINE {}".format(model.avg_test_mi))
        res.append((rho, model.avg_test_mi, true_mi))

    res = np.array(res)
    plt.figure()
    plt.plot(res[:, 0], res[:, 1], label='MINE')
    plt.plot(res[:, 0], res[:, 2], linestyle='--', label='True MI')
    plt.legend()
    plt.show()


def gan_experiment():

    batch_size = 256
    kwargs = {}

    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=batch_size, shuffle=True, **kwargs)

    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])),
        batch_size=batch_size, shuffle=True, **kwargs)

    img, label = next(iter(train_loader))

    output_dim = 28*28
    input_dim = 100
    lr = 2e-3
    print_every = 100

    mi_model = T(output_dim, 1)

    mi_estimator = Mine(mi_model, loss='mine').to(device)
    opt_mi = torch.optim.Adam(mi_estimator.parameters(), lr=lr)

    model = GAN(input_dim, output_dim, conditional_dim=1,
                mi_estimator=mi_estimator, device=device, __lambda__=0.0).to(device)

    epochs = 100
    opt_g = torch.optim.Adam(model.parameters(), lr=lr)
    opt_d = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        for ix, (img, label) in enumerate(train_loader):

            if device == 'cuda':
                label = label.float().cuda()
                img = img.cuda()

            d_loss, generator_loss = model.loss_fn(
                img, opt_g, opt_d, opt_mi, conditional=label)

            if ix % print_every == 0:
                prct = (ix + 2) * batch_size/(batch_size * len(train_loader))
                print(
                    f"Epoch {epoch} [{(ix + 2) * batch_size}/{batch_size * len(train_loader)}] [{100*prct:.3}%] Loss (d/g): [{d_loss.item():.3}/{generator_loss.item():.3}]")


if __name__ == '__main__':
    rho_experiment()
    # function_experiment()
    # gan_experiment()
