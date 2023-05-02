from math import log
from pathlib import Path
from typing import Optional

import torch
import functools

from torchvision.datasets import FashionMNIST
from torchvision.transforms import transforms

from model import ScoreNet

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def marginal_prob_std(t, sigma):
    """Compute the mean and standard deviation of $p_{0t}(x(t) | x(0))$.

    Args:
      t: A vector of time steps.
      sigma: The sigma in our SDE.

    Returns:
      The standard deviation.
    """
    if type(t) != torch.Tensor:
        t = torch.tensor(t, device=DEVICE)
    return torch.sqrt((sigma ** (2 * t.to(DEVICE)) - 1.0) / 2.0 / log(sigma))


def diffusion_coeff(t, sigma):
    """Compute the diffusion coefficient of our SDE.

    Args:
      t: A vector of time steps.
      sigma: The sigma in our SDE.

    Returns:
      The vector of diffusion coefficients.
    """
    if type(t) != torch.Tensor:
        t = torch.tensor(t, device=DEVICE)
    return sigma**t


sigma = 25.0
marginal_prob_std_fn = functools.partial(marginal_prob_std, sigma=sigma)
diffusion_coeff_fn = functools.partial(diffusion_coeff, sigma=sigma)


def loss_fn(model, x, marginal_prob_std, eps=1e-5):
    """The loss function for training score-based generative models.

    Args:
      model: A PyTorch model instance that represents a
        time-dependent score-based model.
      x: A mini-batch of training data.
      marginal_prob_std: A function that gives the standard deviation of
        the perturbation kernel.
      eps: A tolerance value for numerical stability.
    """
    random_t = torch.rand(x.shape[0], device=x.device) * (1.0 - eps) + eps
    z = torch.randn_like(x)
    std = marginal_prob_std(random_t)
    perturbed_x = x + z * std[:, None, None, None]
    score = model(perturbed_x, random_t)
    return torch.mean(
        torch.sum((score * std[:, None, None, None] + z) ** 2, dim=(1, 2, 3))
    )


def load_model(
    run_datetime: Optional[str] = None, pretrained_path: Optional[str] = None
):
    score_model = torch.nn.DataParallel(
        ScoreNet(marginal_prob_std=marginal_prob_std_fn)
    )

    if pretrained_path is not None:
        weights = torch.load(pretrained_path, map_location=DEVICE)
        score_model.load_state_dict(weights)
    if run_datetime is not None:
        (run_dir := Path(f"runs/{run_datetime}")).mkdir(exist_ok=True, parents=True)
        torch.save(score_model.state_dict(), run_dir / "start.pt")

    return score_model


def load_dataset(num_samples: Optional[int] = None):
    dataset = FashionMNIST(
        "./data/", train=True, transform=transforms.ToTensor(), download=True
    )
    if num_samples is not None:
        dataset = torch.utils.data.Subset(
            dataset, range(min(len(dataset), num_samples))
        )
    return dataset
