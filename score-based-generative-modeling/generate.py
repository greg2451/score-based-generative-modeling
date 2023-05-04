from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt

from torchvision.utils import make_grid
import numpy as np

from sampling import euler_maruyama_sampler, pc_sampler, ode_sampler
from utils import load_model, marginal_prob_std_fn, diffusion_coeff_fn

import argparse

parser = argparse.ArgumentParser(
    description="Generate samples from a score-based model."
)

parser.add_argument(
    "--model_path",
    type=str,
    default="pretrained_models/FashionMNIST_UNet.pt",
    help="The path to the pretrained model.",
)

parser.add_argument(
    "--batch_size",
    type=int,
    default=16,
    help="The batch size for generation. It must be a square number.",
)

parser.add_argument(
    "--sampler",
    type=str,
    default="pc_sampler",
    help="The sampler to use for generation. It must be one of 'euler_maruyama_sampler', 'pc_sampler', 'ode_sampler'.",
)

parser.add_argument(
    "--signal_to_noise_ratio",
    type=float,
    default=0.01,
    help="The signal to noise ratio for generation. (Only for 'pc_sampler')",
)

parser.add_argument(
    "--num_steps",
    type=int,
    default=100,
    help="The number of steps for generation.",
)

parser.add_argument(
    "--error_tolerance",
    type=float,
    default=1e-5,
    help="The error tolerance for the ODE solver. (Only for 'ode_sampler')",
)

args = parser.parse_args()


generation_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

if __name__ == "__main__":
    if args.sampler == "pc_sampler":
        kwargs = {"snr": args.signal_to_noise_ratio, "num_steps": args.num_steps}
    elif args.sampler == "ode_sampler":
        kwargs = {"atol": args.error_tolerance, "rtol": args.error_tolerance}
    elif args.sampler == "euler_maruyama_sampler":
        kwargs = {"num_steps": args.num_steps}
    else:
        raise ValueError(
            "The sampler must be one of 'euler_maruyama_sampler', 'pc_sampler', 'ode_sampler'."
        )

    kwargs["batch_size"] = args.batch_size
    print(f"Generating samples using {args.sampler} with {kwargs}...")

    sampler = {
        "euler_maruyama_sampler": euler_maruyama_sampler,
        "pc_sampler": pc_sampler,
        "ode_sampler": ode_sampler,
    }[args.sampler]

    score_model = load_model(pretrained_path=args.model_path)

    samples = sampler(
        score_model,
        marginal_prob_std_fn,
        diffusion_coeff_fn,
        **kwargs,
    )

    samples = samples.clamp(0.0, 1.0)
    sample_grid = make_grid(samples, nrow=int(np.sqrt(args.batch_size)))

    plt.figure(figsize=(6, 6))
    plt.axis("off")
    # Save the generated samples
    file_path = Path("generations") / f"{generation_datetime}.png"
    plt.imsave(file_path, sample_grid.permute(1, 2, 0).cpu().numpy())
    plt.imshow(sample_grid.permute(1, 2, 0).cpu(), vmin=0.0, vmax=1.0)
    plt.show()
