from datetime import datetime

from utils import loss_fn, marginal_prob_std_fn, load_model, load_dataset
from train import train

import argparse


run_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a score-based model.")
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=5,
        help="The number of epochs to train the model for.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=128,
        help="The batch size to use for training.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-5,
        help="The learning rate to use for training.",
    )

    parser.add_argument(
        "--pretrained_path",
        type=str,
        default=None,
        help="The path to the pretrained model.",
    )

    parser.add_argument(
        "--num_samples",
        type=int,
        default=None,
        help="The number of samples to use for training.",
    )

    args = parser.parse_args()
    score_model = load_model(pretrained_path=args.pretrained_path)

    dataset = load_dataset(num_samples=args.num_samples)

    train(
        score_model,
        dataset,
        args.num_epochs,
        args.batch_size,
        args.learning_rate,
        loss_fn,
        marginal_prob_std_fn,
        run_datetime=run_datetime,
    )
