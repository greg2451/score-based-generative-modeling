from datetime import datetime

from utils import loss_fn, marginal_prob_std_fn, load_model, load_dataset
from train import train


num_epochs = 5
batch_size = 128
learning_rate = 1e-5

run_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

if __name__ == "__main__":
    score_model = load_model(run_datetime=run_datetime, pretrained_path=None)

    dataset = load_dataset(100)

    train(
        score_model,
        dataset,
        num_epochs,
        batch_size,
        learning_rate,
        loss_fn,
        marginal_prob_std_fn,
        run_datetime=run_datetime,
    )
