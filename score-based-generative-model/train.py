from pathlib import Path

from torch.utils.data import DataLoader
from torch.optim import Adam
from torch import save, cuda
from tqdm import tqdm
from numpy import Inf

DEVICE = "cuda" if cuda.is_available() else "cpu"


def train(
    score_model,
    dataset,
    num_epochs,
    batch_size,
    learning_rate,
    loss_function,
    marginal_prob_std_fn,
    run_datetime,
    num_workers=4,
):
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=DEVICE == "cuda",
    )

    (run_dir := Path(f"runs/{run_datetime}")).mkdir(exist_ok=True, parents=True)
    save(score_model.state_dict(), run_dir / "start.pt")
    print(f"Starting training. Run directory: {run_dir}")

    optimizer = Adam(score_model.parameters(), lr=learning_rate)
    min_loss = Inf
    for epoch in tqdm(range(num_epochs), "Training"):
        average_loss = 0.0
        num_items = 0
        for image, _ in data_loader:
            image = image.to(DEVICE)
            loss = loss_function(score_model, image, marginal_prob_std_fn)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            average_loss += loss.item() * image.shape[0]
            num_items += image.shape[0]
        current_loss = average_loss / num_items
        print(f"Average Loss at {epoch=}: {current_loss:5f}")
        if current_loss < min_loss:
            save(score_model.state_dict(), run_dir / "best.pt")
            min_loss = current_loss

    save(score_model.state_dict(), run_dir / "last.pt")
    print(f"Training complete. Best loss: {min_loss:5f}.")
