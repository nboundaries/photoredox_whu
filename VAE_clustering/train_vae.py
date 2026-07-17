"""Train a PyTorch VAE on reactant or product feature matrices."""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

try:
    from .utils import ensure_dir, load_numeric_matrix, set_random_seed, write_json
except ImportError:
    from utils import ensure_dir, load_numeric_matrix, set_random_seed, write_json


def weights_init_uniform_rule(module):
    if module.__class__.__name__.find("Linear") != -1:
        n = module.in_features
        y = 1.0 / np.sqrt(n)
        module.weight.data.uniform_(-y, y)
        module.bias.data.fill_(0)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a VAE and export latent mean vectors.")
    parser.add_argument("--dataset", required=True, help="dataset.npy or dataset.csv from build_dataset.py.")
    parser.add_argument("--entity", required=True, choices=["reactant", "product"], help="Entity being trained.")
    parser.add_argument("--out-dir", required=True, help="Output directory.")
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--latent-dim", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=50)
    parser.add_argument("--hidden-dim2", type=int, default=12)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    import torch
    import torch.nn.functional as F
    from torch import nn, optim
    from torch.autograd import Variable
    from torch.utils.data import DataLoader

    class Autoencoder(nn.Module):
        def __init__(self, input_dim: int, hidden_dim: int = 50, hidden_dim2: int = 12, latent_dim: int = 32):
            super().__init__()
            self.linear1 = nn.Linear(input_dim, hidden_dim)
            self.lin_bn1 = nn.BatchNorm1d(hidden_dim)
            self.linear2 = nn.Linear(hidden_dim, hidden_dim2)
            self.lin_bn2 = nn.BatchNorm1d(hidden_dim2)
            self.linear3 = nn.Linear(hidden_dim2, hidden_dim2)
            self.lin_bn3 = nn.BatchNorm1d(hidden_dim2)
            self.fc1 = nn.Linear(hidden_dim2, latent_dim)
            self.bn1 = nn.BatchNorm1d(latent_dim)
            self.fc21 = nn.Linear(latent_dim, latent_dim)
            self.fc22 = nn.Linear(latent_dim, latent_dim)
            self.fc3 = nn.Linear(latent_dim, latent_dim)
            self.fc_bn3 = nn.BatchNorm1d(latent_dim)
            self.fc4 = nn.Linear(latent_dim, hidden_dim2)
            self.fc_bn4 = nn.BatchNorm1d(hidden_dim2)
            self.linear4 = nn.Linear(hidden_dim2, hidden_dim2)
            self.lin_bn4 = nn.BatchNorm1d(hidden_dim2)
            self.linear5 = nn.Linear(hidden_dim2, hidden_dim)
            self.lin_bn5 = nn.BatchNorm1d(hidden_dim)
            self.linear6 = nn.Linear(hidden_dim, input_dim)
            self.lin_bn6 = nn.BatchNorm1d(input_dim)
            self.relu = nn.ReLU()

        def encode(self, x):
            lin1 = self.relu(self.lin_bn1(self.linear1(x)))
            lin2 = self.relu(self.lin_bn2(self.linear2(lin1)))
            lin3 = self.relu(self.lin_bn3(self.linear3(lin2)))
            fc1 = F.relu(self.bn1(self.fc1(lin3)))
            return self.fc21(fc1), self.fc22(fc1)

        def reparameterize(self, mu, logvar):
            if self.training:
                std = logvar.mul(0.5).exp_()
                eps = Variable(std.data.new(std.size()).normal_())
                return eps.mul(std).add_(mu)
            return mu

        def decode(self, z):
            fc3 = self.relu(self.fc_bn3(self.fc3(z)))
            fc4 = self.relu(self.fc_bn4(self.fc4(fc3)))
            lin4 = self.relu(self.lin_bn4(self.linear4(fc4)))
            lin5 = self.relu(self.lin_bn5(self.linear5(lin4)))
            return self.lin_bn6(self.linear6(lin5))

        def forward(self, x):
            mu, logvar = self.encode(x)
            z = self.reparameterize(mu, logvar)
            return self.decode(z), mu, logvar

    class VAELoss(nn.Module):
        def __init__(self):
            super().__init__()
            self.mse_loss = nn.MSELoss(reduction="sum")

        def forward(self, x_recon, x, mu, logvar):
            loss_mse = self.mse_loss(x_recon, x)
            loss_kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            return loss_mse + loss_kld

    out_dir = ensure_dir(args.out_dir)
    set_random_seed(args.seed)

    data = load_numeric_matrix(args.dataset).astype(np.float32)
    if data.shape[0] < 2:
        raise ValueError("VAE training requires at least 2 valid samples because BatchNorm is used.")
    input_dim = int(data.shape[1])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    drop_last = data.shape[0] > args.batch_size and data.shape[0] % args.batch_size == 1
    trainloader = DataLoader(dataset=torch.from_numpy(data), batch_size=args.batch_size, shuffle=True, drop_last=drop_last)

    model = Autoencoder(input_dim, args.hidden_dim, args.hidden_dim2, args.latent_dim).to(device)
    model.apply(weights_init_uniform_rule)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = VAELoss()
    logs = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        for batch in trainloader:
            batch = batch.to(device)
            optimizer.zero_grad()
            recon_batch, mu, logvar = model(batch)
            loss = loss_fn(recon_batch, batch, mu, logvar)
            loss.backward()
            train_loss += loss.item()
            optimizer.step()
        average_loss = train_loss / len(trainloader.dataset)
        logs.append({"epoch": epoch, "average_loss": average_loss})
        if epoch == 1 or epoch % 50 == 0 or epoch == args.epochs:
            print(f"{args.entity} epoch {epoch}: average loss {average_loss:.4f}")

    model.eval()
    latent_batches = []
    eval_loader = DataLoader(dataset=torch.from_numpy(data), batch_size=args.batch_size, shuffle=False)
    with torch.no_grad():
        for batch in eval_loader:
            batch = batch.to(device)
            mu, _ = model.encode(batch)
            latent_batches.append(mu.cpu())
    latent = torch.cat(latent_batches, dim=0).numpy()

    torch.save(model.state_dict(), out_dir / "vae_model.pt")
    np.save(out_dir / "latent.npy", latent)
    pd.DataFrame(logs).to_csv(out_dir / "train_log.csv", index=False)
    write_json(
        out_dir / "vae_config.json",
        {
            "entity": args.entity,
            "input_dim": input_dim,
            "latent_dim": args.latent_dim,
            "hidden_dim": args.hidden_dim,
            "hidden_dim2": args.hidden_dim2,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "seed": args.seed,
            "device": str(device),
            "latent_shape": list(latent.shape),
        },
    )
    print(f"Saved latent vectors with shape {latent.shape} to {out_dir / 'latent.npy'}.")


if __name__ == "__main__":
    main()
