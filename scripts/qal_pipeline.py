"""Minimal QAL loss with a complete train/eval smoke-test pipeline."""

import argparse
import json
import math
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


def qal_loss(prediction, target, eps=0.001, omega=10.0, attraction_weight=1.0):
    """QAL from Algorithm 1; inputs have shape [batch, points, 3]."""
    distances = torch.cdist(prediction.float(), target.float())
    pred_to_target, nearest_target = distances.min(dim=2)
    target_to_pred = distances.min(dim=1).values

    weight_pred = 1.0 - (torch.sigmoid(omega * (eps - pred_to_target)) - 0.5)
    weight_target = 1.0 - (torch.sigmoid(omega * (eps - target_to_pred)) - 0.5)
    coverage = (
        (weight_pred * pred_to_target).mean(dim=1)
        + (weight_target * target_to_pred).mean(dim=1)
    ).mean()

    covered = torch.zeros(target.shape[:2], dtype=torch.bool, device=target.device)
    covered.scatter_(1, nearest_target, True)
    attraction = (
        (~covered)
        * torch.sigmoid(omega * (eps - target_to_pred))
        * target_to_pred
    ).float().mean()
    return coverage + attraction_weight * attraction


class ToyCompletionData(Dataset):
    """Partial ellipse arcs paired with complete ellipses."""

    def __init__(self, size, seed, input_points=32, output_points=64):
        self.params = torch.rand(size, 6, generator=torch.Generator().manual_seed(seed))
        self.input_points = input_points
        self.output_points = output_points

    def __len__(self):
        return len(self.params)

    def __getitem__(self, index):
        p = self.params[index]
        angle = torch.arange(self.output_points) * (2.0 * math.pi / self.output_points)
        angle = angle + 2.0 * math.pi * p[3]
        target = torch.stack(
            (
                (0.65 + 0.35 * p[0]) * angle.cos() + (p[4] - 0.5) * 0.25,
                (0.45 + 0.35 * p[1]) * angle.sin() + (p[5] - 0.5) * 0.25,
                (0.08 + 0.18 * p[2]) * (2.0 * angle).sin(),
            ),
            dim=1,
        )
        indices = torch.linspace(0, self.output_points // 2 - 1, self.input_points).long()
        return target[indices], target


class TinyCompletionNet(nn.Module):
    def __init__(self, output_points=64):
        super().__init__()
        self.output_points = output_points
        self.encoder = nn.Sequential(nn.Linear(3, 64), nn.ReLU(), nn.Linear(64, 128), nn.ReLU())
        self.decoder = nn.Sequential(nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, output_points * 3))

    def forward(self, partial):
        feature = self.encoder(partial).amax(dim=1)
        return self.decoder(feature).reshape(-1, self.output_points, 3)


def evaluate(model, loader, device, eps, omega, attraction_weight, threshold=0.03):
    model.eval()
    total_qal = total_cd = total_f1 = count = 0.0
    with torch.no_grad():
        for partial, target in loader:
            partial, target = partial.to(device), target.to(device)
            prediction = model(partial)
            distances = torch.cdist(prediction, target)
            pred_nn, target_nn = distances.amin(dim=2), distances.amin(dim=1)
            precision = (pred_nn <= threshold).float().mean(dim=1)
            recall = (target_nn <= threshold).float().mean(dim=1)
            batch_size = len(partial)
            total_qal += qal_loss(prediction, target, eps, omega, attraction_weight).item() * batch_size
            total_cd += (pred_nn.mean() + target_nn.mean()).item() * 0.5 * batch_size
            total_f1 += (2 * precision * recall / (precision + recall).clamp_min(1e-8)).sum().item()
            count += batch_size
    return {"qal": total_qal / count, "chamfer_l1": total_cd / count, "fscore": total_f1 / count}


def train(args):
    torch.manual_seed(7)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader = DataLoader(ToyCompletionData(256, 7), batch_size=16, shuffle=True)
    eval_loader = DataLoader(ToyCompletionData(64, 8), batch_size=16)
    model = TinyCompletionNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(args.epochs):
        model.train()
        for partial, target in train_loader:
            partial, target = partial.to(device), target.to(device)
            optimizer.zero_grad()
            loss = qal_loss(model(partial), target)
            loss.backward()
            optimizer.step()
        print(f"epoch={epoch + 1:02d} qal={loss.item():.6f}")

    torch.save(model.state_dict(), args.checkpoint)
    print(json.dumps(evaluate(model, eval_loader, device, 0.001, 10.0, 1.0), indent=2))


def test(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TinyCompletionNet().to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device, weights_only=True))
    loader = DataLoader(ToyCompletionData(64, 8), batch_size=16)
    print(json.dumps(evaluate(model, loader, device, 0.001, 10.0, 1.0), indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("train", "eval"))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--checkpoint", type=Path, default=Path("minimal_qal.pt"))
    arguments = parser.parse_args()
    train(arguments) if arguments.mode == "train" else test(arguments)
