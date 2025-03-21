from __future__ import print_function
import argparse
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from aliyah import monitor, trainingmonitor


"""
Example adapted from https://github.com/pytorch/examples/blob/main/vae/main.py
"""



parser = argparse.ArgumentParser(description="VAE MNIST Example")
parser.add_argument(
    "--batch-size",
    type=int,
    default=128,
    metavar="N",
    help="input batch size for training (default: 128)",
)
parser.add_argument(
    "--epochs",
    type=int,
    default=10,
    metavar="N",
    help="number of epochs to train (default: 10)",
)
parser.add_argument(
    "--no-cuda", action="store_true", default=False, help="disables CUDA training"
)
parser.add_argument(
    "--no-mps", action="store_true", default=False, help="disables macOS GPU training"
)
parser.add_argument(
    "--seed", type=int, default=1, metavar="S", help="random seed (default: 1)"
)
parser.add_argument(
    "--log-interval",
    type=int,
    default=10,
    metavar="N",
    help="how many batches to wait before logging training status",
)
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
use_mps = not args.no_mps and torch.backends.mps.is_available()

torch.manual_seed(args.seed)

if args.cuda:
    device = torch.device("cuda")
elif use_mps:
    device = torch.device("mps")
else:
    device = torch.device("cpu")

kwargs = {"num_workers": 1, "pin_memory": True} if args.cuda else {}
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST(
        "data", train=True, download=False, transform=transforms.ToTensor()
    ),
    batch_size=args.batch_size,
    shuffle=True,
    **kwargs,
)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST("data", train=False, transform=transforms.ToTensor()),
    batch_size=args.batch_size,
    shuffle=False,
    **kwargs,
)


class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        self.fc1 = nn.Linear(784, 400)
        self.fc21 = nn.Linear(400, 20)
        self.fc22 = nn.Linear(400, 20)
        self.fc3 = nn.Linear(20, 400)
        self.fc4 = nn.Linear(400, 784)
        self.layers = [self.fc1, self.fc21, self.fc22, self.fc3, self.fc4]
        self.activation_values = {}
        for i, layer in enumerate(self.layers):
            layer.register_forward_hook(self.get_activation_hook(f" layer_{i}"))

    def get_activation_hook(self, name):
        def hook(module, input, output):
            if isinstance(module, nn.Linear):
                self.activation_values[name] = output

        return hook

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 784))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


model = VAE().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)


# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction="sum")

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE , KLD


def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        if not monitor.check_control():
            return

        data = data.to(device)
        target = target.to(device)  # Add this line to move target to device
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        BCE, KLD = loss_function(recon_batch, data, mu, logvar)
        loss = BCE + KLD
        per_example_bce = BCE / len(data)
        per_example_kld = KLD / len(data)
        per_example_total = loss.item() / len(data)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        pred = recon_batch.argmax(dim=1, keepdim=True)
        accuracy = pred.eq(target.view_as(pred)).float().mean().item() * 100
        monitor.log_batch(
            batch_idx, 
            bce_per_example=per_example_bce,
            kld_per_example=per_example_kld,
            total_per_example=per_example_total,
        )
        if batch_idx % args.log_interval == 0:
            e_loss = loss.item() / len(data)
            monitor.log_epoch(epoch, e_loss)
            for idx, (name, activations) in enumerate(model.activation_values.items()):
                    # Get post-activation values and normalize
                    act_mean = activations.abs().mean(dim=0).detach().cpu().numpy()
                    act_norm = (act_mean - act_mean.min()) / (act_mean.max() - act_mean.min() + 1e-8)
                    monitor.log_layer_state(idx, act_norm.tolist())


def test(epoch):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for i, (data, _) in enumerate(test_loader):
            data = data.to(device)
            recon_batch, mu, logvar = model(data)
            BCE, KLE = loss_function(recon_batch, data, mu, logvar)
            test_loss += BCE + KLE
            if i == 0:
                n = min(data.size(0), 8)
                comparison = torch.cat(
                    [
                        data[:n],
                        recon_batch.view(args.batch_size, 1, 28, 28)[:n],
                    ]
                )

    test_loss /= len(test_loader.dataset)
    print("====> Test set loss: {:.4f}".format(test_loss))

def calculate_reconstruction_similarity(data, reconstructed):
    """Calculates the average pixel-wise similarity between the original and reconstructed images."""
    # Flatten the images
    data_flat = data.view(data.size(0), -1)
    reconstructed_flat = reconstructed.view(reconstructed.size(0), -1)

    # Calculate the cosine similarity for each image in the batch
    similarity = F.cosine_similarity(reconstructed_flat, data_flat, dim=1)

    # Return the average similarity over the batch
    return similarity.mean().item()

if __name__ == "__main__":
    with trainingmonitor() as monitor:
        sample_data, sample_target = next(iter(train_loader))
        sample_data = sample_data.to(device)
        for epoch in range(1, args.epochs + 1):
            train(epoch)
            test(epoch)
            with torch.no_grad():
                sample = torch.randn(64, 20).to(device)
                sample = model.decode(sample).cpu()
                # Generate a sample reconstruction
                reconstructed, mu, logvar = model(sample_data)

                # Calculate reconstruction similarity
                similarity_score = calculate_reconstruction_similarity(
                    sample_data, reconstructed
                )

                # Get the true digit label
                true_digit = sample_target[0].item()

                # Log the reconstruction similarity
                labels = ["similarity"]  # Use a single label for the similarity score
                probabilities = [
                    similarity_score
                ]  # Pass the similarity score as the "probability"
                description = (
                    f"Reconstruction similarity for digit {true_digit}: {similarity_score:.4f}"
                )
                monitor.log_prediction(probabilities, labels, description) 
            if not monitor.check_control():
                break
