from mine.models.mine import Mine
import torch.nn as nn
import numpy as np
import torchvision
import torch
import torchvision.transforms as transforms
from tqdm import tqdm
import torch.nn.functional as F
import matplotlib.pyplot as plt
from math import ceil
import sys

lam_str = sys.argv[1]
lam = float(sys.argv[1])

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = 'cpu'
print(device)

batch_size = 1000
image_dim = 224 // 5


def scale_rgb(x):
    return (x - x.min()) / (x.max() - x.min())


images_flat = np.load("images_flat_lam.npy")
responses = np.load("responses_lam.npy")
x_dim = (224 // 5) * (224 // 5) * 3
y_dim = 1


class Image_network(nn.Module):
    def __init__(self, x_dim, y_dim):
        super().__init__()
        self.fc1x = nn.Linear(x_dim, 1)
        self.fc1y = nn.Linear(y_dim, 1)
        self.fc2 = nn.Linear(2, 100)
        self.fc3 = nn.Linear(100, 1)

    def forward(self, x, y):
        x = F.relu(self.fc1x(x))
        y = F.relu(self.fc1y(y))
        h = torch.cat((x, y), dim=1)
        h = F.relu(self.fc2(h))
        h = self.fc3(h)
        return h


mine = Mine(
    T=Image_network(x_dim, y_dim),
    loss="mine",  # mine_biased, fdiv
    # method = 'concat'
    # method = ''
).to(device)

mi, loss_log = mine.optimize(torch.tensor(images_flat, dtype=torch.float32).to(device), torch.tensor(responses, dtype=torch.float32).to(device), 1000, batch_size, lam)

torch.save(mine.T, "mineT" + lam_str + ".pth")
np.save("mi" + lam_str + ".npy", mi.detach().cpu().numpy())
np.save("loss" + lam_str + ".npy", torch.stack(loss_log).detach().cpu().numpy())

plt.figure()
plt.plot(torch.stack(loss_log).detach().cpu().numpy())
plt.title("loss: new arch, 1000 epochs, lambda=" + lam_str)
plt.ylabel("loss")
plt.xlabel("batches")
plt.savefig("loss" + lam_str+".pdf")

Tweights = mine.T.fc1x.weight.detach().cpu().numpy()[0]

unflat_Tweights = np.reshape(Tweights, (3, 224 // 5, 224 // 5))

plt.clf()
plt.figure()
plt.pcolormesh(scale_rgb(unflat_Tweights[0]), edgecolors="k", linewidth=0.005)
ax = plt.gca()
ax.set_aspect("equal")
plt.colorbar()
plt.savefig("Tweightsc0" + lam_str + ".pdf")

plt.clf()
plt.figure()
plt.pcolormesh(scale_rgb(unflat_Tweights[1]), edgecolors="k", linewidth=0.005)
ax = plt.gca()
ax.set_aspect("equal")
plt.colorbar()
plt.savefig("Tweightsc1" + lam_str + ".pdf")

plt.clf()
plt.figure()
plt.pcolormesh(scale_rgb(unflat_Tweights[2]), edgecolors="k", linewidth=0.005)
ax = plt.gca()
ax.set_aspect("equal")
plt.colorbar()
plt.savefig("Tweightsc2" + lam_str + ".pdf")
plt.clf()

plt.figure()
plt.pcolormesh(
    np.transpose(np.array(list(map(scale_rgb, unflat_Tweights))), (1, 2, 0)),
    edgecolors="k",
    linewidth=0.005,
)
ax = plt.gca()
ax.set_aspect("equal")
plt.colorbar()
plt.savefig("Tweightscomb" + lam_str + ".pdf")
