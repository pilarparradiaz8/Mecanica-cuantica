
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# =========================
# Red neuronal
# =========================
class PINN(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.activation = nn.Tanh()
        self.layers = nn.ModuleList(
            [nn.Linear(layers[i], layers[i+1]) for i in range(len(layers)-1)]
        )

    def forward(self, x):
        for i in range(len(self.layers)-1):
            x = self.activation(self.layers[i](x))
        return self.layers[-1](x)


# =========================
# Modelo PINN
# =========================
class SchrodingerPINN:
    def __init__(self, layers, n):
        self.net = PINN(layers)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=0.001)

        # Energía conocida: E = n + 1/2 (ħ=ω=1)
        self.E = n + 0.5

    def net_f(self, x):
        x.requires_grad_(True)
        u = self.net(x)

        u_x = torch.autograd.grad(u, x, torch.ones_like(u), create_graph=True)[0]
        u_xx = torch.autograd.grad(u_x, x, torch.ones_like(u_x), create_graph=True)[0]

        # Ecuación de Schrödinger:
        # -1/2 u'' + 1/2 x^2 u = E u
        f = -0.5 * u_xx + 0.5 * x**2 * u - self.E * u
        return f

    def loss(self, x):
        # Ecuación diferencial
        f = self.net_f(x)
        loss_f = torch.mean(f**2)

        # Decaimiento en los extremos (túnel)
        loss_bc = torch.mean(self.net(x[[0, -1]])**2)

        # Normalización
        u = self.net(x)
        dx = (x.max() - x.min()) / len(x)
        norm = torch.sum(u**2) * dx
        loss_norm = (norm - 1.0)**2

        return loss_f + loss_bc + loss_norm

    def train(self, x, epochs=3000):
        history = []
        for i in range(epochs):
            self.optimizer.zero_grad()
            loss = self.loss(x)
            loss.backward()
            self.optimizer.step()

            history.append(loss.item())

            if i % 500 == 0:
                print(f"Epoch {i}, Loss: {loss.item():.6f}")

        return history


# =========================
# Entrenamiento
# =========================
x = torch.linspace(-4, 4, 1000).view(-1,1)

layers = [1, 64, 64, 64, 1]

plt.figure(figsize=(8,6))

for n, color in zip([0,1,2], ['blue','red','green']):
    print(f"\nEntrenando para n = {n}")
    model = SchrodingerPINN(layers, n)
    model.train(x, epochs=3000)

    with torch.no_grad():
        u_pred = model.net(x)

    plt.plot(x.detach().numpy(), u_pred.detach().numpy(), label=f"PINN n={n}", color=color)

# =========================
# Solución exacta
# =========================
def psi_exact(n, x):
    from scipy.special import hermite
    from math import factorial, pi

    Hn = hermite(n)(x)
    Nn = 1.0 / np.sqrt((2**n) * factorial(n)) * (1/pi)**0.25
    return Nn * Hn * np.exp(-x**2/2)

x_np = x.numpy()

for n in [0,1,2]:
    plt.plot(x_np, psi_exact(n, x_np), '--', label=f"Exacta n={n}")

plt.title("Oscilador Armónico Cuántico (PINN vs Exacta)")
plt.xlabel("x")
plt.ylabel("ψ(x)")
plt.legend()
plt.grid()
plt.show()
