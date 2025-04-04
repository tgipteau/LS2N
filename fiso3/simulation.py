#!/usr/bin/env python3

"""
Simulation d'un modèle de forêt
déterminé par un système de réaction-diffusion
"""

# Bibliothèques scientifiques
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import block_diag
from scipy.interpolate import griddata
import mpl_toolkits.mplot3d.art3d as art3d
from scipy.integrate import odeint
import os
import random
import math

from tqdm import tqdm

# Paramètres du modèle
alpha = 1
beta = 1
a = 3
b = 1
c = 0.5
delta = 0.5

# Paramètres de discrétisation
L = 50
J = 50
T = 10
N = 1000

dx = L / J
dy = L / J
dt = T / N
dtau = dt / 2

kx = delta * dtau / (dx ** 2)
ky = delta * dtau / (dy ** 2)

# Paramètres feux
p = 0.1  # proba feu à chaque pas de temps
freq = 1  # pas de temps minimal entre feux
intensity = 0.7
minree = 1
maxree = 5

# Nettoyer le fichier des feux
with open(os.path.join("Simulations/default", "fires.dat"), 'w'):
    pass


# Termes de réaction du modèle
def reaction(X, t):
    u, w = X
    du = alpha * w - q(u)
    dw = -beta * w + alpha * u
    return [du, dw]


def q(u):
    return u * (a * (u - b) ** 2 + c)


def run_simulation(save_folder="Simulations/default"):
    dat_file = os.path.join(save_folder, "data.csv")
    param_file = os.path.join(save_folder, "param.txt")
    
    # Écriture des paramètres dans un fichier texte
    params = {
        "alpha": alpha,
        "beta": beta,
        "a": a,
        "b": b,
        "c": c,
        "delta": delta,
        "L": L,
        "J": J,
        "T": T,
        "N": N,
        "dx": dx,
        "dy": dy,
        "dt": dt,
        "dtau": dtau,
        "kx": kx,
        "ky": ky,
        "p": p,
        "freq": freq,
        "intensity": intensity,
        "minree": minree,
        "maxree": maxree,
    }
    
    with open(param_file, "w") as f:
        for key, value in params.items():
            f.write(f"{key} {value}\n")
    
    print(f"Paramètres enregistrés dans {param_file}")
    # Équilibres du modèle
    uplus = b + np.sqrt((alpha ** 2 - beta * c) / (a * beta));
    wplus = alpha * uplus / beta;
    
    umoins = b - np.sqrt((alpha ** 2 - beta * c) / (a * beta));
    wmoins = alpha * umoins / beta;
    
    print("Uplus = (", uplus, ", ", wplus, ")")
    print("Umoins = (", umoins, ", ", wmoins, ")")
    
    # Condition initiale
    x = np.linspace(0.0, L, J)
    y = np.linspace(0.0, L, J)
    X = np.linspace(0.0, L * L, J * J)
    
    u0 = np.zeros(J * J)
    w0 = np.zeros(J * J)
    for i in range(J):
        for j in range(J):
            u0[i * J + j] = umoins + (-1 + 2 * random.random()) * 0.5
            w0[i * J + j] = wmoins + (-1 + 2 * random.random()) * 0.5
    
    # Matrice du schéma
    diag = np.ones(J)
    diagsup = np.ones(J - 1)
    D = np.diag(diag * (1 + 2 * (kx + ky)), 0) + np.diag(diagsup * (-ky), 1) + np.diag(diagsup * (-ky), -1)
    A = block_diag(D)
    for i in range(J - 1):
        A = block_diag(A, D)
    
    # Conditions au bord de Neumann
    for k in [0, J - 1, J * J - J, J * J - 1]:
        A[k][k] = 1 + kx + ky
    for n in range(1, J - 1):
        n1 = J * n
        n2 = J * n + J - 1
        A[n1][n1] = 1 + 2 * kx + ky
        A[n2][n2] = 1 + 2 * kx + ky
    for k in range(1, J - 1):
        A[k][k] = 1 + kx + 2 * ky
    for k in range(J * J - J + 1, J * J - 1):
        A[k][k] = 1 + kx + 2 * ky
    
    grandediag = np.ones(J * (J - 1))
    A = A + np.diag(grandediag * (-ky), J) + np.diag(grandediag * (-ky), -J)
    
    # Calculs
    print("Inversion de matrices...")
    invA = np.linalg.inv(A)
    
    u = u0
    w = w0
    
    print('Calculs en cours...')
    file = open(dat_file, 'w')
    file.write('t i j u w \n')
    
    for t in tqdm(range(T)):
        for i in range(J):
            for j in range(J):
                file.write(
                    str(t) + ' ' + str(i) + ' ' + str(j) + ' ' + str(u[i * J + j]) + ' ' + str(w[i * J + j]) + '\n')
        
        """Méthode de Strang"""
        # diffusion 1/2 pas
        u = np.dot(invA, u)
        w = np.dot(invA, w)
        
        # réaction 1 pas
        for i in range(J):
            for j in range(J):
                X0 = [u[i * J + j], w[i * J + j]]
                orbit = odeint(reaction, X0, [0, dt])
                newpoint = orbit[-1]
                u[i * J + j], w[i * J + j] = newpoint.T
        
        # diffusion 1/2 pas
        u = np.dot(invA, u)
        w = np.dot(invA, w)
        
        if t % freq == 0:
            if np.random.binomial(1, p) == 1:
                
                xee = random.random() * J
                yee = random.random() * J
                
                ree = minree + random.random() * (maxree - minree)
                
                with open(os.path.join(save_folder, "fires.dat"), 'a') as ffires:
                    ffires.write(
                        str(t) + ' ' + str(xee) + ' ' + str(yee) + ' ' + str(ree) + ' ' + str(intensity) + '\n')
                
                var_ee = lambda i, j, var: var[i * J + j] * (1 - intensity) if ((i - xee) ** 2 + (
                            j - yee) ** 2) < ree ** 2 else var[i * J + j]
                
                for i in range(J):
                    for j in range(J):
                        u[i * J + j] = var_ee(i, j, u)
                        w[i * J + j] = var_ee(i, j, w)
    
    print('Fin du programme de simulation.')
    
    return params
