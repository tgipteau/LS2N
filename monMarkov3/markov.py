import pandas as pd
import os
import numpy as np
import random
from scipy.linalg import block_diag
from sphinx.builders.html import validate_html_static_path
from tqdm import tqdm
from scipy.integrate import odeint
import asyncio
from matplotlib import pyplot as plt
from numpy.random import  uniform


"""
#### SIMULATIONS
"""

DONE = False

a = 3
b = 1
c = 0.05

# boreale
alpha = 1
beta = 1
delta = 0.5

# discrétisation
L = 20
J = 40
T = 25
N = 1000
freq = 1
minree = 2
maxree = 6

p = 0.0
intensity = 0.9


# Calcul des paramètres dérivés
dx = L / J
dy = L / J
dt = T / N
dtau = dt / 2
kx = delta * dtau / dx ** 2
ky = delta * dtau / dy ** 2

T_linspace = np.linspace(start=0, stop=T - 2, num=T-1)

uplus = b + np.sqrt((alpha * delta - c) / a)
umoins = b - np.sqrt((alpha * delta - c) / a)
wplus = (alpha / beta) * uplus
wmoins = (alpha / beta) * umoins

print("uplus", uplus)
print("wplus", wplus)
print("umoins", umoins)
print("wmoins", wmoins)


Ueq = np.array([uplus, wplus])
Ueq = np.repeat(Ueq, J*J, axis=0).reshape(J*J, 2)


def q(u):
    return u * (a * (u - b) ** 2 + c)


def reaction(X, t):
    u, w = X
    
    du = beta * delta * w - q(u)
    dw = alpha * u - beta * w
    
    return [du, dw]



def start_in_ball(center, radius):
    
    center = np.ones(J*J) * center
    d = len(center)
    X = np.random.randn(2, d)
    X /= np.linalg.norm(X, axis=1, keepdims=True)  # unit vectors
    R = np.random.rand(2, 1) ** (1 / d) * radius  # uniform radius in ball
    return center + X * R # [u0, w0]

    
async def run_simulation(start_from, nb_simulations, ballcenter, ballradius, save_folder="Simulations"):
    global DONE
    
    os.makedirs(save_folder, exist_ok=True)
    os.makedirs("Fires", exist_ok=True)
    
    # Matrices du schéma : boreale
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
    # print("Inversion de matrices...")
    invA = np.linalg.inv(A)

    
    for isim in tqdm(range(start_from, start_from + nb_simulations)):
        await asyncio.sleep(1)
        
        dat_file = os.path.join(save_folder, "data" + str(isim) + ".csv")
        fires_file = os.path.join('Fires', "fires" + str(isim) + ".dat")
        
        # Condition initiale
        x = np.linspace(0.0, L, J)
        y = np.linspace(0.0, L, J)
        X = np.linspace(0.0, L * L, J * J)
        
        [u_0, w_0] = start_in_ball(ballcenter, ballradius)
        
        u = u_0
        w = w_0

        # print('Calculs en cours...')
        file = open(dat_file, 'w')
        file_fire = open(fires_file, 'w')
        file.write('t i j u w \n')
        
        for t in range(T):
            for i in range(J):
                for j in range(J):
                    file.write(
                        str(t) + ' ' + str(i) + ' ' + str(j) + ' ' + str(u[i * J + j]) + ' ' + str(w[i * J + j]) +'\n'
                    )
            
            """Méthode de Strang"""
            # diffusion 1/2 pas
            u = np.dot(invA, u)
            w = np.dot(invA, w)
            
            # réaction 1 pas
            for i in range(J):
                for j in range(J):
                    idx = i * J + j
                    X0 = [u[idx], w[idx]]
                    orbit = odeint(reaction, X0, [0, dt],)
                    u[idx], w[idx] = orbit[-1]
            
            # diffusion 1/2 pas
            u = np.dot(invA, u)
            w = np.dot(invA, w)

            if t % freq == 0:
                if np.random.binomial(1, p) == 1:
                    
                    xee = random.random() * J
                    yee = random.random() * J
                    
                    ree = minree + random.random() * (maxree - minree)
                    
                    file_fire.write(str(t) + ' ' + str(xee) + ' ' + str(yee) + ' '
                                    + str(ree) + ' ' + str(intensity) + '\n')
                    
                    var_ee = lambda i, j, var: var[i * J + j] * (1 - intensity) if ((i - xee) ** 2 + (
                            j - yee) ** 2) < ree ** 2 else var[i * J + j]
                    
                    for i in range(J):
                        for j in range(J):
                            u[i * J + j] = var_ee(i, j, u)
                            w[i * J + j] = var_ee(i, j, w)

    DONE = True


async def update_plots():
    global DONE
    
    save_path = "Simulations"
    os.makedirs(save_path, exist_ok=True)
    treated_sims = []
    traces = []
    
    plt.ion()
    fig, ax = plt.subplots(figsize=(5, 8))
    persistence_norm = np.linalg.norm(Ueq)
    ax.set_ylim(persistence_norm * 0.7 , persistence_norm * 1.1)

    ax.axhline(y=float(persistence_norm), color='green', linestyle='--')  # ligne "persistence"
    #ax.invert_yaxis()
    
    plt.show(block=False)
    ax.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
    ax.set_xlabel("time")
    
    ax.set_title('Distance from extinction')
    
    while True:
        for sim in os.listdir(save_path):
            if sim not in treated_sims:
                    
                    #print("UPDATE PLOT: with ", str(sim))
                    df = pd.read_csv(save_path + "/" + sim, names=["t", "x", "y", "u", "w"],
                                     header=0, sep=" ", index_col=False)
                    trace = []
                    
                    #print(np.linalg.norm(Ueq - df[df['t'] == 0].iloc[:, -2:]))
                    # pour chaque temps, calculer les normes
                    for i in T_linspace:
                        Ut = df[df['t'] == i].iloc[:, -2:]
                        Ut = Ut.to_numpy()
                        #print("t", i)
                        #print("Ut", Ut)
                        trace.append(np.linalg.norm(Ut))
                    
                    ax.plot(T_linspace, trace)
                    plt.pause(0.01)
                    plt.show(block=False)
                    treated_sims.append(sim)

        
        if DONE:
            plt.savefig("fig.png")
            break
        await asyncio.sleep(1)


async def main():
    await asyncio.gather(
        update_plots(),
        run_simulation(start_from=0, nb_simulations=50, ballcenter=uplus, ballradius=.2)
    )


asyncio.run(main())

