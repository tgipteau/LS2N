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

"""
#### SIMULATIONS
"""
DONE = False

## Chargement des paramètres depuis le fichier YAML (config)
# competitions
a = 3
b = 1
c = 0.5
gamma_south = -0.4
gamma_north = 0.4

# boreale
alpha_b = 1
beta_b = 1
delta_b = 0.5

# mixte
alpha_m = 1
beta_m = 1
delta_m = 0.5

# discrétisation
L = 20
J = 40
T = 10
N = 1000
freq = 1
minree = 2
maxree = 6

# Calcul des paramètres dérivés
dx = L / J
dy = L / J
dt = T / N
dtau = dt / 2
kx_b = delta_b * dtau / dx ** 2
ky_b = delta_b * dtau / dy ** 2
kx_m = delta_m * dtau / dx ** 2
ky_m = delta_m * dtau / dy ** 2

gamma = np.linspace(start=gamma_south, stop=gamma_north, num=J)
T_linspace = np.linspace(start=0, stop=T - 1, num=T)

uplus = b + np.sqrt(abs((alpha_b ** 2 - beta_b * c) / (a * beta_b)))
wplus = alpha_b * uplus / beta_b
umoins = b - np.sqrt(abs((alpha_b ** 2 - beta_b * c) / (a * beta_b)))
wmoins = alpha_b * umoins / beta_b

Ueq = np.array([uplus, wplus, uplus, wplus])
Ueq = np.repeat(Ueq, 1600, axis=0).reshape(1600, 4)


def q_b(ub, um, gamma_i):
    return ub * (a * (ub + um - b) ** 2 + c) + gamma_i * ub * um


def q_m(ub, um, gamma_i):
    return um * (a * (um + ub - b) ** 2 + c) - gamma_i * ub * um


def reaction(X, t, i):
    ub, um, wb, wm = X
    gamma_i = gamma[i]
    
    dub = alpha_b * wb - q_b(ub, um, gamma_i)
    dwb = -beta_b * wb + alpha_b * ub
    
    dum = alpha_m * wm - q_m(ub, um, gamma_i)
    dwm = -beta_m * wm + alpha_m * um
    
    return [dub, dum, dwb, dwm]


def randomize_starter():
    
    """
    # méthode naïve
    u_0 = np.zeros(J * J)
    w_0 = np.zeros(J * J)
    for i in range(J*J):
        u_0[i] = random.random() * uplus
        w_0[i] = random.random() * uplus
    return u_0, w_0
    """
    
    # méthode "à bosses"
    max_bosses = 30
    max_d = 30
    
    nb_bosses = random.randint(0, max_bosses)
    
    u_0 = np.ones(J*J) * 0
    w_0 = np.ones(J*J) * 0

    
    for _ in range(nb_bosses):
        
        h = random.random() * uplus
        d = random.randint(1, max_d)
        cx = random.randint(0, J)
        cy = random.randint(0, J)
        
        #print(f"cx={cx}, cy={cy}")
        
        # print(f"h = {h}, d = {d}, cx = {cx}, cy = {cy}")
        
        def bosse_func(i, j):
            return h * d / ((cx - i) ** 2 + (cy - j) ** 2 + d)
        
        #print(f"h={h}, d={d}, cx={cx}, cy={cy}")
        for i in range(J):
            for j in range(J):
                u_0[i * J + j] += bosse_func(i, j)
                w_0[i * J + j] += bosse_func(i, j)
                u_0[i * J + j] = min(uplus, u_0[i * J + j])
                w_0[i * J + j] = min(wplus, w_0[i * J + j])
                u_0[i * J + j] = max(0, u_0[i * J + j])
                w_0[i * J + j] = max(0, w_0[i * J + j])
    
    return u_0, w_0
    
    
    
async def run_simulation(start_from, nb_simulations, p, intensity, save_folder="Simulations"):
    global DONE
    
    os.makedirs(save_folder, exist_ok=True)
    os.makedirs("Fires", exist_ok=True)
    
    # Matrices du schéma : boreale
    diag = np.ones(J)
    diagsup = np.ones(J - 1)
    D = np.diag(diag * (1 + 2 * (kx_b + ky_b)), 0) + np.diag(diagsup * (-ky_b), 1) + np.diag(diagsup * (-ky_b), -1)
    A_b = block_diag(D)
    for i in range(J - 1):
        A_b = block_diag(A_b, D)
    
    # Conditions au bord de Neumann
    for k in [0, J - 1, J * J - J, J * J - 1]:
        A_b[k][k] = 1 + kx_b + ky_b
    for n in range(1, J - 1):
        n1 = J * n
        n2 = J * n + J - 1
        A_b[n1][n1] = 1 + 2 * kx_b + ky_b
        A_b[n2][n2] = 1 + 2 * kx_b + ky_b
    for k in range(1, J - 1):
        A_b[k][k] = 1 + kx_b + 2 * ky_b
    for k in range(J * J - J + 1, J * J - 1):
        A_b[k][k] = 1 + kx_b + 2 * ky_b
    
    grandediag = np.ones(J * (J - 1))
    A_b = A_b + np.diag(grandediag * (-ky_b), J) + np.diag(grandediag * (-ky_b), -J)
    
    # Matrices du schéma : mixte
    diag = np.ones(J)
    diagsup = np.ones(J - 1)
    D = np.diag(diag * (1 + 2 * (kx_m + ky_m)), 0) + np.diag(diagsup * (-ky_m), 1) + np.diag(diagsup * (-ky_m), -1)
    A_m = block_diag(D)
    for i in range(J - 1):
        A_m = block_diag(A_m, D)
    
    # Conditions au bord de Neumann
    for k in [0, J - 1, J * J - J, J * J - 1]:
        A_m[k][k] = 1 + kx_m + ky_m
    for n in range(1, J - 1):
        n1 = J * n
        n2 = J * n + J - 1
        A_m[n1][n1] = 1 + 2 * kx_m + ky_m
        A_m[n2][n2] = 1 + 2 * kx_m + ky_m
    for k in range(1, J - 1):
        A_m[k][k] = 1 + kx_m + 2 * ky_m
    for k in range(J * J - J + 1, J * J - 1):
        A_m[k][k] = 1 + kx_m + 2 * ky_m
    
    grandediag = np.ones(J * (J - 1))
    A_m = A_m + np.diag(grandediag * (-ky_m), J) + np.diag(grandediag * (-ky_m), -J)
    
    # Calculs
    # print("Inversion de matrices...")
    invA_b = np.linalg.inv(A_b)
    invA_m = np.linalg.inv(A_m)
    
    for isim in tqdm(range(start_from, start_from+nb_simulations)):
        await asyncio.sleep(1)
        
        dat_file = os.path.join(save_folder, "data" + str(isim) + ".csv")
        fires_file = os.path.join('Fires', "fires" + str(isim) + ".dat")
        
        # Condition initiale
        x = np.linspace(0.0, L, J)
        y = np.linspace(0.0, L, J)
        X = np.linspace(0.0, L * L, J * J)
        
        [u_b0, w_b0] = randomize_starter()
        [u_m0, w_m0] = randomize_starter()
        
        u_b = u_b0
        w_b = w_b0
        u_m = u_m0
        w_m = w_m0
        
        # print('Calculs en cours...')
        file = open(dat_file, 'w')
        file_fire = open(fires_file, 'w')
        file.write('t i j ub wb um wm \n')
        
        for t in range(T):
            for i in range(J):
                for j in range(J):
                    file.write(
                        str(t) + ' ' + str(i) + ' ' + str(j) + ' ' + str(u_b[i * J + j]) + ' ' + str(w_b[i * J + j]) +
                        ' ' + str(u_m[i * J + j]) + ' ' + str(w_m[i * J + j]) + '\n'
                    )
            
            """Méthode de Strang"""
            # diffusion 1/2 pas
            u_b = np.dot(invA_b, u_b)
            w_b = np.dot(invA_b, w_b)
            u_m = np.dot(invA_m, u_m)
            w_m = np.dot(invA_m, w_m)
            
            # réaction 1 pas
            for i in range(J):
                for j in range(J):
                    idx = i * J + j
                    X0 = [u_b[idx], u_m[idx], w_b[idx], w_m[idx]]
                    orbit = odeint(reaction, X0, [0, dt], args=(j,))
                    u_b[idx], u_m[idx], w_b[idx], w_m[idx] = orbit[-1]
            
            # diffusion 1/2 pas
            u_b = np.dot(invA_b, u_b)
            w_b = np.dot(invA_b, w_b)
            u_m = np.dot(invA_m, u_m)
            w_m = np.dot(invA_m, w_m)
            
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
                            u_b[i * J + j] = var_ee(i, j, u_b)
                            w_b[i * J + j] = var_ee(i, j, w_b)
                            u_m[i * J + j] = var_ee(i, j, u_m)
                            w_m[i * J + j] = var_ee(i, j, w_m)
    
    DONE = True


async def update_plots():
    global DONE
    
    save_path = "Simulations"
    os.makedirs(save_path, exist_ok=True)
    treated_sims = []
    traces = []
    
    plt.ion()
    fig, ax = plt.subplots(figsize=(5, 8))
    val_span = np.linalg.norm(Ueq)
    ax.set_ylim(0, val_span*1.1)
    ax.axhline(y=float(val_span), color='red', linestyle='--')  # ligne "extinction"
    #ax.invert_yaxis()
    
    plt.show(block=False)
    ax.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
    ax.set_xlabel("time")
    
    ax.set_title('Distance from extinction')
    
    while True:
        for sim in os.listdir(save_path):
            if sim not in treated_sims:
                try:
                    
                    print("UPDATE PLOT: with ", str(sim))
                    df = pd.read_csv(save_path + "/" + sim, names=["t", "x", "y", "ub", "wb", "um", "wm"],
                                     header=0, sep=" ", index_col=False).drop_duplicates()
                    trace = []
                    
                    print(np.linalg.norm(Ueq - df[df['t'] == 0].iloc[:, -4:]))
                    # pour chaque temps, calculer les normes
                    for i in T_linspace:
                        Ut = df[df['t'] == i].iloc[:, -4:]
                        trace.append(np.linalg.norm(Ut))
                    
                    ax.plot(T_linspace, trace)
                    plt.pause(0.01)
                    plt.show(block=False)
                    treated_sims.append(sim)
                except ValueError:
                    print("error")
                    pass
        
        if DONE:
            plt.savefig("fig.png")
            break
        await asyncio.sleep(1)


async def main():
    await asyncio.gather(
        update_plots(),
        run_simulation(start_from=0, nb_simulations=50, p=0.1, intensity=0.9)
    )


asyncio.run(main())
