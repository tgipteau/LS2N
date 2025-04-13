import yaml
import os
import numpy as np
import random
from scipy.linalg import block_diag
from tqdm import tqdm
from scipy.integrate import odeint

with open("config.yaml", 'r') as f:
    config = yaml.safe_load(f)

## Chargement des paramètres depuis le fichier YAML (config)
# competitions
a = config["params_simulation"]["a"]
b = config["params_simulation"]["b"]
c = config["params_simulation"]["c"]
gamma_south = config["params_simulation"]["gamma_south"]
gamma_north = config["params_simulation"]["gamma_north"]

# boreale
alpha_b = config["params_simulation"]["alpha_b"]
beta_b = config["params_simulation"]["beta_b"]
delta_b = config["params_simulation"]["delta_b"]

# mixte
alpha_m = config["params_simulation"]["alpha_m"]
beta_m = config["params_simulation"]["beta_m"]
delta_m = config["params_simulation"]["delta_m"]

# discrétisation
L = config["params_simulation"]["L"]
J = config["params_simulation"]["J"]
T = config["params_simulation"]["T"]
N = config["params_simulation"]["N"]
p = config["params_simulation"]["p"]
freq = config["params_simulation"]["freq"]
intensity = config["params_simulation"]["intensity"]
minree = config["params_simulation"]["minree"]
maxree = config["params_simulation"]["maxree"]

# Calcul des paramètres dérivés
dx = L / J
dy = L / J
dt = T / N
dtau = dt / 2
kx_b = delta_b * dtau / dx ** 2
ky_b = delta_b * dtau / dy ** 2
kx_m = delta_m * dtau / dx ** 2
ky_m = delta_m * dtau / dy ** 2

# paramètres randomisation du starter


gamma = np.linspace(start=gamma_south, stop=gamma_north, num=J)
T_linspace = np.linspace(start=0, stop=T - 1, num=T)


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


def randomize_starter(uplus, wplus, umoins, wmoins):
    """
    # méthode naive : état instable +- 0.5 partout, sur chaque point
    u_0 = np.zeros(J * J)
    w_0 = np.zeros(J * J)
    for i in range(J):
        for j in range(J):
            u_b0[i * J + j] = umoins + (-1 + 2 * random.random()) * 0.5
            w_b0[i * J + j] = wmoins + (-1 + 2 * random.random()) * 0.5
    
    return u_0, w_0
    """
    
    # méthode "à bosses"
    
    max_bosses = 50
    max_height = uplus
    max_d = 5
    
    nb_bosses = random.randint(1, max_bosses)
    u_0 = np.zeros(J * J)
    w_0 = np.zeros(J * J)
    
    for _ in range(nb_bosses):
        
        h = random.random() * max_height
        d = random.randint(1, max_d)
        cx = random.randint(0, J)
        cy = random.randint(0, J)
        
        #print(f"h = {h}, d = {d}, cx = {cx}, cy = {cy}")
        
        bosse_func = lambda i, j: h * d / ((cx - i) ** 2 + (cy - j) ** 2 + d)
        
        for i in range(J):
            for j in range(J):
                u_0[i * J + j] += bosse_func(i, j)
                w_0[i * J + j] += bosse_func(i, j)
    
    return u_0, w_0

def run_simulation(sim_name, save_folder="Simulations"):
    os.makedirs(save_folder, exist_ok=True)
    
    dat_file = os.path.join(save_folder, "data"+sim_name+".csv")
    fires_file = os.path.join(save_folder, "fires"+sim_name+".dat")
    
    # Équilibres du modèle
    uplus = b + np.sqrt(abs((alpha_b ** 2 - beta_b * c) / (a * beta_b)))
    wplus = alpha_b * uplus / beta_b
    umoins = b - np.sqrt(abs((alpha_b ** 2 - beta_b * c) / (a * beta_b)))
    wmoins = alpha_b * umoins / beta_b
    
    #print("Uplus = (", uplus, ", ", wplus, ")")
    #print("Umoins = (", umoins, ", ", wmoins, ")")
    
    # Condition initiale
    x = np.linspace(0.0, L, J)
    y = np.linspace(0.0, L, J)
    X = np.linspace(0.0, L * L, J * J)
    
    [u_b0, w_b0] = randomize_starter(uplus, wplus, umoins, wmoins)
    [u_m0, w_m0] = randomize_starter(uplus, wplus, umoins, wmoins)
    
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
    #print("Inversion de matrices...")
    invA_b = np.linalg.inv(A_b)
    invA_m = np.linalg.inv(A_m)
    
    u_b = u_b0
    w_b = w_b0
    u_m = u_m0
    w_m = w_m0
    
    #print('Calculs en cours...')
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
    
    #print('Fin du programme de simulation.')
    
    return uplus, wplus, umoins, wmoins
