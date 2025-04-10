from django.shortcuts import render
from django.utils.html import escape

# View for 'index' page
def index(request):
    return  render(request, 'simulation/index.html', {})

# Scientific libraries
from matplotlib import pyplot as plt
import numpy as np
from scipy.integrate import odeint
from random import randint


# ====================================================================================================================================================================
# View for "Logging Experiment" article ==============================================================================================================================
# ====================================================================================================================================================================
def logging(request):
    mycache = randint(1000, 10000)
    simu= False
    try:
        mycache = randint(1000, 10000)
        I = escape(request.POST['I'])
        F = escape(request.POST['F'])
        D = escape(request.POST['D'])
        delta = escape(request.POST['delta'])
        I = float(I)
        F = float(F)
        D = float(D)
        delta = float(delta)

        if 0 <= I <= 0.5 and 5 <= F <= 50 and 100 <= D <= 200 and -0.1 <= delta <= 0.1:

            # Parameters of each plot of Paracou site

            # Param      num a      b      c           rho      alpha  mu      phi0   Kmax
            param2 =    [1, 0.5703, 0.795, 0.076024, 0.36297, 0.42479173, 0.0313225, 0.574147, 680]
            param3 =    [2, 0.473, 0.664, 0.0896,     0.446,   0.511, 0.0283, 0.644, 750]
            param4 =    [3, 0.473, 0.656, 0.089,      0.436,   0.507, 0.0284, 0.657, 800]
            param5 =    [4, 0.5703, 0.7342708333333333, 0.069875, 0.48267916, 0.4247916, 0.0313225, 0.70185208, 760]
            param7 =    [5, 0.5354483, 0.795, 0.094471, 0.394853, 0.4247916, 0.0313225, 0.620585, 690]
            param8 =    [6, 0.478, 0.662, 0.0714,     0.421,   0.513, 0.0296, 0.547, 790]
            param9 =    [7, 0.5703, 0.795, 0.0698749, 0.394853, 0.4247916, 0.0313225, 0.620585, 720]
            param10 =   [8, 0.477, 0.668, 0.0854,     0.439,   0.508, 0.0276, 0.685, 720]
            param12 =   [9, 0.5703, 0.7949, 0.07756125, 0.5226, 0.4247916, 0.0313225, 0.64380416, 780]

            PARAM = [param2,
                     param3,
                     param4,
                     param5,
                     param7,
                     param8,
                     param9,
                     param10,
                     param12]

            colors = ["blue", "red", "orange", "green", "cyan", "purple", "magenta", "olive", "brown"]

            # Parameters of the testing

            # I : Intensité des coupes (comprise entre 10 m3 et 40m3 équivalents en nombre d'arbres).
            # F : Intervalle de temps entre deux coupes (fenêtre à tester : entre 20 et 90 ans).
            # D : durée du test
            # incr : increase of mortality

            N = D/F
            incr = 1 + delta
            TIME = np.arange(2020, 2020+F, 0.25)

            # Forest model
            a = 0.1
            b = 0.1
            c = 0.1

            def gamma(v):
                return a*(v-b)**2 + c

            Kmax = 1
            def system(X, t, a, b, c, rho, alpha, mu):
                u, v = X
                du = rho*v - (a*(v-b)**2 + c)*u - alpha*u*v*(1-(u+v)/Kmax)
                dv = alpha*u*v*(1-(u+v)/Kmax)  - mu*v
                return [du, dv]


            dying = 0           # number of dying plots
            notdying = 0        # number of resilient plots
            dying_plots = []    # list of dying plots
            notdying_plots =[]  # list of resilient plots

            # First figure: time series
            fig = plt.figure(figsize=(10, 4))
            subfigs = fig.subfigures(1, 2, wspace=0.07)
            axsLeft = subfigs[0].subplots(1, 1)
            axsRight = subfigs[1].subplots(3, 3, sharey=True, sharex=True)
            u = []
            v = []
            cpt = 0
            for param in PARAM:
                num, a, b, c, rho, alpha, mu, phi0, km = param
                mu = mu * incr
                for j in range(int(N)):
                    time = []
                    for t in TIME:
                        time = time + [j*F + t]
                    if j==0:
                        u0 = phi0*(1-I)
                        v0 = (1-phi0)*(1-I)
                    else:
                        u0 = u[-1]*(1-I)
                        v0 = v[-1]*(1-I)
                    X0 = [u0, v0]
                    orbit = odeint(system, X0, time, args=(a, b, c, rho, alpha, mu))
                    u, v = orbit.T
                    traj = (u+v)*km
                    axsLeft.plot(time, traj, lw='2', c=colors[num-1])
                    axsLeft.set_xlabel("Time (years)")
                    axsLeft.set_ylabel("Density of trees ($10^3$ per ha)")
                    ax = axsRight[cpt//3][cpt%3]
                    ax.plot(time, traj, lw='2', c=colors[num-1])
                    ax.set_title("Plot "+str(num), fontsize='x-small')
                    if num in [7, 8, 9]:
                        ax.set_xlabel("Time (years)")
                    if num in [1, 4, 7]:
                        ax.set_ylabel("Density")
                if traj[-1]<100:
                    dying = dying + 1
                    dying_plots = dying_plots + [num]
                else:
                    notdying = notdying + 1
                    notdying_plots = notdying_plots + [num]
                cpt = cpt + 1
            fig.savefig('/home/velo/mysite/static/simulation/images/logging.svg')
            plt.close(fig)

            fig, ax = plt.subplots(figsize=(6, 3))
            data = [notdying, dying]
            ax.pie(data,
                labels=["Resilient plots", "Dying plots"],
                wedgeprops={'width':0.3},
                startangle=90)
            fig.savefig('/home/velo/mysite/static/simulation/images/logging2.svg')
            plt.close(fig)

            result = "Computation ended correctly ! You can make many other simulations."
            simu = True
        else:
            result = "Warning! You must respect the intervals for $I$, $F$, $D$ and $\delta$."
            I = 0
            F = 0
            D = 200
            delta = 0
            dying = 0
            notdying = 0
            dying_plots = []
            notdying_plots =[]

    except:
        mycache = randint(1000, 10000)
        result = "Warning! You have to respect the intervals for $I$, $F$, $D$ and $\delta$."
        I = 0
        F = 0
        D = 200
        delta = 0
        dying = 0
        notdying = 0
        dying_plots = []
        notdying_plots =[]
        return render(request, 'simulation/logging.html', {
            'result': result,
            'mycache': mycache,
            'simu': simu,
            'I': I,
            'F': F,
            'D': D,
            'delta': delta,
            'dying': dying,
            'notdying': notdying,
            'dying_plots': dying_plots,
            'notdying_plots': notdying_plots
        })
    return render(request, 'simulation/logging.html', {
	    'result': result,
        'mycache': mycache,
        'simu': simu,
        'I': I,
        'F': F,
        'D': D,
        'delta': delta,
        'dying': dying,
        'notdying': notdying,
        'dying_plots': dying_plots,
        'notdying_plots': notdying_plots
    })




