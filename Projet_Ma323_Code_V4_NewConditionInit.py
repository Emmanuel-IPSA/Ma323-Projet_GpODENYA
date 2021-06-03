########## Ma323 - Projet : Equation de transport-diffusion ##############
### Cyrine Grar / David Karalekian / Emmanuel Odenya / Emerick Perrin #####

''' Même résolution avec une nouvelle condition initiale'''

### Imports

import numpy as np
import matplotlib.pyplot as plt


## Valeurs test

V = 1
nu = 1/20
h = 0.05
tau = 0.0025
'''Les deux cas que nous n'affichons pas'''
#h = 0.2
#tau = 0.5

#h = 0.1
#tau = 0.05

x0 = -10
xfin = 10

N = int((xfin - x0)/h)

c = nu*tau/h**2
d = V*tau/h

Tmin = 0
Tmax = 4

### Fonctions 

def matriceM(a, b, c):
    """Construit une matrice carrée de taille N, tridiagonale.
    Le coefficient sur la diagonale est a """
    A = a*np.eye(N)
    for i in range(N-1):
        A[i, i+1] = b
        A[i+1, i] = c
    return A


def U0(x):
    res = np.exp(-(x-2)**2) + np.exp(-(x+2)**2)
    return res


### Schéma explicite centré

# Mec = matriceM(N, 1-2c, c-d/2, d/2+c)

Mec = matriceM(1 - 2*c, c-d/2, d/2+c)

def SolExpliciteC(h, tau, Mec):
    """ Dans le schéma explicite centré U_n+1 = Mec U_n """
    ntfinal = int(Tmax/tau)
    ntdemi = int(ntfinal/2)
    X = np.linspace(-10,10,N)
    T = np.arange(ntfinal + 1)*tau
    U = np.zeros((ntfinal, N))
    U[0, : ] = U0(X)
    for i in range(ntfinal-1):
        U[ i+1, : ] = Mec@U[ i , : ]
    un = U[ntdemi, : ]
    return U, T, X, un


Ue, Te, Xe, ue = SolExpliciteC(h, tau, Mec)

plt.plot(Xe, Ue[0, :], label = 't=0', color='b')
plt.plot(Xe, ue, label = 't=2', color='r')
plt.plot(Xe, Ue[-1, :], label = 't=4', color='g')
plt.grid()
plt.legend()
plt.title('Explicite centré h=0.05 tau=0.0025')
plt.show()



### Schéma explicite décentré amont

# Med = matriceM(N, 1-2c-d, c, d+c)

Med = matriceM(1 - 2*c -d, c, d+c)

def SolExpliciteD(h, tau, Med):
    """ Dans le schéma explicite décentré amont U_n+1 = Med U_n """
    ntfinal = int(Tmax/tau)
    ntdemi = int(ntfinal/2)
    X = np.linspace(-10,10,N)
    T = np.arange(ntfinal + 1)*tau
    U = np.zeros((ntfinal, N))
    U[0, : ] = U0(X)
    for i in range(ntfinal-1):
        U[ i+1, : ] = Med@U[ i , : ]
    un = U[ntdemi, : ]
    return U, T, X, un

Ued, Ted, Xed, ued = SolExpliciteD(h, tau, Mec)

plt.plot(Xed, Ued[0, :], label = 't=0', color='r')
plt.plot(Xed, ued, label = 't=2', color='g')
plt.plot(Xed, Ued[-1, :], label='t=4', color='b')
plt.grid()
plt.legend()
plt.title('Explicite décentré amont h=0.05 tau=0.0025')
plt.show()



### Schéma implicite centré

# Mic = matriceM(N, 1+2c, -c+d/2, -c-d/2)

Mic = matriceM(1 + 2*c, -c + d/2, -c - d/2)

def SolImpliciteC(h, tau, Mic):
    """ Dans le schéma implicite centré Mic U_n+1 = U_n """
    ntfinal = int(Tmax/tau)
    ntdemi = int(ntfinal/2)
    X = np.linspace(-10,10,N)
    T = np.arange(ntfinal + 1)*tau
    U = np.zeros((ntfinal, N))
    U[0, : ] = U0(X)
    for i in range(ntfinal-1):
        U[ i+1, : ] = np.linalg.solve(Mic, U[ i , : ])
    un = U[ntdemi, : ]
    return U, T, X, un

Ui, Ti, Xi, ui = SolImpliciteC(h, tau, Mic)

plt.plot(Xi, Ui[0, :], label = 't=0', color='b')
plt.plot(Xi, ui, label = 't=2', color='r')
plt.plot(Xi, Ui[-1, :], label = 't=4', color='g')
plt.grid()
plt.legend()
plt.title('Implicite centré h=0.05 tau=0.0025')
plt.show()



### Schéma implicite décentré amont

# Mid = matriceM(N, 1+2c+d, -c, -c-d)

Mid = matriceM(1 + 2*c + d, -c, -c - d)

def SolImpliciteD(h, tau, Mid):
    """ Dans le schéma implicite décentré amont Mid U_n+1 = U_n """
    ntfinal = int(Tmax/tau)
    ntdemi = int(ntfinal/2)
    X = np.linspace(-10,10,N)
    T = np.arange(ntfinal + 1)*tau
    U = np.zeros((ntfinal, N))
    U[0, : ] = U0(X)
    for i in range(ntfinal-1):
        U[ i+1, : ] = np.linalg.solve(Mid, U[ i , : ])
    un = U[ntdemi, : ]
    return U, T, X, un

Uid, Tid, Xid, uid = SolImpliciteD(h, tau, Mid)

plt.plot(Xid, Uid[0, :], label = 't=0', color='r')
plt.plot(Xid, uid, label = 't=2', color='g')
plt.plot(Xid, Uid[-1, :], label = 't=4', color='b')
plt.grid()
plt.legend()
plt.title('Implicite décentré amont h=0.05 tau=0.0025')
plt.show()



### Schéma Crank-Nicholson centré

# CNcg = matriceM(N, 1+c, d/4-c/2, -c/2-d/4)
# CNcd = matriceM(N, 1-c, c/2-d/4, d/4+c/2)

CNcg = matriceM(1 + c, d/4 - c/2, -c/2 - d/4)
CNcd = matriceM(1 - c, -d/4 + c/2, c/2 + d/4)

def SolCNc(h, tau, CNcg, CNcd):
    """ Dans le schéma Crank-Nicolson centré CNcg U_n+1 = CNcd U_n """
    ntfinal = int(Tmax/tau)
    ntdemi = int(ntfinal/2)
    X = np.linspace(-10,10,N)
    T = np.arange(ntfinal + 1)*tau
    U = np.zeros((ntfinal, N))
    U[0, : ] = U0(X)
    for i in range(ntfinal-1):
        U[ i+1, : ] = np.linalg.solve(CNcg, CNcd@U[ i , : ])
    un = U[ntdemi, : ]
    return U, T, X, un

Ucn, Tcn, Xcn, ucn = SolCNc(h, tau, CNcg, CNcd)

plt.plot(Xcn, Ucn[0, :], label = 't=0', color='b')
plt.plot(Xcn, ucn, label = 't=2', color='r')
plt.plot(Xcn, Ucn[-1, :], label = 't=4', color='g')
plt.grid()
plt.legend()
plt.title('Crank-Nicolson centré h=0.05 tau=0.0025')
plt.show()



### Schéma Crank-Nicholson décentré amont

# CNdg = matriceM(N, 1+c+d/2, -c/2, -c/2-d/2)
# CNdd = matriceM(N, 1-c-d/2, c/2, c/2+d/2)

CN_cg = matriceM(1 + c + d/2, - c/2, -c/2 - d/2)
CN_cd = matriceM(1 - c - d/2, c/2, d/2 + c/2)

def SolCNd(h, tau, CN_cg, CN_cd):
    """ Dans le schéma Crank-Nicolson centré CNcg U_n+1 = CNcd U_n """
    ntfinal = int(Tmax/tau)
    ntdemi = int(ntfinal/2)
    X = np.linspace(-10,10,N)
    T = np.arange(ntfinal + 1)*tau
    U = np.zeros((ntfinal, N))
    U[0, : ] = U0(X)
    for i in range(ntfinal-1):
        U[ i+1, : ] = np.linalg.solve(CN_cg, CN_cd@U[ i , : ])
    ud = U[ntdemi, : ]
    return U, T, X, ud

U_cn, T_cn, X_cn, ud_cn = SolCNd(h, tau, CN_cg, CN_cd)

plt.plot(X_cn, U_cn[0, :], label = 't=0', color='r')
plt.plot(X_cn, ud_cn, label = 't=2', color='g')
plt.plot(X_cn, U_cn[-1, :], label = 't=4', color='b')
plt.grid()
plt.legend()
plt.title('Crank-Nicolson décentré amont h=0.05 tau=0.0025')
plt.show()
