#!~/.conda/envs/general/bin/python
"""
author : @Vasilis_Tsioulos

Functions python file used for the Quantum Mechanics course project1
"""

import numpy as np
import pandas as pd
import sympy as sp
import matplotlib.pyplot as plt
from scipy.integrate import quad

# Assign symbolic variables
r, k = sp.symbols('r k')

pi = np.pi
sqrt = lambda x : sp.sqrt(x)

# For Sr

# s orbitals
def Sr1s(z, r):
    return 2 * z**(3/2) * sp.exp(-z * r)

def Sr2s(z, r):
    return 2 / sqrt(3) * z**(5/2) * r * sp.exp(-z * r)

def Sr3s(z, r):
    return 2**(3/2)/(3 * sqrt(5)) * z**(7/2)* r**2 * sp.exp(-z * r)

def Sr4s(z, r):
    return 2/(3 * sqrt(35)) * z**(9/2) * r**3 * sp.exp(-z * r)

def Sr5s(z, r):
    return 2**(3/2) / (45 * sqrt(7)) * z**(11/2) * r**4 * sp.exp(-z * r)

# p orbitals

def Sr2p(z, r):
    return 2 / sqrt(3) * z**(5/2) * r * sp.exp(-z * r)

def Sr3p(z, r):
    return 2**(3/2) / (3 * sqrt(5)) * z**(7/2) * r**2 * sp.exp(-z * r)

def Sr4p(z, r):
    return 2 / (3 * sqrt(35)) * z ** (9/2) * r**3 * sp.exp(-z * r)

def Sr5p(z, r):
    return 2**(3/2) / (45 * sqrt(7)) * z**(11/2) * r**4 * sqrt(-z * r)

# d orbitals

def Sr3d(z, r):
    return 2**(3/2) / (3 * sqrt(5)) * z**(7/2) * r**2 * sp.exp(-z * r)

def Sr4d(z, r):
    return 2 / (3 * sqrt(35)) * z**(9/2) * r**3 * sp.exp(-z * r)


# For Sk

# s orbitals

def Sk1s(z, k):
    return 1 / (2 * pi)**(3/2) * 16 * pi * z**(5/2) / (z**2 + k**2)**2

def Sk2s(z, k):
    return 1 / (2 * pi)**(3/2) * 16 * pi * z**(5/2) * (3 * z**2 - k**2) / (sqrt(3) * (z**2 + k**2)**3)

def Sk3s(z, k):
    return 1 / (2 * pi)**(3/2) * 64 * sqrt(10) * pi * z**(9/2) * (z**2 - k**2) / (5 * (z**2 + k**2)**4)

def Sk4s(z, k):
    return 1 / (2 * pi)**(3/2) * 64 * pi * z**(9/2) * (5 * z**4 - 10 * z**2 * k**2 + k**4) / (sqrt(35) * (z**2 + k**2)**5)

def Sk5s(z, k):
    return 1 / (2 * pi)**(3/2) * 128 * sqrt(14) * pi * z**(13/2) * (3 * z**4 - 10 * z**2 + 3 * k**4) / (21 * (z**2 + k**2)**6)

# p orbitals

def Sk2p(z, k):
    return 1 / (2 *pi)**(3/2) * 64 * pi*k*z**(7/2) / (sqrt(3) * (z**2 + k**2)**3)

def Sk3p(z, k):
    return 1 / (2 * pi)**(3/2) * 64 * sqrt(10) * pi *k*z**(7/2) * (5 * z**2 - k**2) / (15 * (z**2 + k**2)**4)

def Sk4p(z, k):
    return 1 / (2 *pi)**(3/2) * 128 * pi*k*z**(11/2) * (5*z**2 - 3 * k**2) / (sqrt(35) * (z**2 + k**2)**5)

def Sk5p(z, k):
    return 1 / (2 * pi)**(3/2) * sqrt(14) * pi*k*z**(11/2) * (35 * z**4 - 42*z**2 * k**2 + 3*k**4) / (105 * (z**2 + k**2)**6)

# d orbitals

def Sk3d(z, k):
    return 1 / (2 * pi)**(3/2) * 128 * sqrt(10) * pi * k**2 * z**(9/2) / (5 * (z**2 + k**2)**4)

def Sk4d(z, k):
    return 1 / (2 * pi)**(3/2) * 128 * pi * k**2 * z**(9/2) * (7 * z**2 - k**2) / (sqrt(35) * (z**2 + k**2)**5)

def wavefunction(c_s, c_p, Z):
    """Calculate wavefunction properties from orbital coefficients

    Args:
        c_s (Dataframe): s-orbital coefficients
        c_p (Dataframe): p-orbital coefficients
        Z (int): Atomic number
    """

    class WaveFunction:
        def __init__(self):
            self.terms = []
        
        def add_term(self, func):
            self.terms.append(func)
            
        def __call__(self, x):
            return sum(f(x) for f in self.terms)

    def lambdify_wavefunction_terms(wf):
        # Converts symbolic terms to numeric numpy functions
        new_terms = []
        x = sp.symbols('x', real=True, positive=True)
        for f in wf.terms:
            expr = f(x)  # Evaluate symbolic expression at sym var x
            func_np = sp.lambdify(x, expr, modules='numpy')
            new_terms.append(func_np)
        wf.terms = new_terms

    # Initialize wavefunctions
    R1s = WaveFunction()
    K1s = WaveFunction()
    R2s = WaveFunction()
    K2s = WaveFunction()
    R2p = WaveFunction()
    K2p = WaveFunction()
    
    # s orbitals
    for _, row in c_s.iterrows():
        orb_type = int(row['orbitals'].replace('s', ''))
        z = row['z_coeff']
        f1 = row['coeff(1s/2p)']
        f2 = row.get('coeff(2s)', 0)

        # R1s/K1s terms
        if orb_type == 1:
            R1s.add_term(lambda r, z=z, f1=f1: f1 * Sr1s(z, r))
            K1s.add_term(lambda k, z=z, f1=f1: f1 * Sk1s(z, k))
        elif orb_type == 2:
            R1s.add_term(lambda r, z=z, f1=f1: f1 * Sr2s(z, r))
            K1s.add_term(lambda k, z=z, f1=f1: f1 * Sk2s(z, k))
        elif orb_type == 3:
            R1s.add_term(lambda r, z=z, f1=f1: f1 * Sr3s(z, r))
            K1s.add_term(lambda k, z=z, f1=f1: f1 * Sk3s(z, k))

        # R2s/K2s if f2 exists
        if f2 != 0:
            if orb_type == 1:
                R2s.add_term(lambda r, z=z, f2=f2: f2 * Sr1s(z, r))
                K2s.add_term(lambda k, z=z, f2=f2: f2 * Sk1s(z, k))
            elif orb_type == 2:
                R2s.add_term(lambda r, z=z, f2=f2: f2 * Sr2s(z, r))
                K2s.add_term(lambda k, z=z, f2=f2: f2 * Sk2s(z, k))
            elif orb_type == 3:
                R2s.add_term(lambda r, z=z, f2=f2: f2 * Sr3s(z, r))
                K2s.add_term(lambda k, z=z, f2=f2: f2 * Sk3s(z, k))

    # p orbitals if exist
    if not c_p.empty:
        for _, row in c_p.iterrows():
            orb_type = int(row['orbitals'].replace('p', ''))
            z  = row['z_coeff']
            f3 = row['coeff(1s/2p)']
            if orb_type == 2:
                R2p.add_term(lambda r, z=z, f3=f3: f3 * Sr2p(z, r))
                K2p.add_term(lambda k, z=z, f3=f3: f3 * Sk2p(z, k))

    # Lambdify wavefunction terms to numeric functions (critical fix)
    lambdify_wavefunction_terms(R1s)
    lambdify_wavefunction_terms(K1s)
    lambdify_wavefunction_terms(R2s)
    lambdify_wavefunction_terms(K2s)
    lambdify_wavefunction_terms(R2p)
    lambdify_wavefunction_terms(K2p)

    # Normalization check
    try:
        result_r1s, _ = quad(lambda r: R1s(r)**2 * r**2, 0, np.inf)
        result_k1s, _ = quad(lambda k: K1s(k)**2 * k**2, 0, np.inf)
    except Exception as e:
        print(f"Integration error: {str(e)}")
        return np.zeros(4)
    
    # Calculate electron density ρ(r) and momentum density n(k)
    factor = 1/(4*np.pi * Z)
    
    pr_terms = []
    nk_terms = []
    
    # 1s electrons (first 2 electrons)
    for _ in range(min(2, Z)):
        pr_terms.append(lambda r, R1s=R1s, factor=factor: factor * R1s(r)**2)
        nk_terms.append(lambda k, K1s=K1s, factor=factor: factor * K1s(k)**2)
    
    # 2s electrons (next 2 electrons if they exist)
    for _ in range(min(2, max(0, Z-2))):
        pr_terms.append(lambda r, R2s=R2s, factor=factor: factor * R2s(r)**2)
        nk_terms.append(lambda k, K2s=K2s, factor=factor: factor * K2s(k)**2)
    
    # 2p electrons (remaining electrons)
    for _ in range(max(0, Z-4)):
        pr_terms.append(lambda r, R2p=R2p, factor=factor: factor * R2p(r)**2)
        nk_terms.append(lambda k, K2p=K2p, factor=factor: factor * K2p(k)**2)
    
    pr = pr_terms
    nk = nk_terms

    # Plots
    r_values = np.linspace(0, 4, 100)
    k_values = np.linspace(0, 4, 100)
    
    pr_values = [sum(f(r) for f in pr) for r in r_values]
    nk_values = [sum(f(k) for f in nk) for k in k_values]
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(r_values, pr_values)
    plt.title(r'Electron Density Distribution $\rho(r)$')
    plt.xlabel('r')
    plt.ylabel(r'$\rho(r)$')
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(k_values, nk_values)
    plt.title(r'Momentum Density Distribution $n(k)$')
    plt.xlabel('k')
    plt.ylabel(r'$n(k)$')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

    # Entropy calculation with safe numeric log
    result_sr, _ = quad(lambda r: max(sum(f(r) for f in pr), 1e-20) * 
                               np.log(max(sum(f(r) for f in pr), 1e-20)) * r**2, 0, np.inf)
    Sr = -4 * np.pi * result_sr

    result_sk, _ = quad(lambda k: max(sum(f(k) for f in nk), 1e-20) * 
                               np.log(max(sum(f(k) for f in nk), 1e-20)) * k**2, 0, np.inf)
    Sk = -4 * np.pi * result_sk

    Sall = Sr + Sk

    result_r2, _ = quad(lambda r: sum(f(r) for f in pr) * r**4, 0, np.inf)
    r2 = 4 * np.pi * result_r2

    result_k2, _ = quad(lambda k: sum(f(k) for f in nk) * k**4, 0, np.inf)
    k2 = 4 * np.pi * result_k2

    Smax = 3 * (1 + np.log(np.pi)) + (3/2) * np.log(4/9 * r2 * k2)

    return np.array([Sr, Sk, Sall, Smax])

