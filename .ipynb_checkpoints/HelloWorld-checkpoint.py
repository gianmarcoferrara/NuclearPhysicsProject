import numpy as np
import matplotlib.pyplot as plt

# Constants
pi = np.pi
tolerance = 1e-10  # Convergence criterion
max_iterations = 1000  # Max iterations for eigenvalue search

def potential(x):
    return 0.5 * x**2  # Harmonic potential

def numerov(x, f, y0, y1):
    """ Numerov integration algorithm """
    y = np.zeros_like(x)
    y[0], y[1] = y0, y1
    for i in range(1, len(x) - 1):
        y[i+1] = ((12 - 10 * f[i]) * y[i] - f[i-1] * y[i-1]) / f[i+1]
    return y

def count_nodes(y):
    """ Counts the number of sign changes in y """
    return np.sum(y[:-1] * y[1:] < 0)

def solve_schrodinger(x, nodes, e_trial=0):
    dx = x[1] - x[0]
    f = 1 - (2/12) * (2 * (potential(x) - e_trial)) * dx**2
    
    # Initial conditions
    if nodes % 2 == 0:
        y0, y1 = 1.0, (12 - 10 * f[0]) * 1.0 / f[1]
    else:
        y0, y1 = 0.0, dx
    
    y = numerov(x, f, y0, y1)
    return y, count_nodes(y)

def find_eigenvalue(x, nodes):
    elw, eup = np.min(potential(x)), np.max(potential(x))
    e = 0.5 * (elw + eup)
    
    for _ in range(max_iterations):
        y, ncross = solve_schrodinger(x, nodes, e)
        if np.abs(eup - elw) < tolerance:
            return e, y
        if ncross > nodes // 2:
            eup = e  # Too many nodes, decrease energy
        else:
            elw = e  # Too few nodes, increase energy
        e = 0.5 * (elw + eup)
    return e, y

# User input
xmax = float(input("Max value of x (e.g., 10): "))
mesh = int(input("Number of grid points (e.g., 200): "))
x = np.linspace(0, xmax, mesh)

while True:
    nodes = int(input("Number of nodes (-1 to exit): "))
    if nodes < 0:
        break
    
    e, y = find_eigenvalue(x, nodes)
    print(f"Eigenvalue for {nodes} nodes: {e:.10f}")
    
    plt.plot(x, y, label=f"n={nodes}, e={e:.4f}")
    
plt.xlabel("x")
plt.ylabel("Wavefunction")
plt.legend()
plt.title("Quantum Harmonic Oscillator Solutions")
plt.show()
