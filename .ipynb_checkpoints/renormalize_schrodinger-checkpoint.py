#Radial Schrodinger equation using Numerov method

import numpy as np
import matplotlib.pyplot as plt
import sys
import pandas as pd
from scipy.special import erf
from scipy.optimize import minimize

def prepare_grid(n_points, x_min, dx):

    #preparing x-array with constant step
    x = np.linspace(x_min, x_min + ((n_points - 1) * dx), n_points)
    x = np.append(x, x[n_points - 1] + dx) #another point for convenience of indices

    #generate r, sqrt_r, and r^2 based on logarithmic x grid
    r = np.exp(x)
    sqrt_r = np.sqrt(r)
    r2 = np.power(r, 2)

    #print grid information
    print("Radial grid information:\n")
    print("dx = ", dx)
    print("x_min = ", x_min)
    print("n_points = ", n_points)
    print("r(0) = ", r[0])
    print("r(n_points) = ", r[n_points])
    print("-----------------------------------------------\n")

    return r, sqrt_r, r2




def init_potential_coulomb(r):

    #definition of the potential
    v_pot = -2/r #Z=1, (Rydberg) atomic units

    #saving the potential to CSV
    df_pot = pd.DataFrame({"r": r, "V(r)": v_pot})
    df_pot.to_csv("potentials/coulomb_potential.csv", index=False)
    print("potential saved in 'potentials/coulomb_potential.csv'\n")

    
    #plotting the potential
    plt.figure()
    plt.plot (r, v_pot, label=r'V(r), atomic units (Ry)', color ='b')
    plt.xlabel("r")
    plt.ylabel("V(r)")
    plt.title("coulomb potential")
    plt.grid(True)
    plt.legend()
    plt.xlim(0, 0.01)
    plt.savefig("potentials/coulomb_potential_plot.pdf")
    plt.close()
    
    return v_pot

def init_potential_short(r):

    #definition of the potential
    v_pot = -2/r - (np.sqrt(2) * np.exp(-r)) / r 

    #saving the potential to CSV
    df_pot = pd.DataFrame({"r": r, "V(r)": v_pot})
    df_pot.to_csv("potentials/short_potential.csv", index=False)
    print("potential saved in 'potentials/short_potential.csv'\n")

    
    #plotting the potential
    plt.figure()
    plt.plot (r, v_pot, label=r'V(r), atomic units (Ry)', color ='b')
    plt.xlabel("r")
    plt.ylabel("V(r)")
    plt.title("potential with short range term")
    plt.grid(True)
    plt.legend()
    plt.xlim(0, 0.01)
    plt.savefig("potentials/short_potential_plot.pdf")
    plt.close()

    return v_pot




def solve_schrodinger(n_points, dx, v_pot, r2, r, sqrt_r, n, l):
    #solve the radial schrodinger equation on a logarithmic
    #grid by Numerov method - atomic units (Ry)

    eps = 1E-10 #tolerance for eigenvalue
    n_iter = 200

    #useful quantities
    ddx12 = (dx * dx) / 12
    sqlhf = (l + 0.5) * (l + 0.5) #
    x2l2 = 2 * l + 2
    
    #initial lower and upper bounds to the eigenvalue
    e_upp = v_pot[n_points]
    e_low = e_upp

    for j in range(0, n_points + 1):
            e_low = np.minimum(e_low, sqlhf / r2[j] + v_pot[j])

    if (e_upp - e_low < eps):
        print("error in solving schrodinger: e_upp and e_low coincide", file = sys.stderr)
        sys.exit(1)
            
    e = (e_low + e_upp) * 0.5 #first rough estimate

    f = np.zeros(n_points + 1) #f-function for Numerov
    
    de = 1E10 #any number greater than eps
    
    class_inv = -1 #index of classical inversion point
    
    #start loop to find energy eigenvalue
    i = 0
    while i  < n_iter and np.absolute(de) > eps:
        #set up the f-function (in a way to determine the position of its last change of sign)

        f[0] = ddx12 * ((r2[0] * (v_pot[0] - e)) + sqlhf)
        for j in range(1, n_points + 1):
            f[j] = ddx12 * ((r2[j] * (v_pot[j] - e)) + sqlhf)

            #if f[j] is exactly zero (unlikely) the change of sign is not observed 
            #trick to prevent missing change of sign
            if (f[j] == 0.):
                f[j] = 1E-20
                
            # f > 0 approximately means classically forbidden region
            # f < 0      ''        ''       ''      allowed     ''
            #take the index of classical inversion
            if np.sign(f[j]) != np.sign(f[j - 1]):
                class_inv = j

        if class_inv < 0 or class_inv >= n_points - 2:
            print(f"{class_inv:4d} {n_points:4d} {n}")
            print("error in solving schrodinger: last change of sign too far", file = sys.stderr)
            sys.exit(1)

        #rewrite the f-function how required by numerov method
        f = 1 - f

        y = np.zeros(n_points + 1) #wavefunction
        nodes = n - l - 1
        
        #wavefunction in the first two points
        y[0] = 1E-12
        y[1] = 1E-5

        #outward integration with node counting
        n_cross = 0
        for j in range(1, class_inv):
            y[j + 1] = ((12. - f[j] * 10.) * y[j] - (f[j - 1] * y[j - 1])) / f[j + 1]
            if np.sign(y[j]) != np.sign(y[j + 1]):
                n_cross += 1

        scale_factor = y[class_inv] #value of the wavefunction at classical turning point, to match outward and inward
        
        #check the number of crossings
        if (n_cross != nodes):
            #incorrect number of nodes, adjusting eigenvalue
            if (n_cross > nodes):
                e_upp = e
            else:
                e_low = e

            e = (e_upp + e_low) * 0.5

        else:
            #correct number of nodes, we can perform inward integration.

            #determination of the wavefunction in last two points
            #assuming y[n_points + 1] = 0 and y[n_points] = dx
            #y[n_points] = dx
            #y[n_points - 1] = (12. - f[n_points] * 10.) * y[n_points] / f[n_points - 1]
            y[n_points] = 0
            y[n_points - 1] = dx

            #inward integration
            for j in range(n_points - 1, class_inv, -1):
                y[j - 1] = ((12. - f[j] * 10.) * y[j] - (f[j + 1] * y[j + 1])) / f[j - 1]
                if (y[j - 1] > 1E10):
                    for m in range(n_points, j - 2, -1):
                        y[m] /= y[j - 1]

            #rescale the function to match at the classical turning point
            scale_factor /= y[class_inv]
            for j in range(class_inv, n_points + 1):
                y[j] *= scale_factor


            #normalize on the segment
            norm = 0.

            for j in range(0, n_points + 1):
                norm += y[j] * y[j] * r2[j] * dx

            norm = np.sqrt(norm)

            for j in range(0, n_points + 1):
                y[j] /= norm

            #find the value of the cusp at the matching point
            j = class_inv
            y_cusp = (y[j - 1] * f[j - 1] + f[j + 1] * y[j + 1] + f[j] * 10. * y[j]) / 12.
            df_cusp = f[j] * ((y[j] / y_cusp) - 1.)

            # eigenvalue update using perturbation theory
            de = df_cusp / ddx12 * y_cusp * y_cusp * dx
            if (de > 0.):
                e_low = e
            if (de < 0.):
                e_upp = e

            e = e + de
            #prevent e to go out of bounds ( e > e_upp or e < e_low)
            #could happen far from convergence
            e = np.minimum(e, e_upp)
            e = np.maximum(e, e_low)

        i += 1
    
    #convergence not achived
    if (np.abs(de) > eps):
        if n_cross != nodes:
            print(f"n_cross={n_cross:4d} nodes={nodes:4d} class_inv={class_inv:4d} " 
                f"e={e:16.8e} e_low={e_low:16.8e} e_upp={e_upp:18.8e}", file=sys.stderr)
        else:
            print(f"e={e:16.8e} de={de:16.8e}", file=sys.stderr)

        print(f"solve_schrodinger not converged after {n_iter} iterations", file=sys.stderr)
        sys.exit(1)

    #convergence achived
    #print(f"convergence for n = {n} achived at iter # {i:4d}, de = {de:16.8e}, e = {e:16.8e}\n")

    #compute phase shift ar r=100
    i_100 = 1261 #index corresponding to r = 100
    phase_shift = np.arcsin(y[i_100]) - np.sqrt(np.absolute(e)) * r[i_100] + (l * np.pi)/2

    return e, y, phase_shift




def analysis(n_max, pot_name, e_n, p_n, y_n, n_points, dx, v_pot, r2 , r, sqrt_r):
    #quantic numbers
    n=1
    l=0 #fixed, considering only s-wave

    for i in range(1, n_max +1):
        e_n[i-1], y_n[i-1], p_n[i-1] = solve_schrodinger(n_points, dx, v_pot, r2, r, sqrt_r, n ,l)

        #saving and plotting eigenfunctions
        y_df = pd.DataFrame({"r": r, "y": y_n[i-1]})
        y_df.to_csv(f"eigenfunctions/eigenfunction_{pot_name}_{i}.csv", index=False)

        plt.figure()
        plt.plot(r, sqrt_r * y_n[i-1], label = r'$\chi$', color = 'red')
        plt.xlabel("r")
        plt.ylabel(r'$\chi(r)$')
        plt.title(f"{pot_name} wavefunction n = {i}")
        plt.legend()
        plt.grid(True)
        plt.savefig(f"eigenfunctions/{pot_name}_wavefunction_{i}.pdf")
        plt.close() #to save memory
        
        n += 1 #next energy level

    n_arr = np.arange(1, n_max + 1)

    #saving eigenvalues, phase shifts
    e_df = pd.DataFrame({"n": n_arr, "E": e_n})
    print("-----------------------------------------------\n")
    print(f"{pot_name} energy eigenvalues\n")
    print(e_df.to_markdown(index=False))
    e_df.to_csv(f"energy/energy_eigenvalues_{pot_name}.csv", index=False)

    p_df = pd.DataFrame({"n": n_arr, "phase shift": p_n})
    print("-----------------------------------------------\n")
    print(f"{pot_name} phase shifts at r = 100\n")
    print(p_df.to_markdown(index=False))
    p_df.to_csv(f"phase_shift/phase_shift_{pot_name}.csv", index=False)

    return e_n, y_n, p_n




#grid parameters
r_max = 400
x_min = -8. #corresponds to r_min == 3 * 1E-4 Bohr radii
dx = 0.01 #grid spacing
n_max = 10 # maximum energy level (# of nodes limited by range)

#number of points of the grid
n_points = int((np.log(r_max) - x_min) / dx)

#initialize logarithmic grid
r, sqrt_r, r2 = prepare_grid(n_points, x_min, dx)

#initialize potential
v_pot_1 = init_potential_coulomb(r)
v_pot_1_name = "coulomb"

#initialize arrays for eigenvalues, eigenfunctions and phase shifts
e_coulomb = np.zeros(n_max)
y_coulomb = np.zeros((n_max, n_points + 1))
ph_shift_coulomb = np.zeros(n_max)

e_coulomb, y_coulomb, ph_shift_coulomb = analysis(n_max, v_pot_1_name, e_coulomb, ph_shift_coulomb, y_coulomb, n_points, dx, v_pot_1, r2 , r, sqrt_r)

#comparison with analytical values -1/n^2
e_theo = np.arange(1, n_max + 1)
e_theo = -1 / (e_theo ** 2)

plt.figure()
plt.plot(np.absolute(e_theo), np.absolute(e_coulomb - e_theo)/ np.absolute(e_theo) , marker = 'o', linestyle = '--', label = "relative error coulomb", color = 'blue')
plt.xlabel(r'$|E|$')
plt.ylabel(r'$|\Delta E / E|$')
plt.title(f"relative error (with analytical values) for energy eigenvalues")
plt.legend()
plt.grid(True)
plt.xscale("log")
plt.yscale("log")
plt.savefig(f"relative_errors/relative_error_coulomb_eigenvalues.pdf")
plt.show()
plt.close()




#initialize potential
v_pot_2 = init_potential_short(r)
v_pot_2_name = "short"

#initialize arrays for eigenvalues, eigenfunctions and phase shifts
e_short = np.zeros(n_max)
y_short = np.zeros((n_max, n_points + 1))
ph_shift_short = np.zeros(n_max)

e_short, y_short, ph_shift_short = analysis(n_max, v_pot_2_name, e_short, ph_shift_short, y_short, n_points, dx, v_pot_2, r2 , r, sqrt_r)




#lowest-energy data is the one corresponing to n_max = 10
c = np.sqrt(np.pi) * (e_short[n_max - 1] * np.power(n_max, 3) + n_max)

print("c = ", c)




#coulomb eigenvalues in e_theo

#approximate eigenvalues
e_app = np.arange(1,n_max + 1)
e_app = - 1/ np.power(e_app, 2) + c / (np.sqrt(np.pi) * np.power(e_app, 3))

#plotting relative errors
plt.figure()
plt.plot(np.absolute(e_short), np.absolute(e_short - e_theo)/ np.absolute(e_short) , marker = 'o', linestyle = '--', label = "coulomb", color = 'red')
plt.plot(np.absolute(e_short), np.absolute(e_short - e_app)/ np.absolute(e_short) , marker = 'o', linestyle = '--', label = "delta, first-order", color = 'blue')
plt.xlabel(r'$|E|$')
plt.ylabel(r'$|\Delta E / E|$')
plt.xscale("log")
plt.yscale("log")
plt.title(f"relative error for energy eigenvalues")
plt.legend()
plt.grid(True)
plt.savefig(f"relative_errors/relative_error_naive_approx_eigenvalues.pdf")
plt.show()
plt.close()




def init_potential_effective(r, a, c, d_1):

    #definition of the potential
    v_pot = -2/r * erf(r / (np.sqrt(2) * a)) + c * np.power(a,2) * np.exp(-np.power(r,2) / (2 * np.power(a,2))) / (np.power(2 * np.pi, 1.5) * np.power(a,3)) + d_1 * np.power(a,4) * ((np.power(r,2) / np.power(a,2)) - 3) * np.exp(-np.power(r,2) / (2 * np.power(a,2))) / (np.power(2 * np.pi, 1.5) * np.power(a,5)) 

    #saving the potential to CSV
    df_pot = pd.DataFrame({"r": r, "V(r)": v_pot})
    if d_1 == 0:
        df_pot.to_csv(f"potentials/effective_potential_a2_cut{a}.csv", index=False)
    else:
        df_pot.to_csv(f"potentials/effective_potential_a4_cut{a}.csv", index=False)
    
    #plotting the potential
    plt.figure()
    plt.plot (r, v_pot, label=r'V(r), atomic units (Ry)', color ='b')
    plt.xlabel("r")
    plt.ylabel("V(r)")
    plt.title("effective potential")
    plt.grid(True)
    plt.legend()
    plt.xlim(0, 0.01)
    if d_1 == 0:
        plt.savefig(f"potentials/effective_potential_a2_cut{a}_plot.pdf")
    else:
        plt.savefig(f"potentials/effective_potential_a4_cut{a}_plot.pdf")
    plt.close()

    return v_pot




a = [1,2,6,10] #cutoff

#tuning of the parameter (lowest energy data - phase shift)
c_a2 = -50 # initial guess of the parameter a2

#initial guess of parameters
init_parameters = [-70, -1] #[c, d_1] a4

#cost function for a2-theory
def cost_function_1(c, a, n_points, dx, r2, r, sqtr_r, n_max):
    v_pot = init_potential_effective(r, a, c, 0) #imposing d_1 = 0
    e, y, phase_shift = solve_schrodinger(n_points, dx, v_pot, r2, r, sqrt_r, n_max, 0)
    return np.absolute(ph_shift_short[-1] - phase_shift)
    
#cost function for a4-theory
def cost_function_2(parameters, a, n_points, dx, r2, r, sqtr_r, n_max):
    c = parameters[0]
    d_1 = parameters[1]
    v_pot = init_potential_effective(r, a, c, d_1)
    e, y, phase_shift = solve_schrodinger(n_points, dx, v_pot, r2, r, sqrt_r, n_max, 0)
    return np.absolute(ph_shift_short[-1] - phase_shift)
    
plt.figure(1)
plt.figure(2)
plt.figure(3)
plt.figure(4)
colors =['blue', 'red', 'green', 'black']

#later we want to use again results for a=1
e_eff_a2_cut1 = np.zeros(n_max)
e_eff_a4_cut1 = np.zeros(n_max)

for a_value, color in zip(a, colors):
    #compute parameter
    result1 = minimize(cost_function_1, c_a2, method='nelder-mead', args=(a_value, n_points, dx, r2, r, sqrt_r, n_max), options={'maxiter': 10000,'maxfev': 10000,'xatol': 1E-8, 'disp': True})
    result2 = minimize(cost_function_2, init_parameters, method='nelder-mead', args=(a_value, n_points, dx, r2, r, sqrt_r, n_max), options={'maxiter': 10000,'maxfev': 10000,'xatol': 1E-8, 'disp': True})

    c_a2 = result1.x
    parameters_a4 = result2.x
    c_a4 = parameters_a4[0]
    d_1 = parameters_a4[1]
    print(f"----------------------|a = {a_value}|----------------------\n")
    print(f"(a^2 theory):    c = {c_a2} \n")
    print(f"(a^4 theory): c = {c_a4}, d_1 = {d_1}\n")

    #initialize potential
    v_pot_3 = init_potential_effective(r, a_value, c_a2, 0)
    v_pot_3_name = f"effective_a^2_{a_value}"
    v_pot_4 = init_potential_effective(r, a_value, c_a4, d_1)
    v_pot_4_name = f"effective_a^4_{a_value}"

    #initialize arrays for eigenvalues, eigenfunctions and phase shifts
    e_eff_a2 = np.zeros(n_max)
    y_eff_a2 = np.zeros((n_max, n_points + 1))
    ph_shift_eff_a2 = np.zeros(n_max)
    e_eff_a4 = np.zeros(n_max)
    y_eff_a4 = np.zeros((n_max, n_points + 1))
    ph_shift_eff_a4 = np.zeros(n_max)

    e_eff_a2, y_eff_a2, ph_shift_eff_a2 = analysis(n_max, v_pot_3_name, e_eff_a2, ph_shift_eff_a2, y_eff_a2, n_points, dx, v_pot_3, r2 , r, sqrt_r)
    e_eff_a4, y_eff_a4, ph_shift_eff_a4 = analysis(n_max, v_pot_4_name, e_eff_a4, ph_shift_eff_a4, y_eff_a4, n_points, dx, v_pot_4, r2 , r, sqrt_r)

    #out of this scope we want to use again results for a=1
    if a_value == 1:
        e_eff_a2_cut1 = e_eff_a2
        e_eff_a4_cut1 = e_eff_a4
        
    #plotting relative errors for phase shifts and enrgies
    #energy
    plt.figure(1)
    plt.plot(np.absolute(e_short), np.absolute(e_short - e_eff_a2)/ np.absolute(e_short) , marker = 'o', linestyle = '--', label = f"a = {a_value}", color = color)
    plt.xlabel(r'$|E|$')
    plt.ylabel(r'$|\Delta E / E|$')
    plt.xscale("log")
    plt.yscale("log")
    plt.title(f"relative error for energy eigenvalues - a^2 theory")
    plt.legend()
    plt.grid(True)
    if a_value == 10:
        plt.savefig(f"relative_errors/relative_error_energy_a2.pdf")

    plt.figure(2)
    plt.plot(np.absolute(e_short), np.absolute(e_short - e_eff_a4)/ np.absolute(e_short) , marker = 'o', linestyle = '--', label = f"a = {a_value}", color = color)
    plt.xlabel(r'$|E|$')
    plt.ylabel(r'$|\Delta E / E|$')
    plt.xscale("log")
    plt.yscale("log")
    plt.title(f"relative error for energy eigenvalues - a^4 theory")
    plt.legend()
    plt.grid(True)
    if a_value == 10:
        plt.savefig(f"relative_errors/relative_error_energy_a4.pdf")
    
    #phase_shift
    plt.figure(3)
    plt.plot(np.absolute(e_short), np.absolute(ph_shift_short - ph_shift_eff_a2) , marker = 'o', linestyle = '--', label = f"a = {a_value}", color = color)
    plt.xlabel(r'$|E|$')
    plt.ylabel(r'$|\Delta \delta(E)|$')
    plt.xscale("log")
    plt.title(f"error for phase shifts - a^2 theory")
    plt.legend()
    plt.grid(True)
    if a_value == 10:
        plt.savefig(f"relative_errors/relative_error_ph_shift_a2.pdf")

    plt.figure(4)
    plt.plot(np.absolute(e_short), np.absolute(ph_shift_short - ph_shift_eff_a4) , marker = 'o', linestyle = '--', label = f"a = {a_value}", color = color)
    plt.xlabel(r'$|E|$')
    plt.ylabel(r'$|\Delta \delta(E)|$')
    plt.xscale("log")
    plt.title(f"error for phase shifts - a^4 theory")
    plt.legend()
    plt.grid(True)
    if a_value == 10:
        plt.savefig(f"relative_errors/relative_error_ph_shift_a4.pdf")        




plt.figure()
plt.plot(np.absolute(e_short), np.absolute(e_short - e_theo)/ np.absolute(e_short) , marker = 'o', linestyle = '--', label = f"coulomb", color = 'black')
plt.plot(np.absolute(e_short), np.absolute(e_short - e_app)/ np.absolute(e_short) , marker = 'o', linestyle = '--', label = f"delta - perturbation", color = 'blue')
plt.plot(np.absolute(e_short), np.absolute(e_short - e_eff_a4_cut1)/ np.absolute(e_short) , marker = 'o', linestyle = '--', label = f"a^4 theory", color = 'red')
plt.xlabel(r'$|E|$')
plt.ylabel(r'$|\Delta E / E|$')
plt.xscale("log")
plt.yscale("log")
plt.title(f"relative error for energy eigenvalues - comparison")
plt.legend()
plt.grid(True)
plt.savefig(f"relative_errors/relative_error_energy_comparison.pdf")
plt.show()
plt.close()

