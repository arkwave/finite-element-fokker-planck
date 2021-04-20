import numpy as np 
from fenics import IntervalMesh, FunctionSpace, DOLFIN_EPS, Constant, \
                   TrialFunction, TestFunction, Function, solve, plot, File, \
                   grad, dx, Expression, interpolate, lhs, rhs, inner, set_log_level, DirichletBC
import matplotlib.pyplot as plt
import itertools
import pandas as pd
# from joblib import Parallel, delayed 

# Modified from class notes to apply FEM to the Fokker-Planck Equation. 

# Create mesh and define function space

def run_fokker_planck(nx, num_steps, t_0 = 0, t_final=10):

    # define mesh
    mesh = IntervalMesh(nx, -200, 200)
    # define function space. 
    V = FunctionSpace(mesh, "Lagrange", 1)

    # Homogenous Neumann BCs don't have to be defined as they are the default in dolfin
    # define parameters.
    dt = (t_final-t_0) / num_steps

    # set mu and sigma 
    mu = Constant(-1)
    D = Constant(1)

    # define initial conditions u_0
    u_0 = Expression('x[0]', degree=1)

    # set U_n to be the interpolant of u_0 over the function space V. Note that 
    # u_n is the value of u at the previous timestep, while u is the current value. 
    u_n = interpolate(u_0, V)

    # Define variational problem
    u = TrialFunction(V)
    v = TestFunction(V)
    F = u*v*dx + dt*inner(D*grad(u), grad(v))*dx + dt*mu*grad(u)[0]*v*dx - inner(u_n, v)*dx
    a, L = lhs(F), rhs(F)

    # initialize function to capture solution. 
    t = 0 
    u_h = Function(V)

    plt.figure(figsize=(15, 15))

    for n in range(num_steps):
        t += dt 
        
        # compute solution 
        solve(a == L, u_h)
        u_n.assign(u_h)

        # Plot solution
        if n % (num_steps // 10) == 0 and num_steps > 0:

            plot(u_h, label='t = %s' % t)
    
    plt.legend() 
    plt.grid()
    plt.title("Finite Element Solutions to Fokker-Planck Equation with $\mu(x, t) = -(x+1)$ , $D(x, t) = e^t x^2$, $t_n$ = %s" % t_final)
    plt.ylabel("$u(x, t)$")
    plt.xlabel("x")
    plt.savefig("fpe/fokker-planck-solutions-mu.png")

    plt.clf() 

    # return the approximate solution evaluated on the coordinates, and the actual coordinates. 
    return u_n.compute_vertex_values(), mesh.coordinates() 


if __name__ == "__main__":

    set_log_level(40)

    nxs = np.arange(20, 1000, 70)
    times = np.arange(20, 1000, 70)

    params = list(itertools.product(nxs, times)) 

    result_lst = []

    for nx, num_steps in params:

        print("="*8 + " Running %s, %s " % (nx, num_steps) + '='*8)
        sol, xvals = run_fokker_planck(nx, num_steps, t_final=100)
        results = {'nx': nx, 
                   'num_steps': num_steps,
                   'xvals': xvals.flatten(), 
                   'sol': sol.flatten()}
        result_lst.append(results)

    result_df = pd.DataFrame(result_lst)
    result_df.to_csv("fpe/results_final.csv", index=False)


