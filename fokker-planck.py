import numpy as np 
from fenics import IntervalMesh, FunctionSpace, DOLFIN_EPS, Constant, \
                   TrialFunction, TestFunction, Function, solve, plot, File, \
                   dot, grad, dx, Expression, interpolate, lhs, rhs, inner, \
                       TensorFunctionSpace, project, DirichletBC, SubDomain
import matplotlib.pyplot as plt

# Modified from class notes to apply FEM to the Fokker-Planck Equation. 

# Create mesh and define function space

def run_fokker_planck(nx, num_steps, mu=0, sigma=1, t_0 = 0, t_final=1,
                      plot_convergence=True):

    mu_const = Constant(mu)

    # define mesh
    mesh = IntervalMesh(nx, -1, 1)
    # define function space. 
    V = FunctionSpace(mesh, "Lagrange", 1)

    # Homogenous Neumann BCs don't have to be defined as they are the default in dolfin

    # define parameters.
    dt = (t_final-t_0) / num_steps

    # derived coefficients. 
    c = Constant(sigma**2 / 2)

    # define initial conditions u_0
    u_0 = Expression('exp(-a*pow(x[0], 2))',
                    degree=1, a=0.5)

    # set U_n to be the interpolant of u_0 over the function space V. Note that 
    # u_n is the value of u at the previous timestep, while u is the current value. 
    u_n = interpolate(u_0, V)

    # Define variational problem
    u = TrialFunction(V)
    v = TestFunction(V)

    if mu > 0:
        F = u*v*dx + dt*inner(c*grad(u), grad(v))*dx + dt*mu_const*grad(u)[0]*v*dx - inner(u_n, v)*dx
    else:
        F = u*v*dx + dt*c*inner(grad(u), grad(v))*dx - inner(u_n, v)*dx

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

        # # Save solution in VTK format

        # Plot solution
        if n % (num_steps // 10) == 0 and num_steps > 0:

            plot(u_h, label='t = %s' % t)
    
    
    filename = "fpe/fokker-planck-" + str(t) + ".pvd"
    file = File(filename)
    file << u_n

    plt.legend() 
    plt.grid()
    plt.title("Finite Element Solutions to Fokker-Planck Equation with $\mu$ = %s, $\sigma$ = %s" % (mu, sigma))
    plt.ylabel("$u(x, t)$")
    plt.xlabel("x")
    plt.savefig("fpe/fokker-planck-solutions.png")

    return u_n 


if __name__ == "__main__":
    nx = 500
    num_steps = 100
    sol = run_fokker_planck(nx, num_steps, mu=2, sigma=2, t_final=0.1)


