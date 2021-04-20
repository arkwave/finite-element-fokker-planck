import numpy as np 
from fenics import IntervalMesh, FunctionSpace, DOLFIN_EPS, Constant, \
                   TrialFunction, TestFunction, Function, solve, plot, File, \
                   grad, dx, Expression, interpolate, lhs, rhs, inner, set_log_level, DirichletBC
import matplotlib.pyplot as plt
import itertools
import pandas as pd


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

    # isolate the bilinear and linear forms. 
    a, L = lhs(F), rhs(F)

    # initialize function to capture solution. 
    t = 0 
    u_h = Function(V)

    plt.figure(figsize=(15, 15))

    # time-stepping section
    for n in range(num_steps):
        t += dt 

        # compute solution 
        solve(a == L, u_h)
        u_n.assign(u_h)

        # Plot solutions intermittently
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


def plot_convergence_delta_t(L, t_0, t_final, u_true, nxs, data):
   # plotting errors vs delta_t for various fixed delta_xs
    fig = plt.figure(figsize=(10, 10))
    for m in [510]:
    # for i in range(len(nxs)):
        t_errors = []
        delta_x = L/float(m+1)

        relevant_data = data[data.nx == m]
        delta_ts = (t_final - t_0) / relevant_data.num_steps.values

        for _, row in relevant_data.iterrows():
            true_sol = u_true(row.xvals, t_final)
            err = np.linalg.norm(true_sol - row.sol)
            t_errors.append(err)

        # fig.add_subplot(len(nxs), 1, i+1)
        plt.semilogy(delta_ts[1:], t_errors[1:], label='$\Delta_x$=%.2f' % delta_x)

    plt.tight_layout() 
    plt.xlabel("$\Delta_t$")
    plt.ylabel("$u(x, t) - U$")
    plt.title("Convergence wrt $\Delta_t$")
    plt.grid() 
    plt.legend() 
    # plt.savefig("test_images/delta_t_convergence_test.png")
    plt.show() 


def plot_convergence_delta_x(L, t_0, t_final, u_true, times, data):
    # plotting errors vs delta_x for various fixed delta_ts
    plt.figure(figsize=(10, 10)) 

    for t in times:
        x_errors = []
        delta_t = (t_final-t_0)/float(t+1)
        relevant_data = data[data.num_steps == t]
        delta_xs = L / relevant_data.nx.values         
        for _, row in relevant_data.iterrows():
            true_sol = u_true(row.xvals, t_final)
            err = np.linalg.norm(true_sol - row.sol)
            x_errors.append(err)
        
        plt.loglog(delta_xs[1:], x_errors[1:], label='$\Delta_t$=%.2f' % delta_t)

    plt.xlabel("$\Delta_x$")
    plt.ylabel("$u(x, t) - U$")
    plt.title("Convergence wrt $\Delta_x$")
    plt.grid() 
    plt.legend() 
    # plt.savefig("test_images/delta_x_convergence_test.png")
    plt.show() 
    

        
def load_and_clean_results(path):
    data = pd.read_csv(path)
    data = data.replace('\n','', regex=True) 
    data = data.replace('\r', '', regex=True)

    for i in range(data.shape[0]):
        data.xvals[i] = np.asarray(np.matrix(data.xvals[i])).flatten() 
        data.sol[i] = np.asarray(np.matrix(data.sol[i])).flatten() 
        data.num_steps[i] = np.asarray(np.matrix(data.num_steps[i])).flatten() 
    
    return data 


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
    # result_df.to_csv("fpe/results_final.csv", index=False)

    # generate plots 
    u_true = lambda x, t: x + t
    plot_convergence_delta_x(400, 0.0, 100, u_true, times, result_df)
    plot_convergence_delta_t(400, 0.0, 100, u_true, nxs, result_df)


