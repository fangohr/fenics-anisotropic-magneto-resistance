import dolfin as df
import numpy as np

def amr(mesh, m, DirichletBoundary, g, d, s0=1, alpha=10):
    V = df.FunctionSpace(mesh, "CG", 1)
    VV = df.VectorFunctionSpace(mesh, 'CG', 1, 3)

    # Define boundary condition
    bcs = df.DirichletBC(V, g, DirichletBoundary())

    def sigma(u):
        E = -df.grad(u)
        costheta = df.dot(m, E)/df.sqrt(df.dot(E, E))
        return s0/(1 + alpha*costheta**2)

    # Define variational problem for Picard iteration
    u = df.TrialFunction(V)
    v = df.TestFunction(V)
    u_k = df.interpolate(g, V)  # previous (known) u
    a = df.inner(sigma(u_k)*df.nabla_grad(u), df.nabla_grad(v))*df.dx
    f = df.Constant(0.0)
    L = f*v*df.dx

    # Picard iterations
    u = df.Function(V)     # new unknown function
    eps = 1.0           # error measure ||u-u_k||
    tol = 1.0E-5        # tolerance
    iter = 0            # iteration counter
    maxiter = 25        # max no of iterations allowed
    while eps > tol and iter < maxiter:
        iter += 1
        df.solve(a == L, u, bcs)
        diff = u.vector().array() - u_k.vector().array()
        eps = np.linalg.norm(diff, ord=np.Inf)
        print 'iter=%d: norm=%g' % (iter, eps)
        u_k.assign(u)   # update for next iteration


    # Plot solution and solution gradient
    df.plot(u, title="Solution")
    df.plot(-sigma(u)*df.grad(u), title="current density")
    df.interactive()

    return df.project(-sigma(u)*df.grad(u), VV)
