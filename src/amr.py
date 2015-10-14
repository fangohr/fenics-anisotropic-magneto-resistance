import dolfin as df

def amr(mesh, m, DirichletBoundary, g, d, s0=1, alpha=1):
    V = df.FunctionSpace(mesh, "CG", 1)

    # Define boundary condition
    bc = df.DirichletBC(V, g, DirichletBoundary())

    # Define variational problem
    u = df.Function(V)
    v = df.TestFunction(V)
    E = -df.grad(u)
    costheta = df.dot(m, E)
    sigma = s0/(1 + alpha*costheta**2)
    F = df.inner(sigma*df.grad(u), df.grad(v))*df.dx

    # Compute solution
    df.solve(F == 0, u, bc, solver_parameters={"newton_solver":
                                               {"relative_tolerance": 1e-6}})

    # Plot solution and solution gradient
    df.plot(u, title="Solution")
    df.plot(sigma*df.grad(u), title="Solution gradient")
    df.interactive()
