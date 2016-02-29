import dolfin as df
import numpy as np

def amr(mesh, m, DirichletBoundary, g, mesh2d, s0=1, alpha=1):
    """Function for computing the Anisotropic MagnetoResistance (AMR),
    using given magnetisation configuration."""
    # Scalar and vector function spaces.
    V = df.FunctionSpace(mesh, "CG", 1)
    VV = df.VectorFunctionSpace(mesh, 'CG', 1, 3)

    # Define boundary conditions.
    bcs = df.DirichletBC(V, g, DirichletBoundary())

    # Nonlinear conductivity.
    def sigma(u):
        E = -df.grad(u)
        costheta = df.dot(m, E)/(df.sqrt(df.dot(E, E))*df.sqrt(df.dot(m, m)))
        return s0/(1 + alpha*costheta**2)

    # Define variational problem for Picard iteration.
    u = df.TrialFunction(V)  # electric potential
    v = df.TestFunction(V)
    u_k = df.interpolate(df.Expression('x[0]'), V)  # previous (known) u
    a = df.inner(sigma(u_k)*df.grad(u), df.grad(v))*df.dx
    
    # RHS to mimic linear problem.
    f = df.Constant(0.0)  # set to 0 -> nonlinear Poisson equation.
    L = f*v*df.dx

    u = df.Function(V)  # new unknown function
    eps = 1.0           # error measure ||u-u_k||
    tol = 1.0e-20       # tolerance
    iter = 0            # iteration counter
    maxiter = 50        # maximum number of iterations allowed
    while eps > tol and iter < maxiter:
        iter += 1
        df.solve(a == L, u, bcs)
        diff = u.vector().array() - u_k.vector().array()
        eps = np.linalg.norm(diff, ord=np.Inf)
        print 'iter=%d: norm=%g' % (iter, eps)
        u_k.assign(u)   # update for next iteration

    j = df.project(-sigma(u)*df.grad(u), VV)
    return u, j, compute_flux(j, mesh2d)


def compute_flux(j, mesh2d):
    """Computes the flux (current) of a vector function (current density)
    through a surface given by mesh2d."""
    # Create two-dimensional functionspace and function onto which
    # the three-dimensional field will be sampled.
    V_2d = df.FunctionSpace(mesh2d, 'CG', 1)
    j_2d = df.Function(V_2d)
    j_2d_array = j_2d.vector().array()
    n_nodes = mesh2d.num_vertices()  # number of mesh nodes
    coords = mesh2d.coordinates()
    for i in xrange(len(coords)):
        sampling_point = coords[i]
        sampled_value = j(0, sampling_point[0], sampling_point[1])

        j_2d_array[i] = sampled_value[0]

    # Convert function array back to dolfin function.
    j_2d.vector().set_local(j_2d_array)

    # Integrate over the domain and return current (flux).
    return df.assemble(j_2d*df.dx)
