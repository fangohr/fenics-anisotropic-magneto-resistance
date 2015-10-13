import dolfin as df

d = 10

# Sub domain for Dirichlet boundary condition
class DirichletBoundary(df.SubDomain):
    def inside(self, x, on_boundary):
        return (abs(x[0] - d) < df.DOLFIN_EPS or abs(x[0]) < df.DOLFIN_EPS) and on_boundary

# Create mesh and define function space
mesh = df.BoxMesh(df.Point(0, 0, 0), df.Point(d, d, d), 15, 15, 15)
df.File("mesh.pvd") << mesh

V = df.FunctionSpace(mesh, "CG", 1)

# Define boundary condition
g = df.Expression("x[0]")
bc = df.DirichletBC(V, g, DirichletBoundary())

# Define variational problem
u = df.Function(V)
v = df.TestFunction(V)
m = df.Constant((0, 0, 1))
F = df.inner((1/(1 + df.dot(m, df.grad(u))))*df.grad(u), df.grad(v))*df.dx

# Compute solution
df.solve(F == 0, u, bc, solver_parameters={"newton_solver":
                                           {"relative_tolerance": 1e-6}})

# Plot solution and solution gradient
df.plot(u, title="Solution")
df.plot(df.grad(u), title="Solution gradient")
df.interactive()

# Save solution in VTK format
file = df.File("nonlinear_poisson.pvd")
file << u