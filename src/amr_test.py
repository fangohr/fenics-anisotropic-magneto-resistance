import dolfin as df
import numpy as np
from amr import amr, compute_flux

d = 1  # lateral dimensions (m)
n = 10  # discretisation
mesh = df.BoxMesh(df.Point(0, 0, 0), df.Point(d, d, d), n, n, n)
mesh2d = df.RectangleMesh(df.Point(0, 0), df.Point(d, d), n, n)

# Sub domain for Dirichlet boundary condition
class DirichletBoundary(df.SubDomain):
    def inside(self, x, on_boundary):
        return (abs(x[0] - d) < df.DOLFIN_EPS or abs(x[0]) < df.DOLFIN_EPS) and on_boundary
g = df.Expression("1 - x[0]/d", d=d)
VV = df.VectorFunctionSpace(mesh, 'CG', 1, 3)

tol = 1e-6
def test_compute_flux():
    f = df.interpolate(df.Constant((1, 0, 0)), VV)
    assert abs(compute_flux(f, mesh2d) - 1) < tol

    f = df.interpolate(df.Constant((1, 0, 1)), VV)
    assert abs(compute_flux(f, mesh2d) - 1) < tol

    f = df.interpolate(df.Constant((0.5, 0, 1)), VV)
    assert abs(compute_flux(f, mesh2d) - 0.5) < tol

    f = df.interpolate(df.Constant((0, 0, 1)), VV)
    assert abs(compute_flux(f, mesh2d) - 0) < tol

def test_m_001():
    theta = np.pi/2
    m = df.project(df.Constant((0, 0, 1)), VV)
    u, j, i = amr(mesh, m, DirichletBoundary, g, mesh2d, s0=1, alpha=1)
    # Check the potential in the middle of the sample.
    # The correct solution should be one half of applied potential (0.5).
    assert abs(u(d/2., d/2., d/2.) - 0.5) < tol
    # Check electric field.
    # The electric field magnitude can be computed as U/d where U
    # is the applied voltage and d is the sample length.
    # In this case the correct E = (1/d, 0, 0)
    E = df.project(-df.grad(u), VV)
    assert abs(E(0.5, 0.5, 0.5)[0] - 1./d) < tol
    assert abs(E(0.5, 0.5, 0.5)[1] - 0) < tol
    assert abs(E(0.5, 0.5, 0.5)[2] - 0) < tol
    # Check current density.
    # Current density is the conductivity * electric field.
    rho = 1 + np.cos(theta)**2
    assert abs(j(0.5, 0.5, 0.5)[0] - 1./(d*rho)) < tol
    assert abs(j(0.5, 0.5, 0.5)[1] - 0) < tol
    assert abs(j(0.5, 0.5, 0.5)[2] - 0) < tol
    # Check current (current density flux).
    assert abs(i - d/rho) < tol

def test_m_100():
    theta = 0
    m = df.project(df.Constant((1, 0, 0)), VV)
    u, j, i = amr(mesh, m, DirichletBoundary, g, mesh2d, s0=1, alpha=1)
    # Check the potential in the middle of the sample.
    assert abs(u(d/2., d/2., d/2.) - 0.5) < tol
    # Check electric field.
    E = df.project(-df.grad(u), VV)
    assert abs(E(0.5, 0.5, 0.5)[0] - 1./d) < tol
    assert abs(E(0.5, 0.5, 0.5)[1] - 0) < tol
    assert abs(E(0.5, 0.5, 0.5)[2] - 0) < tol
    # Check current density.
    rho = 1 + np.cos(theta)**2
    assert abs(j(0.5, 0.5, 0.5)[0] - 1./(d*rho)) < tol
    assert abs(j(0.5, 0.5, 0.5)[1] - 0) < tol
    assert abs(j(0.5, 0.5, 0.5)[2] - 0) < tol
    # Check current (current density flux).
    assert abs(i - d/rho) < tol

def test_m_101():
    theta = np.pi/4
    m = df.project(df.Constant((1, 0, 1)), VV)
    u, j, i = amr(mesh, m, DirichletBoundary, g, mesh2d, s0=1, alpha=1)
    # Check the potential in the middle of the sample.
    assert abs(u(d/2., d/2., d/2.) - 0.5) < tol
    # Check electric field.
    E = df.project(-df.grad(u), VV)
    assert abs(E(0.5, 0.5, 0.5)[0] - 1./d) < tol
    assert abs(E(0.5, 0.5, 0.5)[1] - 0) < tol
    assert abs(E(0.5, 0.5, 0.5)[2] - 0) < tol
    # Check current density.
    rho = 1 + np.cos(theta)**2
    assert abs(j(0.5, 0.5, 0.5)[0] - 1./(d*rho)) < tol
    assert abs(j(0.5, 0.5, 0.5)[1] - 0) < tol
    assert abs(j(0.5, 0.5, 0.5)[2] - 0) < tol
    # Check current (current density flux).
    assert abs(i - d/rho) < tol

def test_m_110():
    theta = np.pi/4
    m = df.project(df.Constant((1, 0, 1)), VV)
    u, j, i = amr(mesh, m, DirichletBoundary, g, mesh2d, s0=1, alpha=1)
    # Check the potential in the middle of the sample.
    assert abs(u(d/2., d/2., d/2.) - 0.5) < tol
    # Check electric field.
    E = df.project(-df.grad(u), VV)
    assert abs(E(0.5, 0.5, 0.5)[0] - 1./d) < tol
    assert abs(E(0.5, 0.5, 0.5)[1] - 0) < tol
    assert abs(E(0.5, 0.5, 0.5)[2] - 0) < tol
    # Check current density.
    rho = 1 + np.cos(theta)**2
    assert abs(j(0.5, 0.5, 0.5)[0] - 1./(d*rho)) < tol
    assert abs(j(0.5, 0.5, 0.5)[1] - 0) < tol
    assert abs(j(0.5, 0.5, 0.5)[2] - 0) < tol
    # Check current (current density flux).
    assert abs(i - d/rho) < tol
