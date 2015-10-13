import os
import dolfin as df
import numpy as np
from finmag import Simulation as Sim
from finmag.energies import Exchange, DMI, Zeeman
from finmag import normal_mode_simulation as evnmsim
import matplotlib.pyplot as plt
from finmag.util.consts import mu0
from amr import amr

# Create directory where results will be saved.
results_dirname = '../results/three_dimensional/'
if not os.path.exists(results_dirname):
    os.makedirs(results_dirname)

Ms = 1.6e5  # saturation magnetisation (A/m)
A = 3.11e-13  # exchange energy constant (J/m)
D = 2.81e-4  # DMI energy constant (J/m**2)
K = 16e3  # uniaxial anisotropy constant (J/m**3)
thickness = 20e-9  # sample thickness (nm)
d = 20e-9  # lateral dimensions (nm)
n = 5  # number of computed eigenvalues

nxy = 15  # discretisation in lateral directions
nz = 5  # discretisation in out-of-plane direction

B = 0

mesh = df.BoxMesh(df.Point(0, 0, 0), df.Point(d, d, thickness), nxy, nxy, nz)
sim = evnmsim(mesh, Ms, m_init=(1, 0, 0),
              A=A, D=D, K1=K, K1_axis=(0, 0, 1),
              H_ext=(0, 0, 0), demag_solver=None)

# Update external magnetic field.
sim.set_H_ext((0, 0, B/mu0))

# Relax the system.
sim.relax()

# Sub domain for Dirichlet boundary condition
class DirichletBoundary(df.SubDomain):
    def inside(self, x, on_boundary):
        return (abs(x[0] - d) < df.DOLFIN_EPS or abs(x[0]) < df.DOLFIN_EPS) and on_boundary

g = df.Expression("x[0]")

amr(mesh, sim.llg.m_field.f, DirichletBoundary, g)
