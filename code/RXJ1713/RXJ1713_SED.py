import astropy.units as u
import numpy as np
from astropy.constants import c
from astropy.io import ascii
from astropy.table import Table

import naima
from naima.models import ExponentialCutoffPowerLaw, InverseCompton, Synchrotron, Bremsstrahlung, PionDecay

#distance of RX J1713
d = 1.0 * u.kpc

# Best-fit parameters
FitPar = Table.read('SynIC.h5_results.ecsv', format='ascii.ecsv')

amplitude = FitPar['median'][1] / u.eV
alpha = FitPar['median'][2]
e_cutoff = FitPar['median'][4] * u.TeV
B = FitPar['median'][5] * u.uG
We = FitPar['median'][6] * u.erg

#Model
ECPL = ExponentialCutoffPowerLaw(amplitude, 1.0 * u.TeV, alpha, e_cutoff)

eopts = {"Eemax": 50 * u.PeV, "Eemin": 0.1 * u.GeV}

SYN = Synchrotron(ECPL, B=B, **eopts)

IC = InverseCompton(
    ECPL,
    seed_photon_fields=[
        "CMB",
        #["FIR", 50 * u.K, 0.5 * u.eV / u.cm ** 3],
        #["OPT", 5000 * u.K, 1 * u.eV / u.cm ** 3],
        ["FIR", 50 * u.K, 0.5 * u.eV / u.cm ** 3],
        ["OPT", 5000 * u.K, 1. * u.eV / u.cm ** 3],
    ],**eopts,
)



# Use plot_data from naima to plot the observed spectra
radio = ascii.read('RXJ1713_Radio.dat')
soft_xray = ascii.read("RXJ1713_Suzaku-XIS.dat")[::5]
fermi = ascii.read("RXJ1713_Fermi.dat")
vhe = ascii.read("RXJ1713_HESS_2007.dat")

figure = naima.plot_data([radio, soft_xray, fermi, vhe], e_unit=u.eV)
ax = figure.axes[0]

# Plot the computed model emission
energy = np.logspace(-9, 15, 100) * u.eV

ax.loglog(
    energy,
    SYN.sed(energy, d),
    lw=2,
    c="green",
    label="SYN",
)

for i, seed, ls in zip(
        range(3), ["CMB", "FIR", "OPT"], ["--", "-.", ":"]#range(3), ["CMB", "FIR", "OPT"], ["--", "-.", ":"]
):
    ax.loglog(
        energy,
        IC.sed(energy, d, seed=seed),
        lw=1.5,
        c=naima.plot.color_cycle[i + 1],
        label=seed,
        ls=ls,
    )

ax.loglog(    
    energy,   
    IC.sed(energy, d) + SYN.sed(energy, d), 
    lw=2,     
    c="black",    
    label="Total",
)

ax.set_xlim(1e-9, 1e15)
ax.set_ylim(1e-14, 1e-8)
ax.legend(loc="upper right", frameon=False)
figure.tight_layout()
figure.savefig("SynIC-BestFitPar.pdf")
