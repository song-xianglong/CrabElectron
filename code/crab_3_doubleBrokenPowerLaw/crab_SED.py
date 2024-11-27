import astropy.units as u
import numpy as np
from astropy.constants import c
from astropy.io import ascii
from astropy.table import Table

import naima
from naima.models import ExponentialCutoffPowerLaw, InverseCompton, Synchrotron, Bremsstrahlung, PionDecay, ExponentialCutoffDoubleBrokenPowerLaw

#distance of RX J1713
d = 2.0 * u.kpc

# Best-fit parameters
FitPar = Table.read('SynIC.h5_results.ecsv', format='ascii.ecsv')

amplitude = FitPar['median'][1] / u.eV
e_break1 = FitPar['median'][3] * u.TeV
e_break2 = FitPar['median'][5] * u.TeV
alpha_1 = FitPar['median'][6]
alpha_2 = FitPar['median'][7]
alpha_3 = FitPar['median'][8]
e_cutoff = FitPar['median'][10] * u.TeV
B = FitPar['median'][11] * u.uG
We = FitPar['median'][12] * u.erg
#Model
ECPL = ExponentialCutoffDoubleBrokenPowerLaw(amplitude, 1.0 * u.TeV, e_break1, e_break2, alpha_1, alpha_2, alpha_3, e_cutoff)

eopts = {"Eemax": 50 * u.PeV, "Eemin": 0.1 * u.GeV}

SYN = Synchrotron(ECPL, B=B, **eopts)

Rpwn = 2.1 * u.pc
Esy = np.logspace(-7, 9, 100) * u.eV
Lsy = SYN.flux(Esy, distance=0 * u.cm)  # use distance 0 to get luminosity
phn_sy = Lsy / (4 * np.pi * Rpwn ** 2 * c) * 2.24

IC = InverseCompton(
    ECPL,
    seed_photon_fields=[
            "CMB",
            ["FIR", 70 * u.K, 0.5 * u.eV / u.cm ** 3],
            ["OPT", 5000 * u.K, 1.0 * u.eV / u.cm ** 3],
            #["SSC", Esy, phn_sy],
        ],
    **eopts,
)

# Use plot_data from naima to plot the observed spectra
crab = ascii.read('crab_data_points.ecsv')

radio = crab[0:18]
micro = crab[18:62]
uv = crab[62:86]
soft_xray = crab[86:123]
hard_xray = crab[123:154]
gamma = crab[154:204]
fermi = crab[204:226]
vhe = crab[226:-1]


figure = naima.plot_data([radio, micro, uv, soft_xray, hard_xray, gamma, fermi, vhe], e_unit=u.eV)
ax = figure.axes[0]

# Plot the computed model emission
energy = np.logspace(-9, 16, 100) * u.eV

ax.loglog(
    energy,
    SYN.sed(energy, d),
    lw=2,
    c="green",
    label="SYN",
)

for i, seed, ls in zip(
    range(3), ["CMB", "FIR", "OPT"], ["--", "-.", ":"]
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

ax.set_xlim(1e-8, 1e16)
ax.set_ylim(1e-14, 1e-7)
ax.legend(loc="upper right", frameon=False)
figure.tight_layout()
figure.savefig("SynIC-BestFitPar.pdf")
