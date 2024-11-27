#!/usr/bin/env python
import astropy.units as u
import numpy as np
from astropy.io import ascii
from astropy.constants import c

import naima
from naima.models import ExponentialCutoffPowerLaw, InverseCompton, Synchrotron, ExponentialCutoffBrokenPowerLaw

#distance of Crab
d = 2.0 * u.kpc


# Do not touch this part: Start-------------------------------------------------------------------------------------------

crab = ascii.read('crab_data_points.ecsv')
radio = crab[0:18]
micro = crab[18:62]
uv = crab[62:86]
soft_xray = crab[86:123]
hard_xray = crab[123:154]
gamma = crab[154:204]
fermi = crab[204:226]
vhe = crab[226:-1]

# Do not touch this part: End---------------------------------------------------------------------------------------------

## Model definition
def ElectronSynIC(pars, data):

    # Match parameters to ECPL properties, and give them the appropriate units
    amplitude = 10 ** pars[0] / u.eV
    e_break = (10 ** pars[1]) * u.TeV
    alpha_1 = pars[2]
    alpha_2 = pars[3]
    e_cutoff = (10 ** pars[4]) * u.TeV
    B = pars[5] * u.uG

    # Initialize instances of the particle distribution and radiative models
    ECBPL = ExponentialCutoffBrokenPowerLaw(amplitude, 1.0 * u.TeV, e_break, alpha_1, alpha_2, e_cutoff)

    eopts = {"Eemax": 50 * u.PeV, "Eemin": 0.1 * u.GeV}

    # Compute IC on CMB and on a FIR component with values from GALPROP for the
    # position of RXJ1713

    SYN = Synchrotron(ECBPL, B = B, **eopts)

    Rpwn = 2.1 * u.pc
    Esy = np.logspace(-7, 9, 100) * u.eV
    Lsy = SYN.flux(Esy, distance=0 * u.cm)  # use distance 0 to get luminosity
    phn_sy = Lsy / (4 * np.pi * Rpwn ** 2 * c) * 2.24

    IC = InverseCompton(
        ECBPL,
        seed_photon_fields=[
            "CMB",
            ["FIR", 70 * u.K, 0.5 * u.eV / u.cm ** 3],
            ["OPT", 5000 * u.K, 1.0 * u.eV / u.cm ** 3],
            #["SSC", Esy, phn_sy],
        ],
        **eopts,
    )

    # compute flux at the energies given in data['energy']
    model = IC.flux(data, distance = d) \
          + SYN.flux(data, distance = d)
    
    # Compute the total energy in e/p above 1 GeV for this realization
    We = IC.compute_We(Eemin = 1 * u.GeV)   

    # The first array returned will be compared to the observed spectrum for
    # fitting. All subsequent objects will be stored in the sampler metadata
    # blobs.
    return model, We


## Prior definition
def lnprior(pars):
    """
    Return probability of parameter values according to prior knowledge.
    Parameter limits should be done here through uniform prior ditributions
    """
    # Limit norm and B to be positive
    logprob = (
        naima.uniform_prior(pars[0], 0.0, np.inf)
        + naima.uniform_prior(pars[1], np.log10(0.1), np.log10(10))
        + naima.uniform_prior(pars[2], 0, 5)
        + naima.uniform_prior(pars[3], 0, 5)
        + naima.uniform_prior(pars[4], np.log10(100), np.log10(3000))
        + naima.uniform_prior(pars[5], 0, 100)
    )
    return logprob

if __name__ == "__main__":

    ## Set initial parameters and labels
    # Estimate initial magnetic field and get value in uG
    B0 = 2 * naima.estimate_B(soft_xray, vhe).to("uG").value

    p0 = np.array((36.67, np.log10(0.265), 1.5, 3.2, np.log10(1863.0), B0))
    labels = ["log10(norm)", "log10(e_break)", "alpha_1", "alpha_2", "log10(cutoff)", "B"]

    ## Run sampler
    sampler, pos = naima.run_sampler(
        data_table=[radio, micro, uv, soft_xray, hard_xray, gamma, fermi, vhe],
        p0=p0,
        labels=labels,
        model=ElectronSynIC,
        prior=lnprior,
        nwalkers=50,
        nburn=10,
        nrun=25,
        threads=8,
        prefit=True,
        interactive=True,
    )

    ## Save run results to HDF5 file (can be read later with naima.read_run)
    out_root = "SynIC.h5"
    naima.save_run(out_root, sampler, compression=True, clobber=True)

    ## Diagnostic plots
    naima.save_diagnostic_plots(
        out_root,
        sampler,
        sed=True,
        last_step=True,
        pdf=True,
        blob_labels=["Spectrum", "$W_e$($E_e>1$ GeV)"]
    )
    naima.save_results_table(out_root, sampler, convert_log=True, last_step=False, include_blobs=True)#, overwrite=True)

