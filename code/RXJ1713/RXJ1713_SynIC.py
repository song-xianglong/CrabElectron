#!/usr/bin/env python
import astropy.units as u
import numpy as np
from astropy.io import ascii

import naima
from naima.models import ExponentialCutoffPowerLaw, InverseCompton, Synchrotron

#distance of RX J1713
d = 1.0 * u.kpc 

## Read data
# We only consider every fifth X-ray spectral point to speed-up calculations for this example

radio = ascii.read('RXJ1713_Radio.dat')
soft_xray = ascii.read("RXJ1713_Suzaku-XIS.dat")[::5]
fermi = ascii.read("RXJ1713_Fermi.dat")
vhe = ascii.read("RXJ1713_HESS_2007.dat")

## Model definition
def ElectronSynIC(pars, data):

    # Match parameters to ECPL properties, and give them the appropriate units
    amplitude = 10 ** pars[0] / u.eV
    alpha = pars[1]
    e_cutoff = (10 ** pars[2]) * u.TeV
    B = pars[3] * u.uG

    # Initialize instances of the particle distribution and radiative models
    ECPL = ExponentialCutoffPowerLaw(amplitude, 1.0 * u.TeV, alpha, e_cutoff)

    eopts = {"Eemax": 50 * u.PeV, "Eemin": 0.1 * u.GeV}

    # Compute IC on CMB and on a FIR component with values from GALPROP for the
    # position of RXJ1713
    IC = InverseCompton(
        ECPL,
        seed_photon_fields=[
            "CMB",
            ["FIR", 50 * u.K, 0.5 * u.eV / u.cm ** 3],
            ["OPT", 5000 * u.K, 1.0 * u.eV / u.cm ** 3],
        ],
        **eopts,
    )
    SYN = Synchrotron(ECPL, B = B, **eopts)

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
        + naima.uniform_prior(pars[1], 0, 5)
        + naima.uniform_prior(pars[2], np.log10(1), np.log10(1000))
        + naima.uniform_prior(pars[3], 0, 100)
    )
    return logprob

if __name__ == "__main__":

    ## Set initial parameters and labels
    # Estimate initial magnetic field and get value in uG
    B0 = 2 * naima.estimate_B(soft_xray, vhe).to("uG").value

    p0 = np.array((34.42, 2.22, 1.529, 14.35))
    labels = ["log10(norm)", "index", "log10(cutoff)", "B"]

    ## Run sampler
    sampler, pos = naima.run_sampler(
        data_table=[radio, soft_xray, fermi, vhe],
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

