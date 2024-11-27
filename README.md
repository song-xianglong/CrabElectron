# From LHAASO multi-wavelength data to electron distribution.

## Project Overview

This project focuses on analyzing and fitting multiwavelength data of the Crab Nebula using the **NAIMA** Python package. The Crab Nebula is a highly studied astrophysical source powered by the Crab pulsar, known for its remarkable ability to accelerate electrons and positrons to high energies. The goal of this project is to model its spectral energy distribution (SED) and explore the physical processes within the nebula, such as synchrotron radiation, inverse Compton scattering, and possible contributions from hadronic components.

---

## Directory Structure

- **`code/`**: Contains the Python scripts used for data analysis and fitting.
- **`presentation/`**: Includes slides summarizing the project's methodology, results, and conclusions.
- **`RXJ1713/`**: Example data and scripts for getting started with NAIMA.
- **`crab_1_brokenPowerLaw/`**: Data and partial scripts for fitting the Crab Nebula's SED with a single broken power law and exponential cutoff.
- **`crab_2_BrokenPowerLaw_SSC/`**: Placeholder for modeling synchrotron self-Compton (SSC) contributions.
- **`crab_3_doubleBrokenPowerLaw/`**: Placeholder for fitting SED with a double-broken power law.
- **`crab_4_doubleBrokenPowerLaw_protonComponent/`**: Placeholder for incorporating hadronic components into the SED model.

---

## Instructions

### I. Introduction to NAIMA

NAIMA is a Python package designed to analyze gamma-ray data from non-thermal particle emissions. It provides tools for:

1. Fitting data using a likelihood-based approach.
2. Visualizing the SED and its components.

An example dataset (RXJ1713) is included to demonstrate the usage of NAIMA.

---

### II. Example: RXJ1713

#### Description
RXJ1713 is a supernova remnant (SNR) capable of accelerating leptons (e-/e+). These particles emit photons across a wide energy range, from radio to gamma-rays. 

#### Usage
In the directory `RXJ1713`, you will find:
- `.dat` files containing gamma-ray data.
- Python scripts illustrating how to use NAIMA for data fitting and SED visualization.

---

### III. Project Tasks: Crab Nebula Analysis

The Crab Nebula emits photons via synchrotron radiation and inverse Compton scattering, with its electron spectrum featuring breaks at $\gamma \approx 10^6$ and $\gamma \approx 10^9$. The project aims to fit the Crab's multiwavelength data and derive its SED by assuming various electron distributions and photon field targets.

#### Task Breakdown

1. **Broken Power Law**
   - **Directory**: `crab_1_brokenPowerLaw/`
   - **Goal**: Fit the Crab's SED with a broken power law and exponential cutoff.

2. **Broken Power Law with SSC**
   - **Directory**: `crab_2_BrokenPowerLaw_SSC/`
   - **Goal**: Extend the broken power law model to include inverse Compton scattering from synchrotron photons.

3. **Double Broken Power Law**
   - **Directory**: `crab_3_doubleBrokenPowerLaw/`
   - **Goal**: Fit the SED with a power law with two breaks and an exponential cutoff.

4. **Double Broken Power Law with Hadronic Component**
   - **Directory**: `crab_4_doubleBrokenPowerLaw_protonComponent/`
   - **Goal**: Extend the model to include a hadronic (proton) component to explain high-energy data points.

---

## Getting Started

1. **Install Dependencies**
   Ensure Python is installed and the following packages are available:
   - `numpy`
   - `scipy`
   - `matplotlib`
   - `astropy`
   - `NAIMA`

2. **Explore the Example**
   Navigate to `RXJ1713/`, run the provided scripts, and familiarize yourself with NAIMA's workflow.

3. **Work on Crab Nebula**
   - Start with `crab_1_brokenPowerLaw/` and refine the provided scripts.
   - Progress through the other tasks incrementally.

4. **Visualize Results**
   Use NAIMA's plotting tools to visualize the SED and its components.

---

## Notes

- The `code/` folder contains reusable utilities for data processing and model fitting.
- Refer to the slides in the `presentation/` folder for a high-level overview of the project.
- For questions about the project or issues with the scripts, consult the NAIMA [documentation](https://naima.readthedocs.io/) or raise an issue in this repository.

---

Happy modeling! 🚀