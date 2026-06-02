# Monte Carlo Mortgage Valuation

This project implements fixed-income valuation models for a 10-year mortgage using calibrated interest-rate trees and Monte Carlo simulation.

## Project overview

The project values a fixed-rate mortgage with an embedded prepayment option using:

- Ho-Lee interest rate model
- Black-Derman-Toy interest rate model
- Binomial short-rate trees
- Monte Carlo simulation
- Mortgage-backed securities valuation
- Behavioural prepayment modelling

## Files

- `fm405_part_a.py` — constructs and calibrates Ho-Lee and BDT interest rate trees
- `fm405_parts_b_to_e.py` — mortgage valuation, MBS valuation, and Monte Carlo simulation
- `fm405_part_f.py` — behavioural prepayment model and sensitivity analysis
- `report.pdf` — full project write-up

## Tools used

- Python
- NumPy
- SciPy
- Matplotlib

## How to run

Install the required packages:

```bash
pip install -r requirements.txt

# fixed-income-mortgage-valuation-python
Python implementation of fixed-income mortgage valuation using Ho-Lee and BDT interest-rate models, binomial trees, Monte Carlo simulation, and prepayment modelling.
