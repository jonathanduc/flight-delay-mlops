# Flight Delay MLOps

A production-oriented machine learning project for predicting flight departure delays and exploring the full ML lifecycle from problem framing to deployment and monitoring.

## Problem

Airport operations teams need to anticipate departure delays early enough to prepare for potential operational disruptions.

The system aims to predict, **two hours before scheduled departure (T-2h)**, the probability that a flight will depart **at least 15 minutes late**.

The project is designed as a decision-support prototype for airport operations rather than an automated airport optimization system.

## Data

The project uses historical flight data from the **US Bureau of Transportation Statistics (BTS) Reporting Carrier On-Time Performance dataset**.

Only information available at the prediction timestamp will be used as model input in order to preserve point-in-time correctness and prevent data leakage.

## Project status

🚧 **Work in progress**

Current phase: **Python project setup and data foundations**.

The project is being developed incrementally, following production ML principles and the Made With ML learning path.

## Development setup

This project uses Python 3.10.11 and `uv` for Python environment and dependency management.

After cloning the repository, synchronize the project environment:

```bash
uv sync
source .venv/bin/activate
```

The project dependencies are declared in `pyproject.toml` and locked in `uv.lock` for reproducible environments.

## Documentation

- [ML Canvas](docs/ml-canvas.md)

## Planned ML lifecycle

The project will progressively cover:

- data ingestion and validation;
- exploratory data analysis;
- feature engineering;
- temporal model evaluation;
- baseline and gradient-boosted models;
- experiment tracking;
- batch inference;
- prediction API;
- automated testing and CI;
- containerization;
- deployment;
- model and data monitoring.

## Disclaimer

The T-2h prediction horizon and the operational use case are project assumptions based on publicly available data.

A real airport deployment would require validation with airport operations experts and access to additional operational data.