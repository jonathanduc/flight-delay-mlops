# ML Canvas — Flight Delay MLOps

> Version: 0.1  
> Status: Draft  
> Last updated: 2026-09-07

## 1. Background

Airport operations teams need to anticipate flight departure delays because unexpected delays can disrupt airport operations, resource allocation, and subsequent flights.

Earlier identification of high-risk departures can help operational teams prepare for potential disruptions and make better-informed resource planning decisions.

### Prediction contract

For every scheduled departure, predict **two hours before scheduled departure (T-2h)** the probability that the flight will depart **at least 15 minutes late**, using only information available at prediction time.

- **Observation:** scheduled flight departure
- **Target:** departure delay >= 15 minutes
- **Prediction horizon:** T-2h
- **Output:** probability of departure delay
- **Primary user:** airport operations team

### Key assumption

The T-2h prediction horizon is a project assumption. In a real deployment, this horizon would need to be validated with airport operations stakeholders to ensure that it provides sufficient time for operational action.

## 2. Value Proposition

Provide airport operations teams with an early departure-delay risk score, two hours before scheduled departure, allowing them to identify high-risk flights and proactively prepare resources and operational capacity to limit additional delays and reduce downstream disruption.

The system provides decision support. It does not automatically make operational decisions or guarantee a reduction in delays.

## 3. Objectives

The project has three main objectives:

1. **Predict** — Generate a calibrated probability of a departure delay of at least 15 minutes for every scheduled flight at T-2h.
2. **Identify** — Identify high-risk departures early enough for airport operations teams to focus attention on likely delays.
3. **Support prioritization** — Provide delay-risk information that can be combined with operational context to support prioritization and disruption-management decisions.

## 4. Solution

Build a decision-support system for airport operations that scores scheduled departures two hours before departure and presents the probability of a delay of at least 15 minutes.

High-risk flights can be surfaced and combined with available operational context to help teams prioritize attention and resource planning.

### V1 outputs

For each scored flight, the system should expose at least:

- flight
- scheduled departure
- origin
- destination
- airline
- delay probability
- risk category

### Out of scope for V1

The first version will not attempt to:

- automatically assign runways;
- automatically allocate personnel or equipment;
- optimize complete airport traffic;
- predict economic impact;
- automatically determine operational priority;
- incorporate real-time weather;
- model complex aircraft delay propagation.

## 5. Feasibility

### Data feasibility

The core prediction problem is feasible using public historical flight data from the US Bureau of Transportation Statistics (BTS).

However, public flight data do not provide the full operational state of an airport. In particular, the project does not have access to internal information such as:

- staff availability;
- gate availability;
- equipment availability;
- runway assignments;
- air traffic control decisions;
- internal operational decisions.

The project should therefore be considered a **production-ready prototype built from public data**, rather than a system ready for direct deployment in a real airport environment.

### Technical feasibility

Departure-delay prediction can be formulated as a supervised binary classification problem on tabular data.

The project can be developed and trained locally without requiring GPU infrastructure.

The main technical challenges are expected to include:

- processing a large historical dataset efficiently;
- preventing temporal data leakage;
- building reproducible data and feature pipelines;
- maintaining consistency between training and inference;
- evaluating the model under realistic temporal conditions.

### Product feasibility

The operational usefulness of the predictions cannot be fully validated without airport operations expertise.

In particular, the **T-2h prediction horizon is a project assumption** that would need to be validated with airport operations stakeholders before real-world deployment.

The usefulness of the system depends not only on predictive performance, but also on whether predictions are available early enough to support meaningful operational action.


## 6. Data

### Data source

Historical training data will come from the **US Bureau of Transportation Statistics (BTS) Reporting Carrier On-Time Performance dataset**.

The dataset contains historical information about scheduled domestic flights, including scheduled and actual flight times, airlines, airports, delays, cancellations, and other flight characteristics.

### Target

The supervised learning target will be derived from the actual departure delay:

`target = DepDelay >= 15 minutes`

A flight will therefore be labeled as delayed when its actual departure delay is at least 15 minutes.

Actual departure delay is used to construct the historical label but **must never be used as a model input**.

### Candidate V1 features

Initial features may include information such as:

- scheduled departure time;
- airline;
- origin airport;
- destination airport;
- scheduled flight duration;
- distance;
- day of week;
- calendar-derived features.

Only information available at or before the T-2h prediction timestamp may be used.

### Point-in-time correctness

For a flight scheduled to depart at 14:30, the prediction timestamp is 12:30.

Any feature used to generate that prediction must therefore represent information that would have been available no later than 12:30.

Variables describing future flight outcomes, including actual departure time, departure delay, taxi-out time, and arrival delay, cannot be used as input features.

Historical or operational aggregate features must also be computed **point-in-time**. They cannot use observations occurring after the prediction timestamp.

This constraint applies during both training and inference and is essential to prevent data leakage.

### Future extensions

A later version may investigate whether recent airport operational state improves departure-delay prediction through features such as:

- number of recent arrivals and departures;
- recent arrival-delay rate;
- recent departure-delay rate;
- mean recent arrival delay;
- airport traffic intensity.

These features will only be introduced if they can be constructed with strict point-in-time correctness.

## 7. Metrics

Because the objective is to identify future departure delays early enough for operational action, **recall will be the primary classification metric**.

A false negative corresponds to a flight that is actually delayed but that the system classified as low risk. These errors are particularly important because the operations team would not be alerted about a departure that eventually becomes delayed.

However, recall cannot be optimized in isolation. Predicting every flight as high risk would achieve perfect recall while producing an unusable alerting system.

Therefore:

- **Recall** will be the primary metric.
- **Precision** will be used as an operational guardrail.
- **PR-AUC** will evaluate performance across classification thresholds, particularly for the delayed-flight class.
- **ROC-AUC** will be tracked as a complementary discrimination metric.
- **Brier score and/or log loss** will evaluate the quality of predicted probabilities.
- **Calibration** will be evaluated to determine whether predicted probabilities correspond to observed delay frequencies.

### Decision threshold

The model will output a probability rather than directly assigning a delayed/on-time class.

The classification threshold will not automatically be fixed at 0.5. It will be selected on the validation set according to the trade-off between detecting delayed flights and maintaining sufficient precision for alerts to remain operationally useful.

The final threshold is therefore considered a product and operational decision informed by model performance.


## 8. Evaluation

Offline evaluation will use a **strict temporal split** to reproduce the production scenario: models are trained on past flights and evaluated on later unseen periods.

A possible initial split is:

- **Training:** 2023–2024
- **Validation:** January–June 2025
- **Test:** July–December 2025

The exact periods may be adjusted after inspecting data availability and distributions.

The validation period will be used for:

- model comparison;
- feature decisions;
- hyperparameter decisions;
- classification-threshold selection.

The final test period will remain untouched until the final model and decision threshold have been selected.

### Evaluation dimensions

Performance will be evaluated globally and across relevant segments such as:

- airport;
- airline;
- month or season.

The evaluation will include recall, precision, PR-AUC, probability calibration, and error analysis, with particular attention to false negatives.

In a real production environment, predictions generated at T-2h would later be joined with actual departure outcomes to measure model performance over time.


## 9. Modeling

Model development will follow an iterative baseline-first approach.

### 1. Naive baseline

Start with a simple naive strategy using `DummyClassifier`.

This establishes a minimum performance reference and verifies that subsequent models provide value beyond trivial prediction strategies.

### 2. Logistic regression

Train logistic regression as the first statistical machine-learning baseline.

Logistic regression provides:

- a relatively simple and interpretable model;
- probabilistic predictions;
- fast training;
- a useful reference against which more complex models can be evaluated.

### 3. Gradient-boosted decision trees

Evaluate a gradient-boosted tree model to capture nonlinear relationships and interactions between flight characteristics.

The exact implementation will be selected after inspecting the dataset and project requirements.

Additional model complexity will only be retained if it provides meaningful improvement on the temporal validation set.

### Modeling principle

The objective is not to maximize model complexity.

Each modeling step should answer the question:

> Does this additional complexity produce enough measurable value to justify its cost?

## 10. Inference

The primary inference strategy will use **scheduled batch prediction**.

At regular intervals, the system will identify upcoming scheduled departures approaching the T-2h prediction horizon, generate delay probabilities, and store the resulting predictions for consumption by an operational interface or downstream system.

Batch inference is preferred as the primary strategy because:

- flights requiring predictions are known in advance;
- predictions are generated around a predefined T-2h horizon;
- ultra-low inference latency is not required by the use case;
- multiple upcoming departures can be scored efficiently together.

### Online API

An online prediction API will also expose individual model predictions.

The API is considered a secondary serving mechanism intended for:

- system integration;
- demonstration;
- testing individual prediction requests.

The existence of an API does not make real-time inference the primary production architecture.


## 11. Feedback

### Model feedback

Predictions generated at T-2h should be stored together with:

- prediction timestamp;
- model version;
- predicted delay probability;
- predicted risk category;
- eventual actual departure delay;
- eventual observed target.

Once the actual outcome becomes available, predictions can be evaluated against observed departure outcomes.

Production monitoring should track metrics such as:

- recall;
- precision;
- calibration;
- false-negative rate;
- prediction distributions.

Performance degradation should trigger investigation rather than automatic retraining.

Potential causes may include:

- data drift;
- seasonal changes;
- changes in airline or airport behavior;
- data-quality problems;
- model degradation.

New labeled observations may then be incorporated into future model iterations when appropriate.

### Product feedback

Feedback from airport operations stakeholders would be required to determine whether predictions are operationally useful.

In particular, future iterations should evaluate whether T-2h provides the appropriate balance between:

- prediction quality;
- information availability;
- time available for operational intervention.

Alternative horizons such as T-1h or T-30min could be evaluated if operational feedback indicates that they would provide more useful predictions.

The statistically best prediction horizon is not necessarily the most useful operational horizon.


## 12. Project

### Team

This is an individual portfolio project covering responsibilities across:

- Data Science;
- Machine Learning Engineering;
- MLOps.

In a real-world deployment, these responsibilities would likely be distributed across multiple roles and would require collaboration with airport operations domain experts.

### Deliverables

The project aims to deliver:

- a reproducible data pipeline;
- a feature engineering pipeline;
- a temporal train/validation/test strategy;
- baseline and final ML models;
- model evaluation and calibration;
- experiment and model tracking;
- scheduled batch inference;
- an online prediction API;
- automated tests;
- continuous integration;
- a containerized application;
- model and data monitoring;
- a deployed end-to-end application;
- technical documentation and architecture documentation.

### Timeline

The initial target is approximately **five weeks of development**, working around **2–3 hours per day**, for an estimated total of **50–60 hours**.

### Definition of success

`flight-delay-mlops` will be considered successful when:

1. the ML system is deployed and functional end-to-end;
2. the technical and product decisions are documented and justified;
3. the model can generate reproducible predictions through the intended inference workflows;
4. the project demonstrates relevant production ML and MLOps practices;
5. I can clearly present and defend the problem framing, architecture, modeling decisions, evaluation methodology, limitations, and trade-offs in a technical interview or project presentation.