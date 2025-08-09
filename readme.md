# 🏈 Football Game Outcome Prediction – Logistic Regression (Rust)

## Status
**WIP – Proof of Concept**

This project is an experimental **logistic regression model** built in **Rust** using the [`linfa`](https://github.com/rust-ml/linfa) crate.  
It predicts the outcome of a football game (win/loss) based on team and opponent strength metrics.

The current implementation:
- Uses a small synthetic dataset
- Trains a logistic regression model
- Evaluates accuracy using a confusion matrix
- Predicts the outcome for new input values

## Planned Features
- [ ] Load real-world datasets from CSV or API
- [ ] Add more predictive features (weather, injuries, historical performance, etc.)
- [ ] Implement data normalization and preprocessing
- [ ] Support multi-class predictions (Win / Loss / Draw)
- [ ] Export trained models for reuse
- [ ] Integrate with a simple CLI or web API for predictions
- [ ] Optimize performance for high-volume predictions

## Tech Stack
- **Language:** Rust
- **ML Framework:** Linfa (`linfa`, `linfa-logistic`)
- **Data Handling:** ndarray
- **Build Tool:** Cargo

## Why Rust?
Rust provides:
- **Memory safety** without garbage collection
- **C/C++-level performance**
- Strong concurrency guarantees
- A growing ecosystem for numerical and ML workloads

## Disclaimer
This is a **proof of concept**.  
The dataset is minimal and purely for demonstrating the algorithm structure in Rust.

---