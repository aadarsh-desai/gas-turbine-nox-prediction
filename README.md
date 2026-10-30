# Gas Turbine NOx Emission Prediction Using PyTorch

A machine learning project that predicts nitrogen oxide (NOx) emission concentrations from gas turbine operating measurements using a feedforward neural network implemented in PyTorch.

The project includes data preprocessing, neural network training, model evaluation, comparison with baseline models, and an interactive Streamlit application.

## 1. Project Overview

Nitrogen oxides (NOx) are air pollutants produced during combustion. Estimating NOx emissions from operational sensor measurements can help researchers study emission behavior.

This project uses a neural network to estimate same-hour NOx concentrations from nine gas turbine measurements.

**Objective:** Develop and evaluate a regression model for predicting NOx concentrations in mg/m³.

## 2. Dataset

**Source:** [UCI Machine Learning Repository — Gas Turbine CO and NOx Emission Data Set](https://archive.ics.uci.edu/dataset/551/gas+turbine+co+and+nox+emission+data+set)

The dataset contains 36,733 hourly observations collected from a gas turbine in Turkey between 2011 and 2015.

The original CSV files were downloaded from UCI and stored in `data/raw/`.

### Input Features

| Feature | Description |
|---|---|
| AT | Ambient Temperature |
| AP | Ambient Pressure |
| AH | Ambient Humidity |
| AFDP | Air Filter Difference Pressure |
| GTEP | Gas Turbine Exhaust Pressure |
| TIT | Turbine Inlet Temperature |
| TAT | Turbine After Temperature |
| TEY | Turbine Energy Yield |
| CDP | Compressor Discharge Pressure |

**Target variable:** NOX (nitrogen oxide concentration, mg/m³).

### Dataset Splitting

| Dataset | Years | Samples |
|---|---|---:|
| Training | 2011–2012 | 15,039 |
| Validation | 2013 | 7,152 |
| Testing | 2014–2015 | 14,542 |

Chronological splitting is used to evaluate model generalization across different years and reduce temporal leakage.

## 3. Technologies Used

- Python
- PyTorch
- NumPy
- Pandas
- Matplotlib
- Streamlit
- Git and GitHub

## 4. Neural Network Architecture

The model uses a fully connected feedforward neural network with the following architecture:

- Input layer: 9 features
- First hidden layer: 32 neurons with ReLU
- Second hidden layer: 16 neurons with ReLU
- Output layer: 1 neuron

Total trainable parameters: **865**.

### Training Configuration

| Parameter | Value |
|---|---|
| Optimizer | Adam |
| Loss function | Mean Squared Error |
| Batch size | 256 |
| Maximum epochs | 120 |
| Early stopping patience | 20 |
| Random seed | 42 |
| Weight decay | 0.0001 |
| Selected learning rate | 0.001 |
| Best epoch | 32 |
| Validation RMSE | 7.507 mg/m³ |

Input features and target values are standardized using training-set statistics only.

The model is trained from randomly initialized parameters. The best checkpoint is selected using validation performance.

## 5. Model Evaluation

The trained model was evaluated on 14,542 observations from 2014–2015.

### Test Results

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| Training Mean Baseline | 11.512 | 13.391 | -0.603 |
| Linear Regression | 7.431 | 10.102 | 0.088 |
| PyTorch Neural Network | 9.073 | 10.407 | 0.032 |

The neural network reduces RMSE by approximately 22.3% compared with the training-mean baseline.

However, linear regression achieves better performance on the test dataset. This indicates that the neural network does not provide an advantage over the simpler linear model in this experiment.

### Evaluation by Year

| Year | MAE | RMSE | R² |
|---|---:|---:|---:|
| 2014 | 7.620 | 9.137 | 0.160 |
| 2015 | 10.480 | 11.506 | -0.068 |

The decline in performance during 2015 indicates limited generalization across time.

### Training Loss

![Training and Validation Loss](artifacts/loss_curve.png)

### Model Comparison

![Baseline Model Comparison](artifacts/baseline_comparison.png)

### Actual vs Predicted NOx

![Actual vs Predicted NOx](artifacts/test_predictions.png)

## 6. Installation and Execution

### Clone Repository

```bash
git clone https://github.com/aadarsh-desai/gas-turbine-nox-prediction.git
cd gas-turbine-nox-prediction
```

### Create Python Virtual Environment (Windows PowerShell)

```powershell
py -3.13 -m venv .venv
```

### Install Dependencies

```powershell
.\.venv\Scripts\python.exe -m pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cpu

.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

### Train Model

```powershell
.\.venv\Scripts\python.exe train.py
```

This command trains the model and saves its checkpoint and training results in `artifacts/`.

### Evaluate Model

```powershell
.\.venv\Scripts\python.exe evaluate.py
```

This command generates the test metrics, predictions, and evaluation plots.

### Run Automated Tests

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

### Launch Streamlit Application

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Open `http://localhost:8501` in your browser.

## 7. Streamlit Application

The application provides three main features:

1. **Individual prediction:** Enter nine turbine measurements and estimate NOx concentration.
2. **Batch prediction:** Upload a CSV containing sensor measurements and download predictions.
3. **Model evaluation:** View the saved test metrics and visualizations.

The application loads the trained PyTorch checkpoint for inference without retraining the model.

## 8. Project Structure

```text
gas-turbine-nox-prediction/
├── .streamlit/
├── artifacts/
├── data/
│   ├── raw/
│   └── README.md
├── tests/
├── turbine/
│   ├── __init__.py
│   ├── config.py
│   ├── data.py
│   ├── inference.py
│   ├── metrics.py
│   └── model.py
├── app.py
├── train.py
├── evaluate.py
├── requirements.txt
└── README.md
```

## 9. Limitations

- The dataset represents a single gas turbine.
- The neural network generalizes poorly to the later test year.
- Linear regression performs better on the held-out dataset.
- The model estimates same-hour emissions rather than forecasting future emissions.
- Predictions do not include calibrated uncertainty intervals.
- Results should not be used directly for industrial control or regulatory compliance.

## 10. Future Improvements

Possible extensions include:

- Experimenting with alternative neural network architectures.
- Comparing regularization and feature selection methods.
- Evaluating stronger regression baselines.
- Investigating temporal distribution changes.
- Testing on additional turbines or new datasets.

Further model selection should use appropriate validation data, with a new untouched test dataset reserved for final assessment.

## 11. Reproducibility and Development

The project was retrained and evaluated locally on Windows using VS Code and PyTorch 2.9.1 (CPU).

The local training run completed in approximately 31.49 seconds. The reproduced validation and test metrics matched the recorded results.

AI assistance was used in project scaffolding, documentation, and code checks. The training and evaluation workflows were subsequently executed locally.

## 12. Dataset Attribution

Gas Turbine CO and NOx Emission Data Set. UCI Machine Learning Repository (2019).

DOI: https://doi.org/10.24432/C5WC95

Dataset license: Creative Commons Attribution 4.0.