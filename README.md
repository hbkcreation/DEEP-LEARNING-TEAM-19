# Anomaly Detection in Industrial IoT Sensor Streams Using Autoencoders

### Deep Learning – Complex Engineering Problem | Task 3

<p align="center">
  <strong>Team 19 | Variant 3E</strong><br>
  Metro Train Air Compressor – Predictive Maintenance
</p>

---

## 1. Project Overview

Industrial equipment monitoring is important for preventing unexpected failures, reducing maintenance costs, and improving operational reliability.

This project focuses on **anomaly detection in industrial IoT sensor streams** using deep learning techniques. The application domain is a metro train air compressor, where unusual operating conditions can lead to equipment failure or service interruptions.

The project uses the **UCI MetroPT-3 dataset**, which contains time-series sensor measurements collected from a metro train air production unit.

Since labelled fault data is limited and equipment failures are relatively rare, the proposed approach uses **unsupervised learning**. The models learn normal operating behaviour and identify deviations through reconstruction errors.

Two autoencoder architectures are investigated:

- Dense Autoencoder
- LSTM Autoencoder

The project compares their anomaly detection capabilities using operational evaluation metrics and cost-sensitive threshold selection.

## 2. Problem Statement

Metro train air compressors operate using multiple sensors that continuously monitor equipment conditions.

Unexpected changes in pressure, temperature, motor current, and compressor operating states may indicate abnormal behaviour.

Traditional monitoring systems may struggle to identify complex patterns across multiple sensor readings.

The objective is to develop an anomaly detection framework that:

- Learns normal compressor operating behaviour.
- Identifies abnormal sensor patterns.
- Detects potential fault events.
- Minimises unnecessary false alarms.
- Reduces fault detection delay.
- Supports cost-effective predictive maintenance.

## 3. Project Objectives

1. Analyse industrial IoT time-series sensor data.
2. Apply unsupervised deep learning for anomaly detection.
3. Implement Dense and LSTM Autoencoder architectures.
4. Generate sliding windows from sequential sensor readings.
5. Calculate reconstruction errors as anomaly scores.
6. Determine suitable anomaly detection thresholds.
7. Compare models using fault-event recall, false alarms, and detection delay.
8. Evaluate the financial impact of missed faults and false alarms.

## 4. Dataset Description

**Dataset:** MetroPT-3  
**Source:** UCI Machine Learning Repository  
**Application:** Metro train air compressor monitoring

The dataset contains timestamped sensor measurements representing the operating conditions of an air production unit.

### Sensor information

The available telemetry includes:

- Pressure measurements
- Temperature measurements
- Motor current
- Compressor states
- Valve states
- Other process measurements

### Dataset usage

| Data component | Purpose |
|---|---|
| Normal operating data | Autoencoder training |
| Normal validation data | Model validation |
| Known fault intervals | Threshold tuning and evaluation |

**Important:** Fault labels are not used to train the autoencoders. They are reserved for threshold tuning and model evaluation.

## 5. Proposed Methodology

The proposed framework follows a sequence of data preprocessing, model training, anomaly scoring, and evaluation stages.

### Overall workflow

```text
        UCI MetroPT-3 Dataset
                  |
                  v
       Data Loading and Cleaning
                  |
                  v
        Timestamp Processing
                  |
                  v
          Feature Scaling
                  |
                  v
       Sliding Window Generation
          Sequence Length: 30
             Stride: 3
                  |
          +-------+-------+
          |               |
          v               v
    Dense Autoencoder  LSTM Autoencoder
          |               |
          v               v
    Reconstruction    Reconstruction
       Error             Error
          |               |
          +-------+-------+
                  |
                  v
        Anomaly Score Calculation
                  |
                  v
          Threshold Selection
                  |
                  v
         Anomaly Detection
                  |
                  v
        Performance Evaluation
```

### Data preprocessing

The preprocessing pipeline includes:

1. Loading the MetroPT-3 dataset.
2. Converting timestamps into the appropriate format.
3. Sorting observations chronologically.
4. Identifying available sensor features.
5. Handling missing values.
6. Scaling sensor measurements using training data statistics.
7. Generating overlapping time-series windows.

The project uses:

- Sequence length: 30 readings
- Sliding-window stride: 3
- Normal training period: February 2020
- Evaluation: Later known fault intervals

The scaler is fitted using training data only to prevent data leakage.

## 6. Deep Learning Models

### 6.1 Dense Autoencoder

A Dense Autoencoder is used as the baseline anomaly detection model.

It consists of two major components:

**Encoder:** Compresses the input sensor measurements into a lower-dimensional latent representation.

**Decoder:** Reconstructs the original input from the latent representation.

The reconstruction error measures the difference between the original sensor input and the reconstructed output.

A high reconstruction error may indicate abnormal operating behaviour.

### 6.2 LSTM Autoencoder

The LSTM Autoencoder is designed to learn temporal relationships between consecutive sensor readings.

It consists of:

- Sequential input layer
- LSTM encoder
- Latent temporal representation
- LSTM decoder
- Reconstructed sequence

Unlike a basic dense architecture, the LSTM model can represent temporal dependencies in sensor streams.

### 6.3 Model Comparison

| Feature | Dense Autoencoder | LSTM Autoencoder |
|---|---|---|
| Architecture | Fully connected layers | LSTM-based sequence model |
| Input | Sensor feature representation | Sequential sensor windows |
| Temporal learning | Limited | Designed for temporal dependencies |
| Main purpose | Baseline comparison | Time-series anomaly detection |
| Anomaly score | Reconstruction error | Sequence reconstruction error |

## 7. Anomaly Detection

The autoencoders learn to reconstruct normal sensor patterns.

During evaluation, reconstruction error is calculated between the original input and reconstructed output.

Mean Squared Error (MSE) can be expressed as:

\[
MSE = \frac{1}{n}\sum_{i=1}^{n}(x_i-\hat{x}_i)^2
\]

Where:

- \(x_i\) = Original sensor value
- \(\hat{x}_i\) = Reconstructed sensor value
- \(n\) = Number of input values

The reconstruction error is used as the anomaly score.

A threshold is selected to distinguish normal and abnormal behaviour.

\[
Anomaly =
\begin{cases}
1, & Error > Threshold\\
0, & Error \leq Threshold
\end{cases}
\]

## 8. Threshold Selection

Two threshold selection approaches are considered.

### Best-F1 Threshold

The threshold is selected to maximise the F1-score, balancing precision and recall.

\[
F1 = 2 \times \frac{Precision \times Recall}
{Precision + Recall}
\]

### Cost-Sensitive Threshold

The cost-sensitive approach considers the financial consequences of missed faults and false alarms.

| Event | Project-defined cost |
|---|---:|
| Missed fault | ₹4,00,000 |
| False alarm episode | ₹2,000 |

A missed fault has a much higher financial impact than a false alarm. Therefore, threshold selection should consider both detection performance and operational cost.

## 9. Evaluation Metrics and Performance Targets

The project evaluates the anomaly detection system using operationally meaningful metrics.

| Metric | Required target |
|---|---:|
| Fault-event recall | ≥ 0.85 |
| False alarms | < 2 per 24 hours |
| Detection delay | < 5 minutes |
| Alternative delay condition | Within 3 readings if sampling is less frequent than once per minute |

### Evaluation metrics

- **Precision:** Proportion of detected anomalies that correspond to actual anomalies.
- **Recall:** Proportion of actual anomalies successfully detected.
- **F1-score:** Harmonic mean of precision and recall.
- **Fault-event recall:** Proportion of actual fault events detected.
- **False alarm rate:** Frequency of false alarm episodes during normal operation.
- **Detection delay:** Time between fault occurrence and its detection.

These values are project requirements. Actual experimental measurements should be reported separately.

## 10. Experimental Analysis

The experimental analysis compares the Dense Autoencoder and LSTM Autoencoder.

The following results should be recorded from model evaluation:

- Training and validation reconstruction loss.
- Precision and recall.
- F1-score.
- Fault-event recall.
- False alarm episodes per 24 hours.
- Detection delay.
- Best-F1 threshold.
- Cost-sensitive threshold.
- Total estimated operational cost.

### Required visualisations

1. Sensor time-series plots.
2. Training and validation loss curves.
3. Reconstruction error distribution.
4. Precision-Recall curve.
5. Anomaly score over time.
6. Comparison of Dense and LSTM Autoencoders.

The Precision-Recall curve should mark both the best-F1 and cost-sensitive thresholds.

## 11. Technology Stack

| Component | Technology |
|---|---|
| Programming language | Python |
| Development environment | Jupyter Notebook / Google Colab |
| Deep learning | TensorFlow / Keras |
| Data processing | Pandas, NumPy |
| Visualisation | Matplotlib |
| Evaluation | Scikit-learn |
| Dataset | UCI MetroPT-3 |

## 12. Project Structure

```text
DL_T19_3E/
│
├── README.md
├── requirements.txt
├── config.json
├── download_data.py
├── run_project.py
│
├── src/
│   ├── data_utils.py
│   ├── models.py
│   ├── metrics.py
│   └── reporting.py
│
├── notebooks/
│   └── anomaly_detection.ipynb
│
├── outputs/
│   ├── figures/
│   ├── metrics/
│   └── models/
│
└── report/
    └── REPORT_TEMPLATE.md
```

## 13. Installation and Execution

### Step 1: Clone the repository

```bash
git clone https://github.com/hbkcreation/DL_T19_3E.git
cd DL_T19_3E
```

### Step 2: Create a virtual environment

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

### Step 3: Install dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Prepare the dataset

Obtain the MetroPT-3 dataset from the UCI Machine Learning Repository and place it in the dataset directory specified by the project configuration.

### Step 5: Run the project

```bash
python run_project.py
```

For notebook-based execution, open the project notebook in Jupyter Notebook or Google Colab and execute the cells in sequence.

## 14. Expected Project Deliverables

- Jupyter Notebook / Google Colab implementation.
- Dense Autoencoder implementation.
- LSTM Autoencoder implementation.
- Model comparison.
- Precision-Recall curve with threshold markers.
- Cost-sensitive analysis table.
- One-page plain-language note for plant engineers.
- Technical project report.
- Team contribution details.
- AI tool usage disclosure.

## 15. Industrial Applications

The proposed anomaly detection approach can support:

- Railway predictive maintenance.
- Industrial compressor monitoring.
- Equipment condition monitoring.
- Early identification of unusual operating patterns.
- Maintenance planning and operational risk reduction.

The same general methodology can also be adapted to other industrial IoT systems with suitable sensor data and normal-operation training periods.

## 16. Limitations and Considerations

- Anomaly detection performance depends on the quality and representativeness of normal training data.
- Sensor noise and missing readings may affect reconstruction quality.
- Threshold selection influences the balance between fault detection and false alarms.
- Temporal resolution affects the achievable detection delay.
- The model's results should be validated against actual fault events before operational deployment.

## 17. Conclusion

This project investigates unsupervised deep learning for anomaly detection in industrial IoT sensor streams using the MetroPT-3 dataset.

By comparing Dense and LSTM Autoencoders, the project explores how reconstruction-based learning can identify deviations from normal compressor operating behaviour.

The evaluation framework considers fault-event recall, false alarms, detection delay, and cost-sensitive threshold selection.

The overall objective is to develop a practical predictive maintenance approach that can support reliable monitoring of metro train air compressors and reduce the operational impact of unexpected equipment failures.

## 18. Team Members

| Roll Number | Role |
|---|---|
| 23261A6648 | Data Lead |
| 23261A6626 | Model Lead |
| 23261A6602 | Evaluation & Report Lead |

**Team Number:** 19  
**Variant:** 3E  
**Institution:** Mahatma Gandhi Institute of Technology  
**Department:** CSE – Artificial Intelligence and Machine Learning

## 19. References

1. UCI Machine Learning Repository, *MetroPT-3 Dataset*. Include the verified dataset citation and URL in the final documentation.
2. H. Bourlard and Y. Kamp, “Auto-Association by Multilayer Perceptrons and Singular Value Decomposition,” *Biological Cybernetics*, 1988.
3. S. Hochreiter and J. Schmidhuber, “Long Short-Term Memory,” *Neural Computation*, 1997.
4. Include additional verified research publications on time-series anomaly detection and industrial predictive maintenance.

---

<p align="center">
  **Team 19 | Variant 3E**<br>
  Deep Learning – MGIT
</p>
