# PREDICTIVE MAINTENANCE: RUL ESTIMATION USING HYBRID DEEP LEARNING (CNN-LSTM)

## Overview
This academic data science project focuses on **Predictive Maintenance (PdM)**, specifically estimating the **Remaining Useful Life (RUL)** of turbofan jet engines using the **NASA C-MAPS** dataset.

The project moves beyond traditional regression by implementing and comparing two distinct Deep Learning architectures: a baseline **Multi-Layer Perceptron (MLP)** and a robust **Hybrid CNN-LSTM** model. By leveraging Convolutional Neural Networks (CNN) for feature extraction and Long Short-Term Memory (LSTM) networks for temporal sequence learning, the project achieves significant improvements in prediction accuracy.

## Key Findings
* **Hybrid Model Superiority:** The Hybrid CNN-LSTM model significantly outperformed the baseline MLP.
    * **MLP RMSE:** ~25 cycles
    * **Hybrid CNN-LSTM RMSE:** ~15 cycles
* **Temporal Dynamics:** The LSTM layers effectively captured the time-dependent degradation of the engines, which the static MLP struggled to generalize.
* **Classification Accuracy:** Confusion matrices generated during evaluation demonstrate that the Hybrid model creates fewer false negatives when classifying engines into "Urgent Maintenance" vs. "Normal Operation" stages.
* **Feature Engineering:** The integration of **Fast Fourier Transform (FFT)** and **Wavelet Transform** features provided the neural networks with cleaner signals, further reducing noise.

## Project Contents
| File Name | Description |
| :--- | :--- |
| **NASA_C_MAPS_RUL.ipynb** | The complete **Jupyter Notebook** containing data preprocessing, signal processing (Wavelet/FFT), and the PyTorch implementation of both MLP and Hybrid CNN-LSTM models. |
| **train_FD001.txt** | Training dataset (Run-to-failure trajectories). |
| **test_FD001.txt** | Test dataset (Trajectories ending prior to failure). |
| **RUL_FD001.txt** | Ground truth RUL values for the test set. |
| **Project_Report.pdf** | Academic report detailing the architecture comparison, confusion matrix analysis, and RMSE results. |

## Setup and Execution

### Prerequisites
* Python 3.x
* **PyTorch** (Deep Learning Framework)
* Scikit-Learn, Pandas, NumPy
* SciPy & PyWavelets (Signal Processing)
* Matplotlib & Seaborn (Visualization)

### Dependencies
```bash
pip install torch pandas numpy scikit-learn scipy PyWavelets matplotlib seaborn

#For Project content first 4 files follow the below link and download the dataset from kaggle.
http://kaggle.com/datasets/bishals098/nasa-turbofan-engine-degradation-simulation
