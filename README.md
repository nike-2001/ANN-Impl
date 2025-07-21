# Artificial Neural Network Project - Bank Customer Churn Classification

## Introduction

This project focuses on developing an end-to-end deep learning solution using Artificial Neural Networks (ANNs) to address a binary classification problem. We utilize TensorFlow and Keras to build, train, and deploy the model. The project aims to predict whether a bank customer will leave the bank based on various features.

## Problem Statement

The dataset, `churn_modelling.csv`, contains bank customer data with the target variable `exited` indicating whether a customer has left the bank. Features include:

* Credit Score
* Geography
* Gender
* Age
* Tenure
* Balance
* Number of Products
* Has Credit Card
* Is Active Member
* Estimated Salary

This is a binary classification task to determine customer churn.

## Project Overview

* **Libraries Used:** TensorFlow (for deep learning) and Keras (as a high-level API).
* **Objective:** Build, train, and deploy an ANN to predict customer churn.
* **Deployment:** The trained model will be saved and integrated into a Streamlit web application, deployed to the Streamlit cloud.

## Project Steps

* **Classification Focus:** Addressing the churn prediction problem using the provided dataset.
* **Feature Engineering:** Converting categorical variables to numerical values and applying standardization.
* **Neural Network Architecture:**
    * Input layer with 11 nodes (matching the number of features).
    * One or more hidden layers.
    * Output layer with 1 node for binary classification.
    * Dropout layers to prevent overfitting.
* **Model Training:** Training the model using Keras and TensorFlow with appropriate loss functions and optimizers.
* **Model Saving and Deployment:** Saving the model in `.h5` or `pickle` format and deploying it via Streamlit.

## Project Structure

| File/Directory             | Description                                          |
| :------------------------- | :--------------------------------------------------- |
| `app.py`                   | Main Python script for the Streamlit application.    |
| `Churn_Modelling.xlsx`     | Excel file containing the dataset.                   |
| `experiments/`             | Directory for Jupyter notebooks with experimental code. |
| `hyperparameter_tuning.ipynb` | Notebook for hyperparameter tuning.                  |
| `label_encoder_gender.pkl` | Pickle file for gender label encoding.               |
| `model.h5`                 | Trained Keras model file.                            |
| `onehot_encoder_geo.pkl`   | Pickle file for geographical one-hot encoding.       |
| `prediction.ipynb`         | Notebook for prediction tasks.                       |
| `requirements.txt`         | List of project dependencies.                        |
| `scaler.pkl`               | Pickle file for standardization scaler.              |
| `git/`                     | Directory for version control (Typically, `.git` is a hidden folder at the project root). |

## Getting Started

1.  **Install dependencies:** Ensure you have Python installed, then install the required libraries:
    ```bash
    pip install -r requirements.txt
    ```
2.  **Explore experimental notebooks:** Navigate to the `experiments/` directory to understand the data exploration, feature engineering, and model training processes.
    ```bash
    jupyter notebook experiments/
    ```
3.  **Train the model:** Run the relevant notebooks (e.g., `hyperparameter_tuning.ipynb` and parts of `prediction.ipynb` or a dedicated training notebook if one existed) to generate `model.h5` and the `.pkl` files.
4.  **Launch the Streamlit application:**
    ```bash
    streamlit run app.py
    ```

## Future Work

* Enhance model performance with advanced feature engineering.
* Explore additional neural network architectures.
* Optimize deployment for scalability.

## Resources

* [TensorFlow Documentation](https://www.tensorflow.org/api_docs/python/)
* [Keras Documentation](https://keras.io/api/)
* [Streamlit Deployment Guide](https://docs.streamlit.io/streamlit-cloud/deploy-your-app)
