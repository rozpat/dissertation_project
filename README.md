# dissertation_project
Algorithms for analysing human gait motion using data from motion capture cameras, a treadmill, and a smartphone's inertial measurement unit (IMU). Includes code for synchronising these instruments and machine learning models for classifying patients' stances (whether they had their eyes closed or open) and predicting the maximum Centre of Pressure (COP) during bending exercises.


# Accelerometer and Gyroscope Data Analysis for Postural Control Prediction and Eye State Classification

This project analysed accelerometer and gyroscope data for two purposes: 
- to classify the subject's state as standing with eyes open or closed 
- predict the centre of pressure (CoP) maxima 
Various data analysis and machine learning techniques were applied to create, evaluate and refine predictive models. 

## Objective 1: Eye condition classification
Using accelerometer and gyroscope data, different machine learning classifiers were tested, And the three best ones were selected: the Random Forest classifier, the Gradient Boosting classifier and the MLP classifier to distinguish between the cases of people standing with their eyes open or closed. Each classifier was tuned using hyperparameter tuning to ensure optimal performance in classifying two different states. A  feature selection was conducted utilising F-value ANOVA to find the most impactful features.

## Objective 2: Predicting CoP maxima
Linear regression was used to predict CoP maxima, an important indicator in understanding the stability and postural control of the subjects. Features were selected using Lasso regression to increase predictive accuracy.

## Preliminary Work: Data Preparation and Exploration
Prior to model development, several data preparation steps were undertaken, including:
- Data Synchronisation: Aligning data from three disparate instruments- treadmill, VICON, and smartphone to ensure consistency in subsequent analyses
- Event Detection: Employing automatic detection mechanisms to identify instances of bending movements within the data.
- Data Structuring: Creating CSV files, which were used for training the selected models.

Additionally, a suite of functions was devised to:
- Calculate population statistics to get insights into the studied population's characteristics
- Calculating the mean difference and standard deviation between peaks detected by the instruments used to align the data to assess the synchronisation performance


## Datasets

The datasets consist of the accelerometer and gyroscope data collected from 20 subjects. Each record contains 48 features, including minimum, maximum, mean, SD, skewness, kurtosis, dominant frequency, its corresponding amplitude, and a target variable, which was either eyes closed/opened ot the COP maxima value. The features were extracted from 3-axial accelerometer and gyroscope data.

The data is stored in CSV format.

## Exclusion Criteria
In the current analysis, data from subject S5 for Quiet Stance and Walk 1 tasks has been excluded due to the fact that the files were malformed.

## Prerequisites
Ensure you have the following Python libraries installed to run the code:

- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- SciPy

## Folders
- /subjects_data: contains each participant's data from 3 instruments
## Python Files
- loadData.py: functions to load and format datasets for analysis
- participants_details.py: The function calculates statistical data of the studied population
- synchronisation_v2.py: Functions synchronise all instruments based on VICON timestamps
- plots.py: provides visualisation functions for peaks detection, raw signals, etc.
- event_detection.py: Automates event detection (bending task) using the COP signal and generates data (e.g. timestamps) for further analysis
- trial.py: Main analysis file utilising above scripts; aligns data, detects events, visualises results, and offers a user interface for subject and task selection.
- alignment_sd.py: functions calculate the median time difference and time differences between accelerometer signal peaks and vicon acceleration signal peaks, and between treadmill signal peaks and vicon signal peaks (before and after data alignment). Then the mean and standard deviation is calculated to show how well the synchronisation functions perform. This file can be run to show the results.
- subjects_bending_data.py: functions create a csv file for training the CoP maxima prediction models. File can be run to create a csv file.
- subjects_static_data.py: functions create a csv file for training the classifiers that classify wheter the person stands with eyes opened or closed. File can be run to create a csv file.
### Jupyter Notebook Files
- best_features_model_selection.ipynb: code in this file is designed to find the best models for classification task and to find the most relevant features. It shows the performance of 5 models before and after feature selection.
- gradient_boosting_classifier.ipynb: code is designed to find the train the GradientBoostingClassifier before and after feature selection
- rfc_classifier.ipynb: code is designed to find the train the RandomForestClassifier before and after feature selection
- mlp_classifier.ipynb: code is designed to find the train the MLPClassifier before and after feature selection
- cop_prediction.ipynb: code in this file is designed to find the relevant features and train the Linear Regression model before and after feature selection

### CSV Files 
- classifier_training_data.csv: csv file creater for training the classifiers
- cop_training_dataset.csv: csv file created for training the CoP maxima prediction model

### X Files 
- patient_data.xlsx: file with information collected about patients (e.g. age, height, weight etc.)
