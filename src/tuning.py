from src.helper_code import *
import numpy as np
import os
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn import svm
from src.team_code import get_features



def load_features(data_folder="data/training_data"):
    # Find the patient data files.
    patient_files = find_patient_files(data_folder)
    print(data_folder)
    num_patient_files = len(patient_files)
    if num_patient_files == 0:
        raise Exception('No data was provided.')
    features = list()
    for i in range(num_patient_files):
        # Load the current patient data and recordings.
        current_patient_data = load_patient_data(patient_files[i])
        current_recordings = load_recordings(data_folder, current_patient_data)
        # Extract features.
        current_features = get_features(
            current_patient_data, current_recordings)
        features.append(current_features)
    # features = np.vstack(features)
    features = np.asarray(features, dtype=np.float32)
    return features


def load_murmurs(data_folder="data/training_data"):
    # Find the patient data files.
    patient_files = find_patient_files(data_folder)
    num_patient_files = len(patient_files)
    if num_patient_files == 0:
        raise Exception('No data was provided.')
    murmur_classes = ['Present', 'Unknown', 'Absent']
    murmurs = list()
    for i in range(num_patient_files):
        # Load the current patient data and recordings.
        current_patient_data = load_patient_data(patient_files[i])
        # Extract labels and use one-hot encoding.
        murmur = get_murmur(current_patient_data)
        if murmur in murmur_classes:
            current_murmur = murmur_classes.index(murmur)
            murmurs.append(current_murmur)
    # murmurs = np.vstack(murmurs)
    murmurs = np.asarray(murmurs, dtype=np.float32)
    return murmurs

def load_outcomes(data_folder="data/training_data"):
    # Find the patient data files.
    patient_files = find_patient_files(data_folder)
    num_patient_files = len(patient_files)

    if num_patient_files == 0:
        raise Exception('No data was provided.')
    outcome_classes = ['Abnormal', 'Normal']
    outcomes = list()
    for i in range(num_patient_files):
        # Load the current patient data and recordings.
        current_patient_data = load_patient_data(patient_files[i])
        outcome = get_outcome(current_patient_data)
        if outcome in outcome_classes:
            current_outcome = outcome_classes.index(outcome)
            outcomes.append(current_outcome)
    # outcomes = np.vstack(outcomes)
    outcomes = np.asarray(outcomes, dtype=np.float32)
    return outcomes

def load_raw_model():
    outcome_classes = ['Abnormal', 'Normal']
    murmur_classes = ['Present', 'Unknown', 'Absent']
    train_features = load_features()
    train_murmurs = load_murmurs()
    train_outcomes = load_outcomes()
    imputer = SimpleImputer().fit(train_features)
    features = imputer.transform(train_features)
    svc_murmur = svm.SVC(probability=True, decision_function_shape='ovo', C=1)
    svc_outcome = svm.SVC(probability=True)
    murmur_classifier = svc_murmur.fit(features, train_murmurs)
    outcome_classifier = svc_outcome.fit(features, train_outcomes)

    test_features = load_features(data_folder="data/test_data")
    test_murmurs = load_murmurs(data_folder="data/test_data")
    test_outcomes = load_outcomes(data_folder="data/test_data")

    imputer = SimpleImputer().fit(test_features)
    test_features = imputer.transform(test_features)
    murmur_predict = murmur_classifier.predict(test_features)
    outcome_predict = outcome_classifier.predict(test_features)
    print(test_murmurs.shape, murmur_predict.shape)
    print(test_features.shape, test_murmurs.shape, test_outcomes.shape, features.shape)
    print(classification_report(test_murmurs, murmur_predict, target_names=murmur_classes))
    print("===================================================")
    print(classification_report(test_outcomes, outcome_predict, target_names=outcome_classes))


def tuning_model():
    param_grid = {'C': [0.1, 1, 10, 100, 1000],
              'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
              'kernel': ['rbf'],
              'max_iter': [10, 50, 100, 150, 200]
              }
    outcome_classes = ['Abnormal', 'Normal']
    murmur_classes = ['Present', 'Unknown', 'Absent']
    train_features = load_features()
    train_murmurs = load_murmurs()
    train_outcomes = load_outcomes()
    imputer = SimpleImputer().fit(train_features)
    features = imputer.transform(train_features)
    grid_murmur = GridSearchCV(svm.SVC(probability=True, decision_function_shape='ovo'), param_grid, refit = True, verbose = 3)
    grid_outcome = GridSearchCV(svm.SVC(), param_grid, refit = True, verbose = 3)

    # svc_murmur = svm.SVC(probability=True, decision_function_shape='ovo', C=1)
    # svc_outcome = svm.SVC(probability=True)
    murmur_classifier = grid_murmur.fit(features, train_murmurs)
    outcome_classifier = grid_outcome.fit(features, train_outcomes)

    test_features = load_features(data_folder="data/test_data")
    test_murmurs = load_murmurs(data_folder="data/test_data")
    test_outcomes = load_outcomes(data_folder="data/test_data")

    imputer = SimpleImputer().fit(test_features)
    test_features = imputer.transform(test_features)
    murmur_predict = murmur_classifier.predict(test_features)
    outcome_predict = outcome_classifier.predict(test_features)
    print(test_murmurs.shape, murmur_predict.shape)
    print(test_features.shape, test_murmurs.shape, test_outcomes.shape, features.shape)
    print(classification_report(test_murmurs, murmur_predict, target_names=murmur_classes))
    print(grid_murmur.best_params_)
    print(grid_murmur.best_estimator_)
    print("===================================================")
    print(classification_report(test_outcomes, outcome_predict, target_names=outcome_classes))
    print(grid_outcome.best_params_)
    print(grid_outcome.best_estimator_)


if __name__ == "__main__":
    print("===============RAW MODEL===========")
    load_raw_model()
    print("===============TUNING MODEL===========")
    tuning_model()