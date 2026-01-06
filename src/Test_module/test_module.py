MY_UNIQUE_ID = "TestUser"

from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import balanced_accuracy_score
import pickle
import pandas as pd
import numpy as np
import os

# Output: unique ID of the team
def getName():
    return MY_UNIQUE_ID


# Input: Dataset dictionary and classifier name
# Output: PreProcessed Dataset dictionary
def preprocess(dataset, clfName):
    
    #TODO: AGGIORNARE CON ULTIME MODIFICHE DI PREPROCESSING
    
    y = dataset['target']

    # 1. Caricamento Asset di Preprocessing
    with open("preprocessing_assets.pkl", "rb") as f:
        assets = pickle.load(f)
    
    # Recupero parametri
    cols_to_drop_nan = assets['cols_to_drop_nan']
    cols_to_drop_manual = assets['cols_to_drop_manual']
    emp_len_map = assets['emp_len_map']
    medians = assets['medians']
    modes = assets['modes']
    oh_encoders = assets['oh_encoders']
    l_encoders = assets['l_encoders']
    final_columns = assets['final_columns']

    # 2. Drop colonne 
    X = X.drop(columns=[c for c in cols_to_drop_nan if c in X.columns])
    X = X.drop(columns=[c for c in cols_to_drop_manual if c in X.columns])

    # 3. Employment Length Map
    if 'borrower_profile_employment_length' in X.columns:
        X['emp_length_num'] = X['borrower_profile_employment_length'].map(emp_len_map)
        X = X.drop(columns=['borrower_profile_employment_length'])

    # 4. Trasformazione Date
    date_cols = ['loan_issue_date', 'credit_history_earliest_line', 'last_payment_date', 'last_credit_pull_date']
    
    # Feature Engineering: Months Credit History
    if 'loan_issue_date' in X.columns and 'credit_history_earliest_line' in X.columns:
        d1 = pd.to_datetime(X['loan_issue_date'], errors='coerce')
        d2 = pd.to_datetime(X['credit_history_earliest_line'], errors='coerce')
        X['months_credit_history'] = (d1.dt.year - d2.dt.year) * 12 + (d1.dt.month - d2.dt.month)
        X['months_credit_history'] = X['months_credit_history'].fillna(0).clip(lower=0)

    # Estrazione Anno/Mese e drop date originali
    for col in date_cols:
        if col in X.columns:
            dt = pd.to_datetime(X[col], errors='coerce')
            X[col + '_year'] = dt.dt.year
            X[col + '_month'] = dt.dt.month
            X = X.drop(columns=[col])

    # 5. Imputazione 
    for col, val in medians.items():
        if col in X.columns:
            X[col] = X[col].fillna(val)
    for col, val in modes.items():
        if col in X.columns:
            X[col] = X[col].fillna(val)

    # 6. Encoding Categorico
    # Label Encoding 
    for col, le in l_encoders.items():
        if col in X.columns:
            # Gestione valori ignoti
            X[col] = X[col].map(lambda s: s if s in le.classes_ else le.classes_[0])
            X[col] = le.transform(X[col].astype(str))
    
    # One-Hot Encoding 
    for col, ohe in oh_encoders.items():
        if col in X.columns:
            encoded = ohe.transform(X[[col]])
            names = [f"{col}_{cat}" for cat in ohe.categories_[0][1:]]
            X_oh = pd.DataFrame(encoded, columns=names, index=X.index)
            X = pd.concat([X.drop(col, axis=1), X_oh], axis=1)

    # 7. Allineamento Colonne (Assicura l'ordine corretto per lo scaler/modello)
    X = X.reindex(columns=final_columns, fill_value=0)

    # 8. Scaling specifico per il modello
    scaler_file = f"{clfName}_scaler.save"
    try:
        scaler = pickle.load(open(scaler_file, 'rb'))
        X_final = scaler.transform(X)
    except:
        X_final = X.to_numpy() 

    return {
        'data': X_final,
        'target': y.map(assets['grade_map']).to_numpy() 
    }


# Input: Classifier name ("lr": Logistic Regression, "svc": Support Vector Classifier)
# Output: Classifier object
def load(clfName):
    clf = None
    
    if (clfName == "lr"):
        clf = pickle.load(open("lr.save", 'rb'))
        
    elif (clfName == "svc"):
        clf = pickle.load(open("svc.save", 'rb'))

    elif (clfName == "rf"):
        clf = pickle.load(open("rf.save", 'rb'))

    return clf


# Input: PreProcessed Dataset dictionary, Classifier Name, Classifier Object 
# Output: Performance dictionary
def predict(dataset, clf):
    X = dataset['data']
    y = dataset['target']
    
    ypred = clf.predict(X)

    acc = accuracy_score(y, ypred)
    bacc = balanced_accuracy_score(y, ypred)
    f1 = f1_score(y, ypred, average="weighted")
    
    perf = {"acc": acc, "bacc": bacc, "f1": f1}
    
    return perf
    
