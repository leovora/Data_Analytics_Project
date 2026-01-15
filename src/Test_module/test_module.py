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
from utils import load_ff_model, internal_preprocess_logic, FeedForward, FeedForwardWrapper

matricola_leonardo = "0001186597"
matricola_carlotta = "0001181860"
MY_UNIQUE_ID = f"{matricola_leonardo}_{matricola_carlotta}"

# Output: unique ID of the team
def getName():
    return MY_UNIQUE_ID


# Input: Dataset dictionary and classifier name
# Output: PreProcessed Dataset dictionary
def preprocess(dataset, clfName):
    
    # Logica per determinare l'encoding
    use_label_only = True if clfName in ['tb', 'tf'] else False
    
    X = dataset['data']
    y = dataset['target'] 
    
    # Caricamento asset
    asset_file = "src/Test_module/preprocessing_assets_only_label.pkl" if use_label_only else "src/Test_module/preprocessing_assets.pkl"
    
    # Preprocess di X
    X_processed = internal_preprocess_logic(X, assets_path=asset_file, use_label_only=use_label_only)
    
    # Map di y
    with open(asset_file, "rb") as f:
        assets = pickle.load(f)
    
    return {
        'data': X_processed, 
        'target': y.map(assets['grade_map']).to_numpy()
    }


# Input: Classifier name ("lr": Logistic Regression, "svc": Support Vector Classifier)
# Output: Classifier object
def load(clfName):
    clf = None
    
    if (clfName == "knn"):
        clf = pickle.load(open("src/Test_module/models/knn_pipeline.save", 'rb'))
        
    elif (clfName == "svm"):
        clf = pickle.load(open("src/Test_module/models/svm_pipeline.save", 'rb'))

    elif (clfName == "rf"):
        clf = pickle.load(open("src/Test_module/models/rf_pipeline.save", 'rb'))

    elif (clfName == "ff"):
        clf = load_ff_model("src/Test_module/models/ff.save")
    
    elif (clfName == "tb"):
        #TODO
        clf = None
    
    elif (clfName == "tf"):
        #TODO
        clf = None

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
    
# TEST NOSTRO
if __name__ == "__main__":
    
    df = pd.read_csv("data/test_raw.csv")
    
    dataset = {
        'data': df.drop(columns=['grade']), 
        'target': df['grade']
    }

    name = getName()

    # # ---- TEST KNN ----
    # print("----------------- KNN -----------------")
    # dataset_processed = preprocess(dataset, 'knn')
    # clf = load('knn')
    # perf = predict(dataset_processed, clf)

    # print(f"Team ID: {name}")
    # print("knn")
    # print(f"Performance: {perf}")

    # # ---- TEST RF ----
    # print("----------------- RF -----------------")
    # dataset_processed = preprocess(dataset, 'rf')
    # clf = load('rf')
    # perf = predict(dataset_processed, clf)

    # print(f"Team ID: {name}")
    # print("rf")
    # print(f"Performance: {perf}")

    # # ---- TEST SVM ----
    # print("----------------- SVM -----------------")
    # dataset_processed = preprocess(dataset, 'svm')
    # clf = load('svm')
    # perf = predict(dataset_processed, clf)

    # print(f"Team ID: {name}")
    # print("svm")
    # print(f"Performance: {perf}")

     # ---- TEST FF ----
    print("----------------- FF -----------------")
    dataset_processed = preprocess(dataset, 'ff')
    clf = load('ff')
    perf = predict(dataset_processed, clf)

    print(f"Team ID: {name}")
    print("ff")
    print(f"Performance: {perf}")