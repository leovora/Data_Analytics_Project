from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import balanced_accuracy_score
import pickle
import pandas as pd
import os
from utils import load_ff_model, internal_preprocess_logic
from pytorch_tabnet.tab_model import TabNetClassifier
import numpy as np
import numpy._core.numeric
from pytorch_tabular import TabularModel
import torch

matricola_leonardo = "0001186597"
matricola_carlotta = "0001181860"
MY_UNIQUE_ID = f"{matricola_leonardo}_{matricola_carlotta}"

# Output: unique ID of the team
def getName():
    return MY_UNIQUE_ID


# Input: Dataset dictionary and classifier name
# Output: PreProcessed Dataset dictionary
def preprocess(dataset, clfName):
    
    X = dataset['data']
    y = dataset['target'] 
    
    # Caricamento asset
    asset_file = "preprocessing_assets.pkl"
    
    # Preprocess di X
    X_processed = internal_preprocess_logic(X, assets_path=asset_file)
    
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
        clf = pickle.load(open("knn_pipeline.save", 'rb'))
        
    elif (clfName == "svm"):
        clf = pickle.load(open("svm_pipeline.save", 'rb'))

    elif (clfName == "rf"):
        clf = pickle.load(open("rf_pipeline.save", 'rb'))

    elif (clfName == "ff"):
        clf = load_ff_model("ff.save")
    
    elif (clfName == "tb"):
        tb_model = TabNetClassifier()
        tb_model.load_model("tabnet.zip")
        clf = tb_model
    
    elif (clfName == "tf"):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        clf = TabularModel.load_model("tabtransformer", map_location="cpu")

    return clf


# Input: PreProcessed Dataset dictionary, Classifier Name, Classifier Object 
# Output: Performance dictionary
def predict(dataset, clf):
    X = dataset['data']
    y = dataset['target']

    if isinstance(clf, TabularModel):
        pred_df = clf.predict(X)
        y_pred = pred_df["grade_prediction"]
    else:
        X_array = X.values if isinstance(X, pd.DataFrame) else X
        y_pred = clf.predict(X_array)

    acc = accuracy_score(y, y_pred)
    bacc = balanced_accuracy_score(y, y_pred)
    f1 = f1_score(y, y_pred, average="weighted")
    
    perf = {"acc": acc, "bacc": bacc, "f1": f1}
    
    return perf
    
# # TEST NOSTRO
# if __name__ == "__main__":
    
#     df = pd.read_csv("test_raw.csv")
    
#     dataset = {
#         'data': df.drop(columns=['grade']), 
#         'target': df['grade']
#     }

#     name = getName()

#     # ---- TEST KNN ----
#     print("----------------- KNN -----------------")
#     dataset_processed = preprocess(dataset, 'knn')
#     clf = load('knn')
#     perf = predict(dataset_processed, clf)

#     print(f"Team ID: {name}")
#     print("knn")
#     print(f"Performance: {perf}")

#     # ---- TEST RF ----
#     print("----------------- RF -----------------")
#     dataset_processed = preprocess(dataset, 'rf')
#     clf = load('rf')
#     perf = predict(dataset_processed, clf)

#     print(f"Team ID: {name}")
#     print("rf")
#     print(f"Performance: {perf}")

#     # ---- TEST SVM ----
#     print("----------------- SVM -----------------")
#     dataset_processed = preprocess(dataset, 'svm')
#     clf = load('svm')
#     perf = predict(dataset_processed, clf)

#     print(f"Team ID: {name}")
#     print("svm")
#     print(f"Performance: {perf}")

#     # ---- TEST FF ----
#     print("----------------- FF -----------------")
#     dataset_processed = preprocess(dataset, 'ff')
#     clf = load('ff')
#     perf = predict(dataset_processed, clf)

#     print(f"Team ID: {name}")
#     print("ff")
#     print(f"Performance: {perf}")

#     # ---- TEST TB ----
#     print("----------------- TB -----------------")
#     dataset_processed = preprocess(dataset, 'tb')
#     clf = load('tb')
#     perf = predict(dataset_processed, clf)

#     print(f"Team ID: {name}")
#     print("tb")
#     print(f"Performance: {perf}")

#     # ---- TEST TF ----
#     print("----------------- TF -----------------")
#     dataset_processed = preprocess(dataset, 'tf')
#     clf = load('tf')
#     perf = predict(dataset_processed, clf)

#     print(f"Team ID: {name}")
#     print("tf")
#     print(f"Performance: {perf}")