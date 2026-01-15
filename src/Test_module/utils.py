import pickle
import torch
import torch.nn as nn
import numpy as np
import pandas as pd

def internal_preprocess_logic(df_input, assets_path='preprocessing_assets.pkl', use_label_only=False):
    with open(assets_path, 'rb') as f:
        assets = pickle.load(f)
    
    df = df_input.copy()

    # Trasformazioni iniaiziali
    if 'borrower_profile_employment_length' in df.columns:
        df['borrower_profile_employment_length'] = df['borrower_profile_employment_length'].map(assets['emp_len_map']).fillna(0)

    # Gestione feature temporali
    date_cols = ['loan_issue_date', 'credit_history_earliest_line', 'last_payment_date', 'last_credit_pull_date']
    for col in date_cols:
        if col in df.columns:
            dt = pd.to_datetime(df[col], errors='coerce')
            df[f"{col}_year"] = dt.dt.year
            df[f"{col}_month"] = dt.dt.month

    if 'loan_issue_date' in df.columns and 'credit_history_earliest_line' in df.columns:
        issue_dt = pd.to_datetime(df['loan_issue_date'], errors='coerce')
        hist_dt = pd.to_datetime(df['credit_history_earliest_line'], errors='coerce')
        df['months_credit_history'] = (issue_dt.dt.year - hist_dt.dt.year) * 12 + (issue_dt.dt.month - hist_dt.dt.month)

    # Imputazione
    for col, value in assets['medians'].items():
        if col in df.columns:
            df[col] = df[col].fillna(value)
            
    for col, values in assets['modes'].items():
        if col in df.columns and col != 'grade':
            val = values[0] if isinstance(values, (list, np.ndarray)) else values
            df[col] = df[col].fillna(val)

    # Clipping
    bounds = assets.get('clipping_bounds', {})
    for col, (low, high) in bounds.items():
        if col in df.columns:
            df[col] = df[col].clip(lower=low, upper=high)

    # Encoding
    for col, ohe in assets['oh_encoders'].items():
        if col in df.columns:
            if use_label_only:
                if col in assets['l_encoders']:
                    le = assets['l_encoders'][col]
                    df[col] = df[col].astype(str).map(lambda x: le.transform([x])[0] if x in le.classes_ else -1)
            else:
                encoded_data = ohe.transform(df[[col]])
                column_names = [f"{col}_{cat}" for cat in ohe.categories_[0][1:]]
                encoded_df = pd.DataFrame(encoded_data, columns=column_names, index=df.index)
                df = pd.concat([df.drop(col, axis=1), encoded_df], axis=1)
    
    for col, le in assets['l_encoders'].items():
        if col in df.columns and (not use_label_only or col not in assets['oh_encoders']):
            df[col] = df[col].astype(str).map(lambda x: le.transform([x])[0] if x in le.classes_ else -1)

    # Allineamento finale 
    final_cols = assets['final_columns']
    for col in final_cols:
        if col not in df.columns:
            df[col] = 0
            
    return df[final_cols].fillna(0)


class FeedForward(nn.Module):
    def __init__(self, input_size, num_classes, hidden_size, dropout_prob=0.2):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_prob),

            nn.Linear(hidden_size, hidden_size // 2),
            nn.BatchNorm1d(hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout_prob),

            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.BatchNorm1d(hidden_size // 4),
            nn.ReLU(),
            nn.Dropout(dropout_prob),

            nn.Linear(hidden_size // 4, num_classes)
        )

    def forward(self, x):
        return self.net(x)

class FeedForwardWrapper:
    def __init__(self, model, scaler=None, device=None):
        self.model = model
        self.scaler = scaler
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
    
    def predict(self, X):

        if isinstance(X, pd.DataFrame):
            X = X.values

        X = X.astype(np.float32)

        if self.scaler:
            X = self.scaler.transform(X)

        X_tensor = torch.from_numpy(X).to(self.device)
        with torch.no_grad():
            outputs = self.model(X_tensor)
            preds = outputs.argmax(dim=1).cpu().numpy()
        return preds

def load_ff_model(model_path="models/ff.save", device=None):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    input_size = checkpoint["input_size"]
    num_classes = checkpoint["num_classes"]
    hidden_size = checkpoint["hidden_size"]

    model = FeedForward(input_size=input_size, num_classes=num_classes, hidden_size=hidden_size)
    model.load_state_dict(checkpoint["model_state"])
    scaler = checkpoint.get("scaler")

    return FeedForwardWrapper(model=model, scaler=scaler, device=device)