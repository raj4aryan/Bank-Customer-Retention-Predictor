import pickle as pkl
import torch
import pandas as pd
from ann_model_class import RegressionModel

# input_scaler = pkl.load('artifacts/reg_input_scaler.pkl')
# FIX 1: pkl.load requires a file object opened in 'rb' (read-binary) mode, not just the string path.
with open('artifacts/reg_input_scaler.pkl', 'rb') as f:
    input_scaler = pkl.load(f)

with open('artifacts/reg_transformer_model.pkl', 'rb') as f:
    input_ct = pkl.load(f)

# 3. Load the output scaler to reverse the standardization
with open('artifacts/reg_output_scaler.pkl', 'rb') as f:
    output_scaler = pkl.load(f)

# Set up prediction model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
input_size = 13
loaded_model = RegressionModel(input_size)
loaded_model.load_state_dict(torch.load("artifacts/regression_model.pth"))
loaded_model.to(device)

# Mock data from frontend
input_data = {
    'CreditScore': 600, 
    'Geography': 'France',
    'Gender': 'Male', 
    'Age': 40, 
    'Tenure': 3, 
    'Balance': 60000, 
    'NumOfProducts': 2, 
    'HasCrCard': 1,
    'IsActiveMember': 1, 
    'Exited': 1
}

# FIX 2: Scikit-learn transformers expect 2D arrays or DataFrames, not raw Python dictionaries.
input_df = pd.DataFrame([input_data])
input_data_transformed = input_ct.transform(input_df)
input_data_scaled = input_scaler.transform(input_data_transformed)

# Tensor conversion
input_data_tensor = torch.FloatTensor(input_data_scaled).to(device)

loaded_model.eval()

# --- THE PREDICTION STEP ---
# 1. Run the forward pass without tracking gradients
with torch.no_grad():
    prediction_scaled = loaded_model(input_data_tensor)

# 2. Move the tensor back to the CPU and convert to a NumPy array
prediction_numpy = prediction_scaled.cpu().numpy()

# 3. Inverse transform to get the real-world salary number
final_salary = output_scaler.inverse_transform(prediction_numpy)

print(f"Predicted Estimated Salary: {final_salary[0][0]:.2f}")



# Note: I faced the problem: ValueError: 
"""The feature names should match those that were passed during fit.
Feature names seen at fit time, yet now missing:
- Unnamed: 0" """

# This is a classic Pandas artifact error.

# When you originally saved your dataset to a CSV, Pandas likely saved the row numbers (the index) as the very first column. When your training script ran pd.read_csv(), it imported those row numbers as a literal feature named Unnamed: 0.

# Because your scaler was fit on that data, it is now strictly expecting a column named Unnamed: 0 to exist in your frontend dictionary.

"""
so whenever saving a data to csv -- df.to_csv('../dataset/RegProcessedData.csv', index=False)
Always use `index = False`
"""

# FINAL OUTPUT: "Predicted Estimated Salary: 99532.81"   