import torch, torch.nn as nn, torch.optim as optim
import pandas as pd
import tensorboard 
import os
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import datetime
from torch.utils.tensorboard import SummaryWriter
import pickle as pkl

from ann_model_class import RegressionModel

class ModelPipeline:
    def __init__(self, path):
        self.path = path #os.path.join(path)

    def data_handler(self, target_col):
        data = pd.read_csv(self.path)
        X = data.drop([target_col], axis = 1)
        y = data[target_col].values.reshape(-1, 1)
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=42)
        # fit_transform and transform output clean NumPy arrays, stripping Pandas formatting
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        # Scale y (NEW STEP)
        y_scaler = StandardScaler()
        y_train_scaled = y_scaler.fit_transform(y_train)
        y_test_scaled = y_scaler.transform(y_test)

        # Now creating torch datatype -- the below 2 lines are WRONG
        # device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # X_train_tensor = torch.FloatTensor(X_train_scaled.to_device(device).values)

        # PyTorch requires a strict sequence: you must convert the data into a PyTorch object before you can send it to the PyTorch device (GPU).
        # Wrong Object: You tried to move the data to the GPU before turning it into a Tensor. Only PyTorch Models and PyTorch Tensors can be sent to a PyTorch device.

        # FIX 2: Create a formal PyTorch device object, not just a string
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # FIX 3: Convert the NumPy array to a Tensor FIRST, then move it to the device.
        # We do NOT use .values here because X_train_scaled is already a NumPy array.
        X_train_tensor = torch.FloatTensor(X_train_scaled).to(device)
        X_test_tensor = torch.FloatTensor(X_test_scaled).to(device)

        y_train_tensor = torch.FloatTensor(y_train_scaled).to(device)
        y_test_tensor = torch.FloatTensor(y_test_scaled).to(device)

        with open('artifacts/reg_input_scaler.pkl', 'wb') as f:
            pkl.dump(scaler, f)
        with open('artifacts/reg_output_scaler.pkl', 'wb') as f:
            pkl.dump(y_scaler, f)

        return X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor, device
    
    def model_trainer(self, X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor, device: torch.device): 

        # Setup
        input_size = X_train_tensor.shape[1]
        # 2. FIX: The model must be created BEFORE the optimizer so the optimizer can target its weights.
        model = RegressionModel(input_size).to(device)

        # optimiser = optim.Adam()
        optimiser = optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.MSELoss()

        log_dir = 'logs/fit/' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        writer = SummaryWriter(log_dir)

        patience = 50
        best_val_loss = float('inf')
        epochs_no_improve = 0
        best_model_weights = None
        num_epochs = 500

        print(f"Data is physically sitting on: {X_train_tensor.device}")
        print(f"Model is physically sitting on: {next(model.parameters()).device}")
        for epoch in range(num_epochs):
            model.train()
            #Forward Pass
            outputs = model(X_train_tensor)
            loss = criterion(outputs, y_train_tensor)

            #Backward Pass
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

            #Validation Phase
            model.eval()

            with torch.no_grad():
                val_outputs = model(X_test_tensor)
                val_loss = criterion(val_outputs, y_test_tensor)

            writer.add_scalar('Loss/train', loss.item(), epoch)
            writer.add_scalar('Loss/validation', val_loss.item(), epoch)
            # writer.add_scalar()

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_weights = model.state_dict()
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve>=patience:
                    print(f"Early Stopping triggered at epoch {epoch}. Restoring best weights")
                    model.load_state_dict(best_model_weights)
                    break

            if (epoch + 1)%10 == 0:
                print(f"Epoch: [{epoch+1}/{num_epochs}], Train Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}")

        writer.close()
        # Save the finalized model weights
        save_path = "artifacts/regression_model.pth"
        torch.save(model.state_dict(), save_path)
        print(f"Model successfully saved to {save_path}")

if __name__ == "__main__":
    pipeline = ModelPipeline("dataset/RegProcessedData.csv")
    X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor, device = pipeline.data_handler('EstimatedSalary')
    pipeline.model_trainer(X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor, device)




#NOTE: 
# Initially, the training looked like: 
"""Epoch: [10/100], Train Loss: 13392844800.0000, Val Loss: 13046122496.0000
Epoch: [20/100], Train Loss: 13385519104.0000, Val Loss: 13037999104.0000
Epoch: [30/100], Train Loss: 13364594688.0000, Val Loss: 13015472128.0000
Epoch: [40/100], Train Loss: 13317353472.0000, Val Loss: 12965551104.0000
Epoch: [50/100], Train Loss: 13226217472.0000, Val Loss: 12870531072.0000
Epoch: [60/100], Train Loss: 13069675520.0000, Val Loss: 12708978688.0000
Epoch: [70/100], Train Loss: 12823986176.0000, Val Loss: 12457506816.0000
Epoch: [80/100], Train Loss: 12465550336.0000, Val Loss: 12093239296.0000
Epoch: [90/100], Train Loss: 11973991424.0000, Val Loss: 11596841984.0000
Epoch: [100/100], Train Loss: 11335830528.0000, Val Loss: 10956273664.0000"""

# The Cause: Unscaled Target Variables
# Earlier in your data_handler code, you used StandardScaler() on your X (features), but you left your y (target) untouched.

# Since you switched to a Regression model, I assume your y column is no longer a simple 0 or 1, but a large continuous number (like transaction amounts or account balances).

# MSE calculates the error and squares it. If your model predicts a transaction is $50,000, but the actual answer is $150,000, the error is $100,000. MSE squares that error, resulting in a loss of 10,000,000,000 for just that one single row!

"""How to Fix It
1. Scale your Target Variable (y)
    Neural networks struggle to output massive numbers naturally. You need to scale your target variable down to small numbers (like between -3 and 3) just like you did with your features.

Update your data_handler to include a second scaler specifically for y

2. Increase your Epochs
    Notice how your loss was still dropping aggressively between epoch 90 and 100? Your model hasn't finished learning yet. Because it had to navigate such massive numbers, 100 epochs wasn't enough time. Try bumping num_epochs up to 500 or 1000

3. Lower the Learning Rate:
    A learning rate of 0.01 is generally too aggressive for the Adam optimizer on regression tasks. It causes the weights to bounce wildly over the optimal minimum and get stuck in a dead zone. Change it to the standard default:
    optimiser = optim.Adam(model.parameters(), lr=0.001)    

4. Increase Early Stopping Patience: 
    A patience of 10 is far too restrictive for a 500-epoch run. The model needs a "warm-up" period to navigate out of the initial random weights. Change patience = 50 so the model has room to experiment and find a downward gradient.
"""



#LATER
#output:
"""Data is physically sitting on: cuda:0
Model is physically sitting on: cuda:0
Epoch: [10/500], Train Loss: 0.9950, Val Loss: 1.0016
Epoch: [20/500], Train Loss: 0.9908, Val Loss: 1.0044
Epoch: [30/500], Train Loss: 0.9871, Val Loss: 1.0081
Epoch: [40/500], Train Loss: 0.9830, Val Loss: 1.0122
Epoch: [50/500], Train Loss: 0.9784, Val Loss: 1.0165
Early Stopping triggered at epoch 50. Restoring best weights
Model successfully saved to artifacts/regression_model.pth"""

# the terminal output reveals a fundamental data problem causing a mathematical collapse: Train Loss is slowly decreasing (0.9950 → 0.9784) while Validation Loss is strictly increasing (1.0016 → 1.0165).

# This exact inverse relationship is the textbook definition of a network OVERFITTING to noise.

# Here is exactly why the model is behaving this way:

# Zero Feature Correlation: When predicting the Fire Weather Index using meteorological data from Algeria, independent variables like temperature and humidity had a direct, causal mathematical relationship to the target. In this banking dataset, features like NumOfProducts, HasCrCard, or Tenure do not mathematically dictate a person's exact EstimatedSalary.

# Memorizing Randomness: Because the network cannot find any real underlying patterns, it starts memorizing the specific random variations of the training rows (which slowly lowers the Train Loss). When it attempts to apply that memorized noise to the unseen validation data, it fails completely, causing the Validation Loss to climb immediately from epoch 10 onward.

# The 1.0 Baseline: An MSE hovering precisely around 1.0 on a standard-scaled target variable means the model has given up on finding patterns and is just defaulting to predicting the exact average salary for every single customer.