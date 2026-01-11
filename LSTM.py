# lstm_stock_predictor_final.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score, recall_score, precision_score
from sklearn.model_selection import TimeSeriesSplit
from tensorflow.keras.models import Sequential, clone_model
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import ModelCheckpoint 
import math
import os

# --- Configuration ---
EXCEL_FILE_NAME = 'Kotak_15min_data.xlsx'
TIME_STEP = 60 
EPOCHS = 50
BATCH_SIZE = 64
N_SPLITS = 5 
TEST_HOLDOUT_PCT = 0.20 
CHECKPOINT_EPOCHS = 10# Save weights every 5 epochs
CHECKPOINT_DIR = 'model_checkpoints' # Directory to save checkpoints

# --- Data Utility Functions ---

def load_local_data(file_name):
    """Loads historical stock data from a local Excel file."""
    if not os.path.exists(file_name):
        raise FileNotFoundError(f"Error: File '{file_name}' not found.")
    
    print(f"Loading data from {file_name}...")
    try:
        df = pd.read_excel(file_name)
        df.columns = [col.title() for col in df.columns]
        if 'Close' not in df.columns:
            raise ValueError("Data file must contain a 'Close' price column.")
        return df
    except Exception as e:
        raise ValueError(f"Error processing data file: {e}")

def create_dataset(dataset, time_step=1):
    """
    Transforms a single column of time series data into overlapping sequences.
    X = sequences (TIME_STEP values), Y = next value (the target)
    """
    dataX, dataY = [], []
    for i in range(len(dataset) - time_step - 1):
        a = dataset[i:(i + time_step), 0]
        dataX.append(a)
        dataY.append(dataset[i + time_step, 0])
    return np.array(dataX), np.array(dataY)

def build_model(time_step):
    """Defines the LSTM model architecture."""
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(time_step, 1)))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dense(25))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

def calculate_classification_metrics(y_true, y_pred):
    """Converts regression predictions into directional predictions and calculates metrics."""
    # y_true is the actual price history at time T-1. y_pred is the predicted price at time T.
    if len(y_true) <= 1 or len(y_pred) <= 1:
        return 0.0, 0.0, 0.0

    y_pred_flat = y_pred.flatten() 
    
    # 1. Actual Direction: True Price at T vs True Price at T-1
    y_actual_direction = (y_true[1:] > y_true[:-1]).astype(int)
    
    # 2. Predicted Direction: Predicted Price at T vs True Price at T-1
    y_pred_direction = (y_pred_flat[1:] > y_true[:-1].flatten()).astype(int)
            
    # Metrics
    accuracy = accuracy_score(y_actual_direction, y_pred_direction)
    recall = recall_score(y_actual_direction, y_pred_direction, zero_division=0) 
    precision = precision_score(y_actual_direction, y_pred_direction, zero_division=0)
    
    return accuracy, recall, precision

# ----------------------------------------------------------------------
## Main Execution
# ----------------------------------------------------------------------

def main():
    try:
        # Create checkpoint directory
        os.makedirs(CHECKPOINT_DIR, exist_ok=True)

        data = load_local_data(EXCEL_FILE_NAME)
        df_close = data['Close'].values.reshape(-1, 1)
        
        # 1. Initial 80:20 Train/Test Split
        holdout_size = int(len(df_close) * TEST_HOLDOUT_PCT)
        train_cv_data_scaled = df_close[:-holdout_size]
        final_test_data_scaled = df_close[-holdout_size - TIME_STEP:]
        
        cv_scaler = MinMaxScaler(feature_range=(0, 1))
        train_cv_data_scaled = cv_scaler.fit_transform(train_cv_data_scaled)
        final_test_data_scaled = cv_scaler.transform(final_test_data_scaled)

        # 2. Create Sequence Datasets
        X_cv, y_cv = create_dataset(train_cv_data_scaled, TIME_STEP)
        X_final_test, y_final_test = create_dataset(final_test_data_scaled, TIME_STEP)
        
        X_cv = X_cv.reshape(X_cv.shape[0], X_cv.shape[1], 1)
        X_final_test = X_final_test.reshape(X_final_test.shape[0], X_final_test.shape[1], 1)

        print(f"\nTotal samples for CV (80%): {len(X_cv)}")
        print(f"Total samples for Final Holdout Test (20%): {len(X_final_test)}")
        
        # 3. K-FOLD CROSS-VALIDATION (TSS) on the 80% Data
        tscv = TimeSeriesSplit(n_splits=N_SPLITS)
        cv_results = []
        base_model = build_model(TIME_STEP)
        
        print("\n" + "="*70)
        print(f"🔬 Running {N_SPLITS}-Fold Time Series Cross-Validation (on 80% data)")
        print("="*70)

        for fold, (train_index, val_index) in enumerate(tscv.split(X_cv)):
            
            X_train, X_val = X_cv[train_index], X_cv[val_index]
            y_train, y_val = y_cv[train_index], y_cv[val_index]
            
            print(f"\n--- FOLD {fold + 1}/{N_SPLITS} ---")
            print(f"Training on {len(X_train)} samples, Validating on {len(X_val)} samples.")

            model = clone_model(base_model)
            model.compile(loss='mean_squared_error', optimizer='adam')

            steps_per_epoch = math.ceil(len(X_train) / BATCH_SIZE)
            save_freq_batches = steps_per_epoch * CHECKPOINT_EPOCHS
            
            checkpoint_filepath = os.path.join(CHECKPOINT_DIR, f'fold_{fold+1}_epoch_{{epoch:02d}}.weights.h5')
            model_checkpoint_callback = ModelCheckpoint(
                filepath=checkpoint_filepath,
                monitor='val_loss',
                verbose=1, # Set to 1 so you can see when it saves
                save_best_only=False,
                save_weights_only=True,
                mode='min',
                save_freq=save_freq_batches # Corrected to calculate batches
            )
            
            # Train and validate with checkpoint
            history = model.fit(X_train, y_train, 
                                epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=0,
                                validation_data=(X_val, y_val),
                                callbacks=[model_checkpoint_callback])
            # --- CHECKPOINT CALLBACK (Corrected save_freq and filepath extension) ---
            # checkpoint_filepath = os.path.join(CHECKPOINT_DIR, f'fold_{fold+1}_epoch_{{epoch:02d}}.weights.h5')
            # model_checkpoint_callback = ModelCheckpoint(
            #     filepath=checkpoint_filepath,
            #     monitor='val_loss',
            #     verbose=0,
            #     save_best_only=False,
            #     save_weights_only=True,
            #     mode='min',
            #     save_freq=CHECKPOINT_EPOCHS # Corrected to integer value
            # )
            
            # Train and validate with checkpoint
            history = model.fit(X_train, y_train, 
                                epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=0,
                                validation_data=(X_val, y_val),
                                callbacks=[model_checkpoint_callback])
            
            print("Epoch Losses:")
            for e in range(EPOCHS):
                loss = history.history['loss'][e]
                val_loss = history.history['val_loss'][e]
                print(f"  Epoch {e+1}: Loss={loss:.6f}, Val_Loss={val_loss:.6f}")

            # Predict and evaluate
            y_val_predict_scaled = model.predict(X_val, verbose=0)
            y_val_predict = cv_scaler.inverse_transform(y_val_predict_scaled)
            y_val_actual = cv_scaler.inverse_transform(y_val.reshape(-1, 1))
            
            y_val_history = cv_scaler.inverse_transform(X_val[:, -1])
            
            rmse = math.sqrt(mean_squared_error(y_val_actual, y_val_predict))
            mae = mean_absolute_error(y_val_actual, y_val_predict)
            acc, recall, precision = calculate_classification_metrics(y_val_history, y_val_predict)

            cv_results.append({'RMSE': rmse, 'MAE': mae, 'Accuracy': acc, 'Recall': recall, 'Precision': precision})
            
            print(f"Fold {fold + 1} Metrics: RMSE={rmse:.4f}, MAE={mae:.4f}, Acc={acc:.2f}")

        # 4. Report CV Results
        avg_results = pd.DataFrame(cv_results).mean()
        
        # 5. Final Test on the 20% Holdout Data
        print("\n" + "="*70)
        print("--- Final Evaluation on 20% Holdout Test Set ---")
        
        final_model = build_model(TIME_STEP)
        
        steps_per_epoch_final = math.ceil(len(X_cv) / BATCH_SIZE)
        save_freq_batches_final = steps_per_epoch_final * CHECKPOINT_EPOCHS

        final_checkpoint_filepath = os.path.join(CHECKPOINT_DIR, 'final_model_epoch_{epoch:02d}.weights.h5')
        final_checkpoint_callback = ModelCheckpoint(
            filepath=final_checkpoint_filepath,
            monitor='loss', 
            verbose=1,
            save_best_only=False,
            save_weights_only=True,
            mode='min',
            save_freq=save_freq_batches_final # Corrected
        )

        print(f"Training Final Model on ALL {len(X_cv)} CV samples...")
        final_history = final_model.fit(X_cv, y_cv, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=0,
                                         callbacks=[final_checkpoint_callback])
        print("Final Model Epoch Losses:")
        for e in range(EPOCHS):
            loss = final_history.history['loss'][e]
            print(f"  Epoch {e+1}: Loss={loss:.6f}")
        
        final_predict_scaled = final_model.predict(X_final_test, verbose=0)
        final_predict = cv_scaler.inverse_transform(final_predict_scaled)
        y_final_actual = cv_scaler.inverse_transform(y_final_test.reshape(-1, 1))

        y_final_history = cv_scaler.inverse_transform(X_final_test[:, -1])
        
        final_rmse = math.sqrt(mean_squared_error(y_final_actual, final_predict))
        final_mae = mean_absolute_error(y_final_actual, final_predict)
        final_acc, final_recall, final_precision = calculate_classification_metrics(y_final_history, final_predict)

        print("\n" + "="*70)
        print("🏆 FINAL RESULTS SUMMARY")
        print("="*70)
        
        print("--- Cross-Validation Averages ---")
        print(f"Average RMSE: {avg_results['RMSE']:.4f}")
        print(f"Average Accuracy: {avg_results['Accuracy']*100:.2f}%")
        
        print("\n--- Final 20% Holdout Test Results (Unbiased) ---")
        print(f"Final Root Mean Squared Error (RMSE): {final_rmse:.4f}")
        print(f"Final Mean Absolute Error (MAE): {final_mae:.4f}")
        print(f"Final Accuracy (Directional): {final_acc*100:.2f}%")
        print(f"Final Precision (Reliability of 'Up' calls): {final_precision*100:.2f}%")
        print("="*70)
        
    except FileNotFoundError as e:
        print(f"Execution Error: {e}")
    except ValueError as e:
        print(f"Execution Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == '__main__':
    main()