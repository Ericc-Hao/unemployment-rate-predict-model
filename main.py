import subprocess
import sys
import os

def run_model(script_path):
    try:
        # Execute the script using the current Python interpreter
        subprocess.run([sys.executable, script_path], check=True)
    except subprocess.CalledProcessError as e:
        print(f"An error occurred while running {script_path}: {e}")
    except FileNotFoundError:
        print(f"Script {script_path} not found. Please check the path.")

def main():
    # Define the paths to your model scripts
    lstm_model_script = os.path.join('models', 'LSTM_model.py')
    linear_model_script = os.path.join('models', 'linear.py')
    
    # Run LSTM Model
    print("Running LSTM model now...")
    run_model(lstm_model_script)
    
    # Run Linear Regression Model
    print("Running Linear Regression model now...")
    run_model(linear_model_script)

if __name__ == "__main__":
    main()
