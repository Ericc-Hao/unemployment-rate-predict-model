# Group 2 Milestone 2 Code Folder

## Installing Dependencies
Install the required `Python packages` using the provided `requirement.txt` file

```bash
pip install -r requirements.txt
```

## Running the experiment

Execute both models sequentially:

```bash
python main.py
```

if you want to run Models Individually, please follow below steps:

Linear Regression:

```bash
python models/linear.py
```

LSTM Model:

```bash
python models/LSTM_model.py
```

## File Structure
```
├── README.md
├── datasets/
│   ├── filtered_data/
│   │   └── [Filtered datasets]
│   └── raw-data/
│       └── [Raw datasets downloaded from Statistics Canada]
├── main.py (Run both models)
├── models/
│   ├── data_preproces_piplines/
│   │   └── [Scripts for data cleaning and formatting]
│   ├── linear.py (linear regression model)
│   └── LSTM_model.py (LSTM model)
└── requirements.txt
```