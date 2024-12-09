# Group 2 Milestone 2 Code Folder

## Installing Dependencies
Install the required `Python packages` using the provided `requirement.txt` file

```bash
pip install -r requirements.txt
```

## Running the experiment

Execute Training process

```bash
python Training.py
```

Execute Test process

```bash
python Test.py
```

## File Structure
```
├── README.md
├── datasets/
│   ├── filtered_data/
│   │   └── [Filtered datasets]
│   └── raw-data/
│       └── [Raw datasets downloaded from Statistics Canada]
├── Test.py (Testing and validations)
├── Training.py (Training LSTM and LR models)
├── models/
│   ├── data_preproces_piplines/
│   │   └── [Scripts for data cleaning and formatting]
│   ├── [saved training and validation losses during training process of LSTM model]
│   └── [saved model files]
└── requirements.txt
```