# Unemployment Rate Predict Model

A comprehensive machine learning system for predicting unemployment rates in Canada using multiple economic indicators. This project combines traditional statistical methods with modern deep learning approaches to provide accurate short-term unemployment rate forecasts.

## 📖 Project Description

For detailed information about the project's technology stack, methodology, and architecture, please see our comprehensive [Project Description](PROJECT_DESCRIPTION.md).

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

## 👥 Credits

This project was developed collaboratively by:

- **[Yunze(Figo) Li](https://github.com/Figo-Li)**
- **[Qiwen(Kyra) Jiao](https://github.com/jqiwen)**
- **[Chenrui(Eric) Hao](https://github.com/Ericc-Hao)**

*Please replace the placeholder names and GitHub usernames with the actual contributors to this project.*