# Unemployment Rate Prediction Model

## Project Overview

This project implements a comprehensive machine learning system for predicting unemployment rates in Canada using multiple economic indicators. The system combines traditional statistical methods with modern deep learning approaches to provide accurate short-term unemployment rate forecasts.

## Core Idea

The project leverages the relationship between unemployment rates and various macroeconomic indicators to build predictive models. By analyzing historical patterns in economic data from 2003-2024, the system can forecast unemployment rates 2 months into the future, providing valuable insights for economic planning and policy decisions.

## Technology Stack

### Machine Learning Frameworks
- **TensorFlow/Keras**: Deep learning framework for LSTM neural networks
- **Scikit-learn**: Traditional machine learning algorithms and preprocessing tools
- **NumPy & Pandas**: Data manipulation and numerical computing

### Data Processing & Visualization
- **Matplotlib & Seaborn**: Data visualization and plotting
- **Tabulate**: Professional table formatting for results display

### Key Libraries
- **h5py**: Model serialization and storage
- **Joblib**: Model persistence and loading
- **DateTime**: Time series data handling

## Architecture & Methodology

### 1. Data Sources & Collection
The system integrates multiple economic datasets from Statistics Canada:

- **Unemployment Data**: Monthly unemployment rates by province (2003-2024)
- **Consumer Price Index (CPI)**: Inflation indicators across provinces
- **Gross Domestic Product (GDP)**: Economic output metrics
- **Minimum Wage Data**: Provincial wage floor information
- **Population & Participation Rates**: Labor market demographics

### 2. Data Preprocessing Pipeline

#### Feature Engineering
- **Cyclical Encoding**: Monthly seasonality captured using sine/cosine transformations
- **Temporal Alignment**: Multi-source data synchronized to monthly intervals
- **Missing Value Handling**: Forward-fill strategy for incomplete time series
- **Geographic Filtering**: Focus on Canada-wide aggregated data

#### Data Scaling
- **RobustScaler**: Handles outliers in economic indicators
- **StandardScaler**: Normalizes features for neural network training
- **Target Scaling**: Unemployment rates scaled for model optimization

### 3. Model Architecture

#### LSTM Neural Network
```
Input Layer → Bidirectional LSTM (64 units) → Dropout (0.3)
           → LSTM (32 units) → Dropout (0.3) → Dense (1 unit)
```

**Key Features:**
- **Bidirectional Processing**: Captures both forward and backward temporal dependencies
- **Regularization**: L2 regularization (0.001) and dropout layers prevent overfitting
- **Sequence Learning**: 12-month sliding window for temporal pattern recognition
- **Early Stopping**: Prevents overfitting with patience-based training termination

#### Linear Regression Baseline
- **Traditional Statistical Model**: Provides interpretable baseline performance
- **Feature Importance**: Reveals which economic indicators most influence unemployment
- **Fast Training**: Quick model comparison and validation

### 4. Training Strategy

#### Time Series Validation
- **Temporal Split**: Last 2 months reserved for testing (realistic forecasting scenario)
- **Sliding Window**: 12-month sequences for LSTM training
- **Future Prediction**: 2-month ahead forecasting (realistic policy planning horizon)

#### Hyperparameters
- **Learning Rate**: 0.0001 (Adam optimizer)
- **Batch Size**: 8 (small batches for stable gradient updates)
- **Epochs**: 100 (with early stopping)
- **Validation Split**: 20% for model selection

### 5. Evaluation Metrics

#### Performance Measures
- **Mean Absolute Error (MAE)**: Average prediction deviation
- **Mean Squared Error (MSE)**: Penalizes larger prediction errors
- **Visual Analysis**: Time series plots comparing predictions vs. actual values

#### Model Comparison
- **LSTM vs. Linear Regression**: Deep learning vs. traditional statistical approach
- **Training History**: Loss curves showing model convergence
- **Prediction Accuracy**: Month-by-month forecast comparison

## Key Innovations

### 1. Multi-Source Data Integration
- **Comprehensive Economic View**: Combines labor, inflation, output, and wage data
- **Provincial Aggregation**: Canada-wide perspective with regional data sources
- **Temporal Consistency**: Handles different data collection frequencies

### 2. Advanced Feature Engineering
- **Cyclical Time Encoding**: Captures seasonal unemployment patterns
- **Economic Indicator Fusion**: Multiple correlated variables for robust predictions
- **Scalable Preprocessing**: Handles missing data and outliers gracefully

### 3. Hybrid Modeling Approach
- **Deep Learning + Traditional ML**: LSTM for complex patterns, Linear Regression for interpretability
- **Ensemble Potential**: Framework supports model combination strategies
- **Baseline Comparison**: Clear performance benchmarking

### 4. Production-Ready Implementation
- **Model Persistence**: Saved models for deployment
- **Scalable Architecture**: Modular design for easy extension
- **Comprehensive Evaluation**: Multiple metrics and visualizations

## Applications & Impact

### Economic Forecasting
- **Policy Planning**: Government agencies can anticipate unemployment trends
- **Business Strategy**: Companies can adjust hiring and expansion plans
- **Academic Research**: Foundation for economic modeling studies

### Technical Contributions
- **Time Series Methodology**: Demonstrates effective LSTM application to economic data
- **Data Integration Techniques**: Shows how to combine disparate economic datasets
- **Model Evaluation Framework**: Comprehensive approach to forecasting model assessment

## Future Enhancements

### Model Improvements
- **Attention Mechanisms**: Transformer-based models for better long-term dependencies
- **Ensemble Methods**: Combine multiple models for improved accuracy
- **Real-time Updates**: Streaming data integration for live predictions

### Feature Expansion
- **External Indicators**: Interest rates, trade data, commodity prices
- **Sentiment Analysis**: News and social media economic sentiment
- **Regional Models**: Province-specific unemployment forecasting

### Deployment Features
- **API Development**: RESTful service for real-time predictions
- **Dashboard Interface**: Interactive visualization and monitoring
- **Automated Retraining**: Self-updating models with new data

## Technical Specifications

### System Requirements
- **Python 3.8+**: Core programming language
- **TensorFlow 2.18.0**: Deep learning framework
- **Memory**: 8GB+ RAM recommended for model training
- **Storage**: ~500MB for datasets and models

### Performance Characteristics
- **Training Time**: ~5-10 minutes on modern hardware
- **Prediction Speed**: <1 second for new forecasts
- **Model Size**: ~50MB for saved LSTM model
- **Data Volume**: ~1,700 monthly observations across multiple indicators

This project represents a comprehensive approach to economic forecasting, combining domain expertise in labor economics with modern machine learning techniques to create a robust, scalable unemployment prediction system.
