# 🌤️ Weather Prediction - Machine Learning Project

A machine learning project that predicts temperature based on weather conditions using multiple regression algorithms.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0%2B-orange)
![License](https://img.shields.io/badge/License-MIT-green)

## 📋 About

This project demonstrates a complete ML pipeline that compares 5 different regression algorithms to predict temperature. It includes data preprocessing, feature engineering, model evaluation, and professional visualizations.

## 🎯 Key Skills Demonstrated

- **Data Preprocessing** - Feature scaling and train-test splitting
- **Feature Engineering** - Creating interaction features
- **Multiple ML Algorithms** - Linear, Ridge, Decision Tree, Random Forest, Gradient Boosting
- **Model Evaluation** - RMSE, MAE, R² Score, Cross-Validation
- **Data Visualization** - Professional plots using Matplotlib
- **Clean Code** - Object-oriented programming

## 📊 Results

The best model (Random Forest) achieves:
- **RMSE**: ~2.5°C
- **MAE**: ~1.8°C  
- **R² Score**: ~0.95

![Sample Output](weather_prediction_results.png)

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/weather-ml-prediction.git
cd weather-ml-prediction

# Install dependencies
pip install -r requirements.txt
```

### Run the Project

```bash
python weather_predictor.py
```

The script will:
1. Generate synthetic weather data
2. Train 5 different ML models
3. Compare their performance
4. Create visualization as `weather_prediction_results.png`
5. Show a demo prediction

## 💻 Making Predictions

```python
from weather_predictor import WeatherPredictor

predictor = WeatherPredictor()
# ... after training ...

temperature = predictor.predict_new_weather(
    humidity=65,       # %
    wind_speed=15,     # km/h
    pressure=1013,     # hPa
    cloud_cover=40     # %
)
print(f"Predicted: {temperature:.2f}°C")
```

## 🛠️ Technologies

- **Python 3.8+**
- **NumPy** - Numerical computations
- **Pandas** - Data manipulation
- **Scikit-learn** - ML algorithms
- **Matplotlib** - Visualization

## 📁 Project Structure

```
weather-ml-prediction/
├── weather_predictor.py           # Main code
├── requirements.txt               # Dependencies
├── README.md                      # Documentation
├── .gitignore                     # Git ignore file
└── weather_prediction_results.png # Output visualization
```

## 📈 Features

- Synthetic weather data generation
- 5 algorithm comparison
- Cross-validation for reliability
- Comprehensive metrics
- Professional visualizations
- Interactive predictions


⭐ **Star this repo if it helped you!** how shall i create this in github the branched or to add all in same branch

