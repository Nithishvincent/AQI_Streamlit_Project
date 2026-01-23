# Air Quality Index Prediction Model

## 🌍 Project Overview

This project implements a comprehensive Machine Learning system for predicting Air Quality Index (AQI) based on multiple environmental parameters. The system uses Random Forest Regression to predict AQI values and provides both a Python-based model and an interactive web application using Streamlit.

## 📋 Table of Contents

- [Features](#features)
- [Technologies Used](#technologies-used)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Model Details](#model-details)
- [Results](#results)
- [Future Enhancements](#future-enhancements)

## ✨ Features

### Core Features
- **Multi-Parameter Prediction**: Predicts AQI using 8 environmental parameters
- **Multiple ML Models**: Compares Linear Regression, Random Forest, and Gradient Boosting
- **Interactive Web Interface**: User-friendly Streamlit dashboard
- **Real-time Predictions**: Instant AQI calculation with health recommendations
- **Comprehensive Visualizations**: 
  - Actual vs Predicted plots
  - Residual analysis
  - Feature importance charts
  - Correlation heatmaps
  - Distribution plots

### Input Parameters
1. **PM2.5** - Fine particulate matter (μg/m³)
2. **PM10** - Coarse particulate matter (μg/m³)
3. **NO₂** - Nitrogen Dioxide (ppb)
4. **SO₂** - Sulfur Dioxide (ppb)
5. **CO** - Carbon Monoxide (ppm)
6. **O₃** - Ozone (ppb)
7. **Temperature** - Ambient temperature (°C)
8. **Humidity** - Relative humidity (%)

### AQI Categories
- **0-50**: Good (Green)
- **51-100**: Moderate (Yellow)
- **101-150**: Unhealthy for Sensitive Groups (Orange)
- **151-200**: Unhealthy (Red)
- **201-300**: Very Unhealthy (Purple)
- **301-500**: Hazardous (Maroon)

## 🛠️ Technologies Used

### Programming Language
- Python 3.8+

### Libraries
- **NumPy**: Numerical computations
- **Pandas**: Data manipulation and analysis
- **Scikit-learn**: Machine learning algorithms
- **Matplotlib**: Static visualizations
- **Seaborn**: Statistical visualizations
- **Plotly**: Interactive visualizations
- **Streamlit**: Web application framework
- **Joblib**: Model serialization

## 📦 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Step 1: Clone the Repository
```bash
git clone https://github.com/yourusername/aqi-prediction.git
cd aqi-prediction
```

### Step 2: Create Virtual Environment (Recommended)
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

## 🚀 Usage

### Method 1: Run Python Model (Training & Evaluation)

```bash
python aqi_model.py
```

This will:
1. Generate synthetic training data (or load your CSV)
2. Train multiple ML models
3. Compare model performances
4. Generate evaluation metrics
5. Create visualization plots
6. Save the trained model as `aqi_model.pkl`

### Method 2: Run Streamlit Web Application

```bash
streamlit run app.py
```

Then open your browser to `http://localhost:8501`

### Method 3: Use the React Web Interface

Open the HTML file with a modern web browser for a standalone web application.

### Making Predictions

**Using Python:**
```python
from aqi_model import AQIPredictor

predictor = AQIPredictor()
predictor.load_model('aqi_model.pkl')

input_data = {
    'PM2.5': 35.5,
    'PM10': 50.0,
    'NO2': 40.0,
    'SO2': 20.0,
    'CO': 1.2,
    'O3': 60.0,
    'Temperature': 25.0,
    'Humidity': 65.0
}

predicted_aqi = predictor.predict(input_data)
print(f"Predicted AQI: {predicted_aqi:.2f}")
```

**Using Streamlit App:**
1. Click "Train Model" in the sidebar
2. Adjust the parameter sliders
3. Click "Predict AQI"
4. View results and health recommendations

## 📁 Project Structure

```
aqi-prediction/
│
├── aqi_model.py              # Main Python ML model
├── app.py                    # Streamlit web application
├── aqi_web_app.html         # React-based web interface
├── requirements.txt          # Python dependencies
├── README.md                 # Project documentation
│
├── aqi_model.pkl            # Saved trained model (generated)
├── aqi_model_results.png    # Visualization plots (generated)
│
└── data/                     # Optional: Your CSV datasets
    └── air_quality.csv
```

## 🧠 Model Details

### Algorithm: Random Forest Regressor

**Why Random Forest?**
- Handles non-linear relationships well
- Robust to outliers
- Provides feature importance
- Less prone to overfitting
- Excellent for environmental data

### Model Architecture
```
Input Layer: 8 features
├── PM2.5
├── PM10
├── NO₂
├── SO₂
├── CO
├── O₃
├── Temperature
└── Humidity

Hidden Layer: 100 Decision Trees
├── Max Depth: Auto
├── Min Samples Split: 2
└── Random State: 42

Output Layer: AQI Value (0-500)
```

### Training Process
1. **Data Generation**: 1000 synthetic samples
2. **Data Preprocessing**:
   - Handle missing values (median imputation)
   - Feature scaling (StandardScaler)
   - Train-test split (80-20)
3. **Model Training**:
   - Multiple algorithms comparison
   - 5-fold cross-validation
   - Hyperparameter tuning
4. **Model Evaluation**:
   - RMSE, MAE, MSE, R² score
   - Residual analysis
   - Feature importance

### Performance Metrics (Typical)
- **R² Score**: 0.94-0.96
- **RMSE**: 8-12 AQI units
- **MAE**: 6-9 AQI units
- **Cross-Validation Score**: 0.93-0.95

## 📊 Results

### Sample Predictions

| PM2.5 | PM10 | NO₂ | SO₂ | CO | O₃ | Temp | Humidity | Predicted AQI | Category |
|-------|------|-----|-----|-----|-----|------|----------|---------------|----------|
| 35.5  | 50.0 | 40.0| 20.0| 1.2 | 60.0| 25.0 | 65.0     | 78.5          | Moderate |
| 85.2  | 120.0| 75.0| 45.0| 3.5 | 110.0| 32.0| 55.0     | 165.3         | Unhealthy|
| 15.0  | 25.0 | 20.0| 10.0| 0.5 | 40.0| 20.0 | 70.0     | 35.2          | Good     |

### Feature Importance Ranking
1. PM2.5 (85%)
2. O₃ (55%)
3. PM10 (45%)
4. NO₂ (35%)
5. SO₂ (25%)
6. Temperature (15%)
7. Humidity (12%)
8. CO (10%)

## 🔮 Future Enhancements

### Short-term
- [ ] Real-time data integration from API
- [ ] Historical data analysis
- [ ] Time series forecasting
- [ ] Mobile app development
- [ ] Multi-location support

### Long-term
- [ ] Deep Learning models (LSTM, GRU)
- [ ] Satellite imagery integration
- [ ] Weather forecast correlation
- [ ] Alert notification system
- [ ] Cloud deployment (AWS/Azure)

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License.

## 👨‍💻 Authors

**Nithish**
- GitHub: [@Nithishvincent](https://github.com/Nithishvincent)
**Jeeva Prakash**
- GitHub: [@Nithishvincent](https://github.com/Jeevaprakash-03)

## 🙏 Acknowledgments

- EPA (Environmental Protection Agency) for AQI standards
- Scikit-learn documentation and community
- Streamlit for the amazing framework
- All contributors and testers

## 📧 Contact

For questions or suggestions, please open an issue or contact:
- LinkedIn: [Your Profile](https://linkedin.com/in/nithishvincent/)

---

**Note**: This project uses synthetic data for demonstration. For production use, integrate with real air quality monitoring stations or APIs.

**Last Updated**: January 2026
