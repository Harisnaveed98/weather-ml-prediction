"""
Weather Prediction ML Project
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

class WeatherPredictor:
    """ML model to predict temperature based on weather features"""
    
    def __init__(self):
        self.models = {
            'Linear Regression': LinearRegression(),
            'Ridge Regression': Ridge(alpha=1.0),
            'Decision Tree': DecisionTreeRegressor(max_depth=5, random_state=42),
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
        }
        self.scaler = StandardScaler()
        self.best_model = None
        self.best_model_name = None
        
    def generate_synthetic_data(self, n_samples=1000):
        """Generate realistic synthetic weather data"""
        np.random.seed(42)
        
        # Features
        humidity = np.random.uniform(30, 90, n_samples)
        wind_speed = np.random.uniform(0, 30, n_samples)
        pressure = np.random.uniform(980, 1040, n_samples)
        cloud_cover = np.random.uniform(0, 100, n_samples)
        
        # Target: Temperature with realistic relationships
        temperature = (
            15 + 
            0.2 * humidity + 
            -0.5 * wind_speed + 
            0.3 * pressure + 
            -0.15 * cloud_cover +
            np.random.normal(0, 3, n_samples)
        )
        
        df = pd.DataFrame({
            'humidity': humidity,
            'wind_speed': wind_speed,
            'pressure': pressure,
            'cloud_cover': cloud_cover,
            'temperature': temperature
        })
        
        return df
    
    def preprocess_data(self, df):
        """Feature engineering and preprocessing"""
        # Create interaction features
        df['humidity_wind'] = df['humidity'] * df['wind_speed']
        df['pressure_cloud'] = df['pressure'] * df['cloud_cover']
        
        # Split features and target
        X = df.drop('temperature', axis=1)
        y = df['temperature']
        
        return X, y
    
    def train_and_evaluate(self, X, y):
        """Train multiple models and compare performance"""
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        results = {}
        
        print("=" * 70)
        print("MODEL EVALUATION RESULTS")
        print("=" * 70)
        
        for name, model in self.models.items():
            # Train model
            model.fit(X_train_scaled, y_train)
            
            # Predictions
            y_pred = model.predict(X_test_scaled)
            
            # Metrics
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            # Cross-validation
            cv_scores = cross_val_score(model, X_train_scaled, y_train, 
                                       cv=5, scoring='r2')
            
            results[name] = {
                'model': model,
                'rmse': rmse,
                'mae': mae,
                'r2': r2,
                'cv_mean': cv_scores.mean(),
                'predictions': y_pred
            }
            
            print(f"\n{name}:")
            print(f"  RMSE: {rmse:.3f}")
            print(f"  MAE:  {mae:.3f}")
            print(f"  R²:   {r2:.3f}")
            print(f"  CV R² (mean): {cv_scores.mean():.3f} (+/- {cv_scores.std():.3f})")
        
        # Select best model
        self.best_model_name = max(results.keys(), key=lambda k: results[k]['r2'])
        self.best_model = results[self.best_model_name]['model']
        
        print("\n" + "=" * 70)
        print(f"BEST MODEL: {self.best_model_name}")
        print("=" * 70)
        
        return results, X_test, y_test, X_test_scaled
    
    def visualize_results(self, results, X_test, y_test):
        """Create comprehensive visualizations using matplotlib"""
        # Set style
        plt.style.use('default')
        
        # Create figure with custom layout
        fig = plt.figure(figsize=(16, 10))
        gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)
        
        # Define colors
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8']
        
        # 1. Model Comparison - Horizontal Bar Chart
        ax1 = fig.add_subplot(gs[0, :2])
        model_names = list(results.keys())
        r2_scores = [results[m]['r2'] for m in model_names]
        
        bars = ax1.barh(model_names, r2_scores, color=colors)
        ax1.set_xlabel('R² Score', fontsize=12, fontweight='bold')
        ax1.set_title('Model Performance Comparison', fontsize=14, fontweight='bold', pad=20)
        ax1.set_xlim([0, 1])
        ax1.grid(axis='x', alpha=0.3, linestyle='--')
        
        # Add value labels on bars
        for i, (bar, score) in enumerate(zip(bars, r2_scores)):
            ax1.text(score + 0.01, bar.get_y() + bar.get_height()/2, 
                    f'{score:.3f}', va='center', fontsize=10, fontweight='bold')
        
        # 2. Metrics Comparison Table
        ax2 = fig.add_subplot(gs[0, 2])
        ax2.axis('off')
        
        # Create table data
        table_data = [['Model', 'RMSE', 'MAE', 'R²']]
        for name in model_names[:3]:  # Top 3 models
            table_data.append([
                name.split()[0],  # Short name
                f"{results[name]['rmse']:.2f}",
                f"{results[name]['mae']:.2f}",
                f"{results[name]['r2']:.3f}"
            ])
        
        table = ax2.table(cellText=table_data, cellLoc='center', loc='center',
                         colWidths=[0.4, 0.2, 0.2, 0.2])
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 2)
        
        # Style header
        for i in range(4):
            table[(0, i)].set_facecolor('#4ECDC4')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        # Alternate row colors
        for i in range(1, len(table_data)):
            for j in range(4):
                if i % 2 == 0:
                    table[(i, j)].set_facecolor('#f0f0f0')
        
        ax2.set_title('Top 3 Models Summary', fontsize=12, fontweight='bold', pad=20)
        
        # 3. Best Model: Actual vs Predicted
        ax3 = fig.add_subplot(gs[1, 0])
        best_pred = results[self.best_model_name]['predictions']
        
        ax3.scatter(y_test, best_pred, alpha=0.6, s=50, color='#FF6B6B', edgecolors='black', linewidth=0.5)
        
        # Perfect prediction line
        min_val, max_val = y_test.min(), y_test.max()
        ax3.plot([min_val, max_val], [min_val, max_val], 'k--', lw=2, label='Perfect Prediction')
        
        ax3.set_xlabel('Actual Temperature (°C)', fontsize=11, fontweight='bold')
        ax3.set_ylabel('Predicted Temperature (°C)', fontsize=11, fontweight='bold')
        ax3.set_title(f'{self.best_model_name}\nActual vs Predicted', fontsize=12, fontweight='bold')
        ax3.legend(loc='upper left', framealpha=0.9)
        ax3.grid(True, alpha=0.3, linestyle='--')
        
        # Add R² annotation
        r2_best = results[self.best_model_name]['r2']
        ax3.text(0.05, 0.95, f'R² = {r2_best:.3f}', transform=ax3.transAxes,
                fontsize=11, verticalalignment='top', bbox=dict(boxstyle='round', 
                facecolor='wheat', alpha=0.8))
        
        # 4. Residuals Plot
        ax4 = fig.add_subplot(gs[1, 1])
        residuals = y_test.values - best_pred
        
        ax4.scatter(best_pred, residuals, alpha=0.6, s=50, color='#4ECDC4', edgecolors='black', linewidth=0.5)
        ax4.axhline(y=0, color='red', linestyle='--', lw=2, label='Zero Error')
        ax4.set_xlabel('Predicted Temperature (°C)', fontsize=11, fontweight='bold')
        ax4.set_ylabel('Residuals (°C)', fontsize=11, fontweight='bold')
        ax4.set_title('Residuals Analysis', fontsize=12, fontweight='bold')
        ax4.legend(loc='upper right', framealpha=0.9)
        ax4.grid(True, alpha=0.3, linestyle='--')
        
        # 5. Error Distribution
        ax5 = fig.add_subplot(gs[1, 2])
        n, bins, patches = ax5.hist(residuals, bins=30, color='#98D8C8', 
                                     alpha=0.7, edgecolor='black', linewidth=1.2)
        
        # Color gradient for histogram
        cm = plt.cm.viridis
        for i, patch in enumerate(patches):
            patch.set_facecolor(cm(i / len(patches)))
        
        ax5.axvline(x=0, color='red', linestyle='--', lw=2, label='Zero Error')
        ax5.set_xlabel('Prediction Error (°C)', fontsize=11, fontweight='bold')
        ax5.set_ylabel('Frequency', fontsize=11, fontweight='bold')
        ax5.set_title('Error Distribution', fontsize=12, fontweight='bold')
        ax5.legend(framealpha=0.9)
        ax5.grid(True, alpha=0.3, linestyle='--', axis='y')
        
        # Add mean and std to plot
        mean_error = residuals.mean()
        std_error = residuals.std()
        ax5.text(0.05, 0.95, f'Mean: {mean_error:.3f}°C\nStd: {std_error:.3f}°C', 
                transform=ax5.transAxes, fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        # Overall title
        fig.suptitle('Weather Prediction ML Model - Comprehensive Analysis', 
                    fontsize=16, fontweight='bold', y=0.98)
        
        plt.savefig('weather_prediction_results.png', dpi=300, bbox_inches='tight')
        print("\n✓ Visualization saved as 'weather_prediction_results.png'")
        plt.show()
    
    def predict_new_weather(self, humidity, wind_speed, pressure, cloud_cover):
        """Make prediction for new weather data"""
        # Create feature vector with interactions
        features = np.array([[
            humidity, wind_speed, pressure, cloud_cover,
            humidity * wind_speed,
            pressure * cloud_cover
        ]])
        
        # Scale and predict
        features_scaled = self.scaler.transform(features)
        prediction = self.best_model.predict(features_scaled)[0]
        
        return prediction

def main():
    """Main execution function"""
    print("\n🌤️  WEATHER PREDICTION ML PROJECT 🌤️\n")
    
    # Initialize predictor
    predictor = WeatherPredictor()
    
    # Generate data
    print("📊 Generating synthetic weather data...")
    df = predictor.generate_synthetic_data(n_samples=1000)
    print(f"   Dataset shape: {df.shape}")
    print(f"\nFirst few rows:\n{df.head()}\n")
    
    # Preprocess
    print("⚙️  Preprocessing and feature engineering...")
    X, y = predictor.preprocess_data(df)
    print(f"   Features: {list(X.columns)}\n")
    
    # Train and evaluate
    print("🤖 Training models...")
    results, X_test, y_test, X_test_scaled = predictor.train_and_evaluate(X, y)
    
    # Visualize
    print("\n📈 Creating visualizations...")
    predictor.visualize_results(results, X_test, y_test)
    
    # Demo prediction
    print("\n🔮 DEMO PREDICTION:")
    print("-" * 70)
    test_humidity = 65
    test_wind = 15
    test_pressure = 1013
    test_cloud = 40
    
    predicted_temp = predictor.predict_new_weather(
        test_humidity, test_wind, test_pressure, test_cloud
    )
    
    print(f"Input Weather Conditions:")
    print(f"  • Humidity: {test_humidity}%")
    print(f"  • Wind Speed: {test_wind} km/h")
    print(f"  • Pressure: {test_pressure} hPa")
    print(f"  • Cloud Cover: {test_cloud}%")
    print(f"\n→ Predicted Temperature: {predicted_temp:.2f}°C")
    print("-" * 70)
    
    print("\n✅ Project completed successfully!")
    print("📁 Output: weather_prediction_results.png")

if __name__ == "__main__":
    main()