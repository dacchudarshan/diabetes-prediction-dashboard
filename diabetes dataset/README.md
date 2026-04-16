# 🩺 Diabetes Prediction Models Dashboard

A comprehensive machine learning dashboard that compares multiple predictive models for diabetes prediction with beautiful visualizations and real-time analytics.

---

## ✨ Features

- **3 ML Algorithms Comparison**
  - Logistic Regression (Best: 99.79% accuracy)
  - Polynomial Regression (99.61% R² score)
  - Multiple Linear Regression (92.50% R² score)

- **Advanced Visualizations**
  - Model performance comparison charts
  - Actual vs Predicted scatter plots
  - Performance metrics heatmap
  - Interactive responsive dashboard

- **Beautiful UI/UX**
  - Modern glassmorphic design
  - Smooth animations and transitions
  - Gradient backgrounds
  - Responsive layout (mobile & desktop)
  - Dark mode ready styles

- **Detailed Metrics**
  - Accuracy scores
  - Mean Squared Error (MSE)
  - Mean Absolute Error (MAE)
  - Cross-validation results
  - Model ranking system

---

## 📋 File Structure

```
diabetes dataset copy/
│
├── app.py                          # Main Python script for model training
├── diabetes_dataset.csv            # Dataset (100,000 samples)
├── dashboard.html                  # Interactive web dashboard
├── README.md                        # This file
│
├── model_comparison.png            # Generated: Model metrics comparison
├── predictions_comparison.png       # Generated: Actual vs Predicted plots
└── metrics_heatmap.png             # Generated: Performance heatmap
```

---

## 🚀 Quick Start

### Prerequisites
- Python 3.7+
- pip (Python package manager)

### Installation

1. **Clone or navigate to the project directory:**
   ```bash
   cd "/Users/darshu/Projects/ML/diabetes dataset copy"
   ```

2. **Install required packages:**
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn
   ```

### Run the Project

#### Step 1: Train Models and Generate Charts
```bash
python app.py
```

This will:
- Load the diabetes dataset (100,000 samples)
- Train 3 machine learning algorithms (Logistic Regression, Polynomial Regression, Multiple Linear Regression)
- Generate 3 performance visualization PNG files
- Display metrics in terminal

#### Step 2: View the Dashboard
Open the dashboard in your browser:

**Option A: Direct file open**
```bash
open dashboard.html
```

**Option B: Using local server (recommended)**
```bash
python -m http.server 8000
```
Then visit: `http://localhost:8000/dashboard.html`

---

## 📊 Model Information

### Dataset
- **Total Samples:** 100,000
- **Training Samples:** 70,000 (sampled for faster processing)
- **Test Split:** 20%
- **Features:** 31 medical/demographic attributes
- **Target Variables:** 
  - For Classification (Logistic Regression): Binary classification (0/1)
  - For Regression (Polynomial & Multiple Linear): Continuous diabetes values

### Models & Performance

| Model | Performance Metric | MSE | MAE | Status |
|-------|-------------------|-----|-----|--------|
| **Logistic Regression** | Accuracy: 99.79% | 0.0021 | 0.0021 | 🏆 Best |
| **Polynomial Regression** | R² Score: 99.61% | 0.0009 | 0.0086 | ⭐ Excellent |
| **Multiple Linear Regression** | R² Score: 92.50% | 0.0180 | 0.0832 | ✅ Very Good |

### Validation & Preprocessing
- **Cross-Validation:** 5-fold (for Logistic Regression)
- **Scaler:** StandardScaler (normalized features)
- **Encoding:** LabelEncoder for categorical variables
- **Polynomial Features:** Degree 2 (for Polynomial Regression)
- **Data Split:** Binary target for classification, continuous for regression

---

## 🎨 Dashboard Features

### Header Section
- Project title and description
- User profile badge (darshu)
- Animated gradient background

### Best Model Banner
- Highlights the top-performing model
- Shows key metrics (Accuracy, MSE, MAE)
- Animated hover effects

### Performance Charts Section
1. **Model Comparison** - Bar charts for accuracy, MSE, and MAE
2. **Predictions Comparison** - Scatter plots of actual vs predicted values
3. **Metrics Heatmap** - Color-coded performance matrix

### Detailed Metrics Table
- All 4 models with complete metrics
- Performance indicators (🏆, ⭐, ✅)
- Interactive hover animations

### Key Performance Indicators
- Best accuracy achieved
- Lowest error rates
- Average model performance
- Visual metric cards with animations

### Model Ranking
- Complete ranking list with detailed metrics
- Interactive hover states
- Emoji indicators for position

---

## 🛠️ Technologies Used

### Backend (Training)
- **Python 3.8+**
- **Pandas** - Data manipulation
- **NumPy** - Numerical computing
- **Scikit-learn** - Machine learning models
- **Matplotlib** - Data visualization
- **Seaborn** - Statistical visualization

### Frontend (Dashboard)
- **HTML5** - Structure
- **CSS3** - Styling with animations
- **Vanilla JavaScript** - Interactive elements (future enhancement)

---

## 📈 Animations & Features

### CSS Animations
- `fadeInUp` - Smooth fade-in with upward motion
- `slideInLeft/Right` - Directional slide animations
- `bounceIn` - Bouncy entry animation
- `float` - Floating heading effect
- `glow` - Pulsing glow effect
- `gradientShift` - Background gradient animation
- `pulse` - Metric values pulsing

### Interactive Effects
- Hover transformations on cards
- Backdrop blur effects
- Gradient text on metric values
- Animated borders and separators
- Staggered animations for list items

---

## 🎯 Usage Examples

### View Model Performance
1. Run `python app.py` to train models
2. Open `dashboard.html` in browser
3. Scroll through charts to see visualizations
4. Check the metrics table for detailed numbers
5. Review model ranking at the bottom

### Modify Models
Edit `app.py` to:
- Change sampling size: `df.sample(n=70000, ...)`
- Adjust polynomial degree: `PolynomialFeatures(degree=2)`
- Modify Logistic Regression parameters: `LogisticRegression(max_iter=1000)`
- Change cross-validation folds: `cv=5`
- Update feature scaling or encoding methods

### Customize Dashboard
Edit `dashboard.html` to:
- Update colors and gradients
- Modify animations timing
- Add/remove sections
- Change metric displays

---

## 📝 Requirements

Create a `requirements.txt` file:
```
pandas==1.5.0
numpy==1.23.0
matplotlib==3.6.0
seaborn==0.12.0
scikit-learn==1.1.0
```

Install all requirements:
```bash
pip install -r requirements.txt
```

---

## 🤝 Contributing

To extend this project:
1. Add more models in `app.py`
2. Include additional metrics
3. Enhance dashboard visualizations
4. Add JavaScript interactivity
5. Implement data preprocessing improvements

---

## 📞 Support

For issues or questions:
1. Check that all files are in the same directory
2. Verify Python version (3.7+)
3. Ensure all packages are installed
4. Check dataset file exists (`diabetes_dataset.csv`)

---

## 📄 License

This project is open source and available for educational and research purposes.

---

## 👤 Author

**darshu** - Machine Learning Enthusiast

**Date Created:** April 9, 2026

---

## 📚 Dataset Source

The diabetes dataset contains comprehensive medical records with:
- Age, Gender, Ethnicity
- Education Level
- Blood Glucose, HbA1c values
- Diabetes stage classification
- Risk scoring metrics

---

## 🎓 Model Explanations

### Random Forest (99.96% ✓ Best)
- Ensemble of decision trees
- Excellent for non-linear relationships
- Robust to outliers
- Great generalization

### Gradient Boosting (99.95%)
- Sequential tree building
- Minimizes errors iteratively
- Strong performance on complex patterns
- Slightly slower training

### SVM (99.84%)
- Boundary-based classification
- Effective in high dimensions
- Requires feature scaling
- Good for binary classification

### Logistic Regression (99.79%)
- Linear probability model
- Interpretable results
- Fast training
- Good baseline model

---

## 🔍 Next Steps

To further improve:
1. Try ensemble voting classifiers
2. Implement hyperparameter tuning
3. Add feature importance analysis
4. Create prediction interface
5. Add real-time model updates
6. Implement user authentication

---

## ✅ Checklist

- [x] Models trained and evaluated
- [x] Visualizations generated
- [x] Dashboard created
- [x] Animations implemented
- [x] Responsive design added
- [x] User profile badge added
- [x] README documentation

---

**Happy Analyzing! 🚀**
