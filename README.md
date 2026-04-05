🚀 Smart Price Predictor

An advanced Machine Learning solution designed to accurately predict electronics product prices using data-driven intelligence and scalable modeling techniques. This project demonstrates a complete ML pipeline with high-performance outcomes and real-world applicability.

📌 Project Overview

Smart Price Predictor leverages a dataset of 14,000+ Amazon electronics products to forecast discounted prices with exceptional accuracy.

It integrates:

Data preprocessing & cleaning
Feature engineering
Exploratory Data Analysis (EDA)
Model development & evaluation
Interactive prediction system
🧠 Models Implemented
Linear Regression — Baseline analytical model
Decision Tree Regressor — Rule-based prediction
Random Forest Regressor — Ensemble learning (Top Performer)

⭐ Achieved ~99.5% R² Accuracy with Random Forest



📂 Dataset
Domain: Amazon Electronics
Size: ~14,592 records
Key Attributes:
Actual Price (MRP)
Discounted Price (Target Variable)
Discount Percentage
Product Category
Brand & Availability
Engineered Features (Ratings, Reviews)


⚙️ Core Capabilities

🔹 Data Engineering
Intelligent column mapping from raw dataset
Category classification via keyword-based logic
Synthetic feature generation for missing attributes
🔹 Feature Engineering
Log transformation for price normalization
Price-to-discount ratio
Encoded categorical variables
🔹 Data Optimization
Outlier removal (1st–99th percentile filtering)
Missing value handling
Data validation for consistency


📊 Exploratory Data Analysis

Includes 13+ insightful visualizations, such as:

Category-wise product distribution
Price distribution trends
Discount vs price relationships
Correlation heatmaps
Feature importance ranking
Residual analysis


📈 Model Evaluation Metrics
R² Score (Accuracy)
Mean Absolute Error (MAE)
Root Mean Squared Error (RMSE)
Mean Absolute Percentage Error (MAPE)


🏆 Performance Summary
Model	Performance Level
Linear Regression	Moderate
Decision Tree	High
Random Forest	⭐ Excellent (~99.5%)

🖥️ Interactive Prediction System

A command-line interface enables users to:

Select product category
Input pricing and rating details
Generate predictions from all models
Compare outputs in real-time


🛠️ Tech Stack
Python
NumPy, Pandas
Matplotlib, Seaborn
Scikit-learn

📁 Project Structure
├── amazon_price_prediction.py
├── projdata.gz
├── graphs/
│   ├── graph1_products_per_category.png
│   ├── graph2_price_distribution.png
│   ├── ...
│   └── graph13_actual_vs_predicted_all.png
├── README.md


▶️ Execution Guide
pip install numpy pandas matplotlib seaborn scikit-learn
python amazon_price_prediction.py
💡 Key Insights
Pricing is strongly influenced by MRP and discount percentage
Engineered features significantly enhance predictive accuracy
Ensemble models outperform traditional regression techniques
🔮 Future Roadmap
Web deployment (Flask / FastAPI)
Real-time data integration
Advanced ML/DL model enhancements
Pricing recommendation engine


👤 Author

Sahil Sharma
B.Tech Computer Science

📜 License

This project is intended for academic and demonstration purposes.
