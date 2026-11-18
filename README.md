# Pakistan E-Commerce Sales Analysis

Data analysis + machine learning project (SBBU, Introduction to Data Science).
It answers five business questions from a Pakistani online-store dataset and predicts the value of an order.

## Questions answered
1. Which product has the highest demand?
2. Which city places the most orders?
3. What is the monthly revenue?
4. Which payment methods do customers use?
5. In which month are sales highest?

## Dataset
data/Pakistan_Ecommerce_sales_dataset.xlsx - 11,505 orders, 22 columns, Jan 2023 - Sep 2026.

## Project steps
- src/preprocess.py - load data, understand it, pre-processing
- src/eda.py - EDA charts for the 5 questions
- src/train_model.py - feature selection, train/test split, model selection, evaluation
- app.py - Streamlit web app

## Tools
Python, pandas, NumPy, Matplotlib, Seaborn, scikit-learn, Streamlit, Plotly

## Run
pip install -r requirements.txt
python src/preprocess.py
python src/eda.py
python src/train_model.py
streamlit run app.py

## Results
- Revenue is counted from delivered orders only.
- Linear Regression gave the best test R2 (0.961) for order-value prediction.
