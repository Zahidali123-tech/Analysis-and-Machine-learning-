from pathlib import Path
import sys
import json
import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from data_utils import load_data, prepare_analysis_data
from train_model import train_and_save, make_features, FEATURES

st.set_page_config(page_title="Pakistan E-Commerce Sales Analysis", page_icon="🛒", layout="wide")
st.title("🇵🇰 Pakistan E-Commerce Sales Analysis")
st.caption("Sales dashboard, exploratory analysis and machine-learning sales estimator")

@st.cache_data
def get_data():
    return load_data()

df = get_data()
st.sidebar.header("Dashboard Filters")
cities = ["All"] + sorted(df["City"].dropna().unique().tolist())
selected_city = st.sidebar.selectbox("City", cities)
statuses = ["Delivered", "All statuses"] + sorted(df["Order_Status"].dropna().unique().tolist())
selected_status = st.sidebar.selectbox("Order status", statuses, index=0)
date_min, date_max = df["Order_Date"].min().date(), df["Order_Date"].max().date()
date_range = st.sidebar.date_input("Order date range", value=(date_min, date_max), min_value=date_min, max_value=date_max)

filtered = df.copy()
if selected_city != "All":
    filtered = filtered[filtered["City"] == selected_city]
if selected_status == "Delivered":
    filtered = filtered[filtered["Order_Status"] == "Delivered"]
elif selected_status != "All statuses":
    filtered = filtered[filtered["Order_Status"] == selected_status]
if isinstance(date_range, (tuple, list)) and len(date_range) == 2:
    filtered = filtered[filtered["Order_Date"].dt.date.between(date_range[0], date_range[1])]

tab1, tab2, tab3, tab4 = st.tabs(["Dashboard", "Exploratory Data Analysis", "ML Model", "Dataset & Methodology"])

with tab1:
    st.subheader("Business KPIs")
    delivered = filtered[filtered["Order_Status"] == "Delivered"]
    total_orders = filtered["Order_ID"].nunique()
    revenue = delivered["Sales"].sum()
    profit = delivered["Profit"].sum()
    quantity = delivered["Quantity"].sum()
    a,b,c,d = st.columns(4)
    a.metric("Orders in selection", f"{total_orders:,}")
    b.metric("Delivered revenue", f"PKR {revenue:,.0f}")
    c.metric("Delivered profit", f"PKR {profit:,.0f}")
    d.metric("Units sold (delivered)", f"{quantity:,}")

    left, right = st.columns(2)
    with left:
        st.markdown("#### Top products by units sold")
        prod = delivered.groupby("Product_Name")["Quantity"].sum().sort_values(ascending=False).head(10)
        st.bar_chart(prod)
    with right:
        st.markdown("#### Orders by city")
        city_orders = filtered.groupby("City")["Order_ID"].nunique().sort_values(ascending=False).head(10)
        st.bar_chart(city_orders)

    monthly = delivered.assign(Month=delivered["Order_Date"].dt.to_period("M").astype(str)).groupby("Month")["Sales"].sum()
    st.markdown("#### Monthly delivered revenue")
    st.line_chart(monthly)
    st.markdown("#### Payment method usage")
    st.bar_chart(filtered["Payment_Method"].value_counts())

with tab2:
    st.subheader("Explore the selected data")
    c1,c2 = st.columns(2)
    with c1:
        st.markdown("#### Revenue by product category")
        st.bar_chart(delivered.groupby("Category")["Sales"].sum().sort_values(ascending=False))
        st.markdown("#### Order status distribution")
        st.bar_chart(filtered["Order_Status"].value_counts())
    with c2:
        st.markdown("#### Payment method order count")
        st.bar_chart(filtered["Payment_Method"].value_counts())
        st.markdown("#### Sales by customer type")
        st.bar_chart(delivered.groupby("Customer_Type")["Sales"].sum())
    st.markdown("#### Numeric summary")
    st.dataframe(filtered[["Customer_Age","Quantity","Unit_Price","Sales","Discount_Percentage","Profit"]].describe().round(2), use_container_width=True)

with tab3:
    st.subheader("Machine Learning: Predict order sales")
    model_path = ROOT / "models" / "sales_model.joblib"
    info_path = ROOT / "models" / "model_info.json"
    metrics_path = ROOT / "models" / "model_metrics.csv"
    if not model_path.exists():
        with st.spinner("Training the models for the first time..."):
            model, metrics, best = train_and_save()
    else:
        model = joblib.load(model_path)
    if metrics_path.exists():
        metrics = pd.read_csv(metrics_path)
        st.markdown("#### Model evaluation on held-out test data")
        st.dataframe(metrics.round(3), use_container_width=True)
    if info_path.exists():
        info = json.loads(info_path.read_text(encoding="utf-8"))
        st.success(f"Selected model: {info['best_model']} | Test rows: {info['test_rows']:,}")
    st.info("Prediction uses order/customer/product details. It intentionally excludes Sales-derived fields such as Profit, Cost and Discount_Amount.")
    with st.form("prediction_form"):
        col1,col2,col3 = st.columns(3)
        age = col1.number_input("Customer age", 18, 90, 30)
        qty = col2.number_input("Quantity", 1, 100, 2)
        unit_price = col3.number_input("Unit price (PKR)", 1, 1000000, 5000)
        discount = col1.slider("Discount percentage", 0, 80, 5)
        gender = col2.selectbox("Gender", sorted(df["Gender"].dropna().unique()))
        city = col3.selectbox("City", sorted(df["City"].dropna().unique()))
        region = col1.selectbox("Region", sorted(df["Region"].dropna().unique()))
        category = col2.selectbox("Category", sorted(df["Category"].dropna().unique()))
        subcats = sorted(df.loc[df["Category"] == category, "Sub_Category"].dropna().unique())
        subcategory = col3.selectbox("Sub-category", subcats)
        payment = col1.selectbox("Payment method", sorted(df["Payment_Method"].dropna().unique()))
        customer_type = col2.selectbox("Customer type", sorted(df["Customer_Type"].dropna().unique()))
        order_date = col3.date_input("Order date", value=pd.Timestamp.today().date())
        submit = st.form_submit_button("Predict sales")
    if submit:
        row = pd.DataFrame([{
            "Customer_Age": age, "Quantity": qty, "Unit_Price": unit_price,
            "Discount_Percentage": discount, "Order_Month": order_date.month,
            "Order_Day": order_date.day, "Order_Year": order_date.year,
            "Gender": gender, "City": city, "Region": region, "Category": category,
            "Sub_Category": subcategory, "Payment_Method": payment, "Customer_Type": customer_type
        }])[FEATURES]
        prediction = float(model.predict(row)[0])
        st.success(f"Estimated sales amount: PKR {prediction:,.2f}")
        st.caption("This is a model estimate, not a guaranteed selling price or actual transaction.")

with tab4:
    st.subheader("Dataset overview")
    st.write(f"Rows: **{len(df):,}** | Columns: **{len(df.columns)}**")
    st.write(f"Date coverage: **{df['Order_Date'].min().date()}** to **{df['Order_Date'].max().date()}**")
    st.markdown("#### Column names and data types")
    st.dataframe(pd.DataFrame({"Column":df.columns, "Data type":df.dtypes.astype(str).values}), use_container_width=True)
    st.markdown("#### Data quality checks")
    q1,q2,q3 = st.columns(3)
    q1.metric("Missing cells", f"{int(df.isna().sum().sum()):,}")
    q2.metric("Duplicate rows", f"{int(df.duplicated().sum()):,}")
    q3.metric("Unique orders", f"{df['Order_ID'].nunique():,}")
    st.markdown("""
    **Project workflow**
    1. Load data from Excel.
    2. Understand data: shape, columns, data types, date range and descriptive statistics.
    3. Preprocess: parse dates, check missing values/duplicates and create month/day/year features.
    4. EDA: product demand, city orders, monthly revenue and payment-method usage.
    5. Feature selection: use business-relevant input fields; exclude identifiers and target-derived fields to reduce leakage.
    6. Train/test split: 80% training and 20% testing (`random_state=42`).
    7. Model selection: compare Ridge Regression, Random Forest and Extra Trees.
    8. Evaluation: MAE, RMSE and R²; select the lowest-MAE model.
    9. Web app: Streamlit dashboard and sales prediction form.

    **Metric definition:** revenue and product demand in the dashboard default to delivered orders. Order counts can be viewed with the status filter. The dataset's currency is treated as PKR based on this project's Pakistan context; confirm with the dataset owner if the source uses another currency.
    """)
    st.markdown("#### Preview")
    st.dataframe(filtered.head(100), use_container_width=True)
    st.download_button("Download filtered data as CSV", filtered.to_csv(index=False).encode("utf-8"), "filtered_ecommerce_data.csv", "text/csv")
