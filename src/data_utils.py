from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "Pakistan_Ecommerce_sales_dataset.xlsx"

def load_data(path=DATA_PATH):
    df = pd.read_excel(path)
    df["Order_Date"] = pd.to_datetime(df["Order_Date"], errors="coerce")
    return df

def prepare_analysis_data(df, delivered_only=True):
    """Use delivered orders for sales/revenue analysis by default."""
    data = df.copy()
    if delivered_only and "Order_Status" in data.columns:
        data = data[data["Order_Status"].str.lower() == "delivered"].copy()
    return data
