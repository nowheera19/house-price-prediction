
import streamlit as st
import pandas as pd
import pickle
from pathlib import Path
from sklearn.linear_model import LinearRegression

st.set_page_config(
    page_title="House Price Predictor",
    page_icon="🏡",
    layout="centered"
)

st.title("🏡 House Price Prediction")
st.write("Predict an estimated house price using Machine Learning.")

@st.cache_resource
def load_model():
    path = Path(__file__).parent / "house_price_model.pkl"

    if path.exists():
        with open(path, "rb") as f:
            return pickle.load(f)

    # Fallback model, so the webpage works without the downloaded file
    data = pd.DataFrame({
        "area_sqft": [600, 800, 1000, 1200, 1500,
                      1800, 2000, 2200, 2500, 3000],
        "bedrooms": [1, 2, 2, 2, 3, 3, 4, 4, 4, 5],
        "age_years": [15, 10, 8, 5, 9, 4, 6, 3, 2, 1],
        "price_lakhs": [25, 32, 40, 48, 60,
                        75, 82, 95, 110, 140]
    })

    model = LinearRegression()
    model.fit(
        data[["area_sqft", "bedrooms", "age_years"]],
        data["price_lakhs"]
    )
    return model


with st.form("prediction_form"):
    area = st.number_input(
        "Area (square feet)",
        min_value=100,
        max_value=20000,
        value=1500
    )

    bedrooms = st.number_input(
        "Number of bedrooms",
        min_value=1,
        max_value=20,
        value=3
    )

    age = st.number_input(
        "House age (years)",
        min_value=0,
        max_value=200,
        value=5
    )

    submit = st.form_submit_button(
        "Predict House Price",
        use_container_width=True
    )

if submit:
    model = load_model()

    house = pd.DataFrame([{
        "area_sqft": area,
        "bedrooms": bedrooms,
        "age_years": age
    }])

    price = model.predict(house)[0]

    st.success(f"Estimated Price: ₹{price:,.2f} lakhs")
    st.caption(
        "Demo prediction only. Actual prices depend on location "
        "and other property factors."
    )
