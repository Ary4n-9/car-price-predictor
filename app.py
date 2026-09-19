
import joblib
import numpy as np
import pandas as pd
import gradio as gr
from sklearn.preprocessing import MinMaxScaler

MODEL_PATH = "car_price_predictor.pkl"
DATA_PATH = "car data.csv"
REFERENCE_YEAR = 2026

model = joblib.load(MODEL_PATH)
cars = pd.read_csv(DATA_PATH)

cars["Age"] = REFERENCE_YEAR - cars["Year"]
cars = cars.rename(
    columns={
        "Selling_Price": "Selling_Price(lacs)",
        "Present_Price": "Present_Price(lacs)",
        "Owner": "Past_Owners"
    }
)

recommendation_features = [
    "Selling_Price(lacs)",
    "Present_Price(lacs)",
    "Kms_Driven",
    "Past_Owners",
    "Age"
]


def recommend_cars(
    predicted_price,
    present_price,
    kms_driven,
    fuel_type,
    seller_type,
    transmission,
    past_owners,
    age,
    top_n=3
):
    rec = cars.copy()

    numeric_input = np.array(
        [predicted_price, present_price, kms_driven, past_owners, age],
        dtype=float
    )

    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(rec[recommendation_features])
    scaled_input = scaler.transform([numeric_input])[0]

    numeric_distance = np.sqrt(
        np.sum((scaled_data - scaled_input) ** 2, axis=1)
    )

    categorical_penalty = (
        (rec["Fuel_Type"] != fuel_type).astype(float) * 0.60
        + (rec["Seller_Type"] != seller_type).astype(float) * 0.40
        + (rec["Transmission"] != transmission).astype(float) * 0.40
    )

    rec["match_distance"] = numeric_distance + categorical_penalty

    return (
        rec.sort_values("match_distance")
        .drop_duplicates(subset=["Car_Name"])
        .head(top_n)
        .copy()
    )


def predict_car(
    present_price,
    kms_driven,
    fuel_type,
    seller_type,
    transmission,
    past_owners,
    age
):
    try:
        input_df = pd.DataFrame([{
            "Present_Price(lacs)": float(present_price),
            "Kms_Driven": float(kms_driven),
            "Fuel_Type": fuel_type,
            "Seller_Type": seller_type,
            "Transmission": transmission,
            "Past_Owners": int(past_owners),
            "Age": int(age)
        }])

        predicted_price = max(0.0, float(model.predict(input_df)[0]))

        top = recommend_cars(
            predicted_price,
            float(present_price),
            float(kms_driven),
            fuel_type,
            seller_type,
            transmission,
            int(past_owners),
            int(age)
        )

        rows = []
        for rank, (_, car) in enumerate(top.iterrows(), 1):
            rows.append({
                "Rank": rank,
                "Recommended Car": car["Car_Name"],
                "Dataset Price (Lakhs)": round(float(car["Selling_Price(lacs)"]), 2),
                "Fuel": car["Fuel_Type"],
                "Transmission": car["Transmission"],
                "Kilometres": int(car["Kms_Driven"]),
                "Age": int(car["Age"])
            })

        result_df = pd.DataFrame(rows)

        recommendation_text = "\n".join(
            f"**{r['Rank']}. {r['Recommended Car']}** — ₹{r['Dataset Price (Lakhs)']:.2f} L"
            for _, r in result_df.iterrows()
        )

        return (
            f"## 💰 Estimated Selling Price\n### ₹{predicted_price:.2f} Lakhs",
            recommendation_text or "No matching cars found.",
            result_df
        )

    except Exception as e:
        return f"### ❌ Error\n`{e}`", "Please check your inputs.", pd.DataFrame()


def clear_form():
    return 8.0, 35000, "Petrol", "Dealer", "Manual", 0, 5, "", "", pd.DataFrame()


custom_css = """
.gradio-container {
    max-width: 1100px !important;
    margin: auto !important;
}
"""

with gr.Blocks(
    title="Smart Car Price & Recommendation System",
    theme=gr.themes.Soft(),
    css=custom_css
) as demo:

    gr.Markdown("""
    # 🚗 Smart Car Price & Recommendation System
    ### Predict the estimated selling price and find matching cars
    """)

    gr.Markdown(
        "Enter the car details. The system predicts the price and returns the **Top 3 car names** from the dataset."
    )

    with gr.Row():
        with gr.Column():
            gr.Markdown("### 🔧 Car Details")

            present_price = gr.Number(
                label="Present Price (Lakhs)", value=8.0, minimum=0.1
            )
            kms_driven = gr.Number(
                label="Kilometres Driven", value=35000, minimum=0, precision=0
            )
            fuel_type = gr.Dropdown(
                ["Petrol", "Diesel", "CNG"], value="Petrol", label="Fuel Type"
            )
            seller_type = gr.Dropdown(
                ["Dealer", "Individual"], value="Dealer", label="Seller Type"
            )
            transmission = gr.Dropdown(
                ["Manual", "Automatic"], value="Manual", label="Transmission"
            )
            past_owners = gr.Number(
                label="Previous Owners", value=0, minimum=0, maximum=10, precision=0
            )
            age = gr.Number(
                label="Car Age (Years)", value=5, minimum=0, maximum=50, precision=0
            )

            with gr.Row():
                predict_btn = gr.Button("🚀 Predict & Recommend", variant="primary")
                clear_btn = gr.Button("↺ Clear")

        with gr.Column():
            gr.Markdown("### 📊 Prediction")
            price_output = gr.Markdown("Enter details and click **Predict & Recommend**.")

            gr.Markdown("### ⭐ Recommended Cars")
            recommendation_output = gr.Markdown("Your Top 3 matching cars will appear here.")

    gr.Markdown("### 📋 Recommendation Details")

    recommendation_table = gr.Dataframe(
        headers=[
            "Rank", "Recommended Car", "Dataset Price (Lakhs)",
            "Fuel", "Transmission", "Kilometres", "Age"
        ],
        interactive=False,
        wrap=True
    )

    gr.Markdown("""
    ---
    **Price model:** `car_price_predictor.pkl`  
    **Car-name recommendation source:** `car data.csv`
    """)

    predict_btn.click(
        predict_car,
        inputs=[
            present_price, kms_driven, fuel_type, seller_type,
            transmission, past_owners, age
        ],
        outputs=[price_output, recommendation_output, recommendation_table]
    )

    clear_btn.click(
        clear_form,
        outputs=[
            present_price, kms_driven, fuel_type, seller_type,
            transmission, past_owners, age,
            price_output, recommendation_output, recommendation_table
        ]
    )


if __name__ == "__main__":
    demo.launch()
