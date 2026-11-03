
"""Customer Travel Churn Prediction - Gradio application."""

from pathlib import Path

import gradio as gr
import spaces
import joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model.pkl"
DATA_PATH = BASE_DIR / "Customertravel.csv"
TARGET_COL = "Target"

# Load the dataset and trained model.
df = pd.read_csv(DATA_PATH, sep="\t")
model = joblib.load(MODEL_PATH)

if TARGET_COL not in df.columns:
    raise ValueError(
        f"Target column '{TARGET_COL}' was not found."
    )

FEATURE_COLS = [column for column in df.columns if column != TARGET_COL]

# Re-create encoders using the dataset categories.
encoders = {}

for column in FEATURE_COLS:
    if not pd.api.types.is_numeric_dtype(df[column]):
        encoder = LabelEncoder()
        encoder.fit(df[column].dropna().astype(str))
        encoders[column] = encoder

# Check the model's expected feature order.
model_features = getattr(model, "feature_names_in_", None)

if model_features is not None and list(model_features) != FEATURE_COLS:
    raise ValueError(
        f"Feature mismatch. Dataset: {FEATURE_COLS}; "
        f"Model: {list(model_features)}"
    )


# Decorate the prediction function for ZeroGPU.
@spaces.GPU
def predict_churn(
    age,
    frequent_flyer,
    annual_income_class,
    services_opted,
    account_synced_to_social_media,
    booked_hotel_or_not,
):
    age = int(round(float(age)))
    services_opted = int(round(float(services_opted)))

    user_values = {
        "Age": age,
        "FrequentFlyer": frequent_flyer,
        "AnnualIncomeClass": annual_income_class,
        "ServicesOpted": services_opted,
        "AccountSyncedToSocialMedia": account_synced_to_social_media,
        "BookedHotelOrNot": booked_hotel_or_not,
    }

    input_df = pd.DataFrame(
        [user_values],
        columns=FEATURE_COLS,
    )

    # Encode categorical inputs.
    for column, encoder in encoders.items():
        input_df[column] = encoder.transform(
            input_df[column].astype(str)
        )

    # Ensure numeric inputs remain numeric.
    for column in FEATURE_COLS:
        if column not in encoders:
            input_df[column] = pd.to_numeric(
                input_df[column],
                errors="raise",
            )

    prediction = int(model.predict(input_df)[0])

    probability_dict = {}
    churn_probability = None
    stay_probability = None

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(input_df)[0]
        classes = [int(value) for value in model.classes_]
        probability_by_class = dict(zip(classes, probabilities))

        stay_probability = float(
            probability_by_class.get(0, 0.0)
        )
        churn_probability = float(
            probability_by_class.get(1, 0.0)
        )

        probability_dict = {
            "Stay": stay_probability,
            "Churn": churn_probability,
        }

    if prediction == 1:
        result = "## ⚠️ This customer is likely to CHURN."
    else:
        result = "## ✅ This customer is likely to STAY."

    if churn_probability is not None:
        result += (
            f"\n\n**Churn probability:** "
            f"{churn_probability * 100:.2f}%"
            f"\n\n**Retention probability:** "
            f"{stay_probability * 100:.2f}%"
        )
    else:
        result += "\n\nThe model does not provide probability scores."

    summary = pd.DataFrame({
        "Customer Feature": FEATURE_COLS,
        "Selected Value": [
            user_values[column] for column in FEATURE_COLS
        ],
    })

    return result, probability_dict, summary


# Set slider ranges from the dataset.
age_min = int(pd.to_numeric(df["Age"]).min())
age_max = int(pd.to_numeric(df["Age"]).max())
age_default = int(round(pd.to_numeric(df["Age"]).median()))

services_min = int(pd.to_numeric(df["ServicesOpted"]).min())
services_max = int(pd.to_numeric(df["ServicesOpted"]).max())
services_default = int(
    round(pd.to_numeric(df["ServicesOpted"]).median())
)


def category_choices(column):
    return [str(value) for value in encoders[column].classes_]


# Build the Gradio interface.
with gr.Blocks(title="Customer Travel Churn Prediction") as demo:

    gr.Markdown(
        """
        # ✈️ Customer Travel Churn Prediction

        Enter customer details and click **Predict Churn** to estimate
        whether the customer is likely to stay or leave.
        """
    )

    with gr.Row():

        with gr.Column():
            gr.Markdown("### 📋 Customer Details")

            age_input = gr.Slider(
                minimum=age_min,
                maximum=age_max,
                value=age_default,
                step=1,
                label="Age",
            )

            frequent_flyer_input = gr.Dropdown(
                choices=category_choices("FrequentFlyer"),
                value=category_choices("FrequentFlyer")[0],
                label="Frequent Flyer",
            )

            annual_income_input = gr.Dropdown(
                choices=category_choices("AnnualIncomeClass"),
                value=category_choices("AnnualIncomeClass")[0],
                label="Annual Income Class",
            )

            services_input = gr.Slider(
                minimum=services_min,
                maximum=services_max,
                value=services_default,
                step=1,
                label="Services Opted",
            )

            social_media_input = gr.Dropdown(
                choices=category_choices(
                    "AccountSyncedToSocialMedia"
                ),
                value=category_choices(
                    "AccountSyncedToSocialMedia"
                )[0],
                label="Account Synced to Social Media",
            )

            hotel_input = gr.Dropdown(
                choices=category_choices("BookedHotelOrNot"),
                value=category_choices("BookedHotelOrNot")[0],
                label="Booked Hotel or Not",
            )

            predict_button = gr.Button(
                "🚀 Predict Churn",
                variant="primary",
            )

        with gr.Column():
            gr.Markdown("### 🔍 Prediction Result")

            result_output = gr.Markdown(
                "Your prediction will appear here."
            )

            probability_output = gr.Label(
                label="Prediction Confidence"
            )

            summary_output = gr.Dataframe(
                headers=["Customer Feature", "Selected Value"],
                label="📊 Input Summary",
                interactive=False,
                wrap=True,
            )

    predict_button.click(
        fn=predict_churn,
        inputs=[
            age_input,
            frequent_flyer_input,
            annual_income_input,
            services_input,
            social_media_input,
            hotel_input,
        ],
        outputs=[
            result_output,
            probability_output,
            summary_output,
        ],
    )

    with gr.Accordion("📂 View Dataset Sample", open=False):
        gr.Dataframe(
            value=df.head(20),
            interactive=False,
            wrap=True,
        )

    with gr.Accordion("📈 Churn Distribution", open=False):
        distribution = (
            df[TARGET_COL]
            .value_counts()
            .reindex([0, 1], fill_value=0)
            .rename(index={0: "Stay", 1: "Churn"})
            .rename_axis("Customer Status")
            .reset_index(name="Number of Customers")
        )

        gr.BarPlot(
            value=distribution,
            x="Customer Status",
            y="Number of Customers",
            title="Customer Churn Distribution",
        )

    gr.Markdown(
        "---\nBuilt with Python, Scikit-learn and Gradio | "
        "Model: Random Forest"
    )


if __name__ == "__main__":
    demo.launch(ssr_mode=False)
