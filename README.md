# ✈️ Customer Travel Churn Prediction

An end-to-end machine-learning project that predicts whether a travel customer is likely to **stay** or **churn** using customer and service-related information. It includes a training and evaluation notebook, a saved Random Forest model, a Gradio web interface and a Hugging Face Spaces deployment configuration.

<p align="center">
  <strong>Python · Machine Learning · Random Forest · Gradio · scikit-learn</strong>
</p>

## 🌐 Live demo

**Hugging Face Space:** https://huggingface.co/spaces/neelverse713/customer-churn

If the Space is private or unavailable, access may be restricted to authorized Hugging Face users.

## 📌 Project overview

Customer churn occurs when a customer stops using a company's products or services. Identifying customers who may be at risk can help an organization investigate customer experience and plan retention activities.

This project uses a Random Forest classification model trained on travel-customer attributes. A user enters six feature values in the Gradio interface and receives a predicted class:

- **0 — Stay**
- **1 — Churn**

When supported by the saved estimator, the interface also displays class probability estimates.

> This is an educational machine-learning project. Predictions are estimates, not guarantees or business decisions.

## ✨ Features

- Interactive Gradio interface for entering customer information.
- Random Forest classification using a pre-trained model.
- Categorical feature encoding for the model input.
- Stay/churn prediction.
- Probability display when `predict_proba` is available.
- Input summary after prediction.
- Dataset preview.
- Churn-class distribution chart.
- Jupyter notebook covering data inspection, preprocessing, training and evaluation.
- Hugging Face Spaces configuration for deployment.

## 🧰 Tech stack

| Technology | Purpose |
|---|---|
| Python | Application and model code |
| pandas | Dataset loading and tabular data handling |
| scikit-learn | Label encoding, Random Forest and evaluation |
| Joblib | Saving and loading the trained model |
| Gradio | Interactive web application |
| Jupyter Notebook | Exploratory analysis, training and evaluation |
| Hugging Face Spaces | Hosting the interactive demo |

## 📊 Dataset

The supplied `Customertravel.csv` file is tab-separated and contains **954 rows and 7 columns**.

| Feature | Description | Values / range in supplied data |
|---|---|---|
| `Age` | Customer age | 27–38 |
| `FrequentFlyer` | Frequent-flyer status | `Yes`, `No`, `No Record` |
| `AnnualIncomeClass` | Income category | `Low Income`, `Middle Income`, `High Income` |
| `ServicesOpted` | Number of services opted | 1–6 |
| `AccountSyncedToSocialMedia` | Whether account is synced to social media | `Yes`, `No` |
| `BookedHotelOrNot` | Whether a hotel was booked | `Yes`, `No` |
| `Target` | Prediction label | `0` = Stay, `1` = Churn |

The target column is excluded from the prediction inputs.

### Dataset loading

The CSV is tab-separated, so it must be loaded using:

```python
import pandas as pd

df = pd.read_csv("Customertravel.csv", sep="\t")
```

## 🧪 Machine-learning workflow

The notebook `customer_churn.ipynb` documents this workflow:

1. **Load data** — read the tab-separated dataset.
2. **Explore data** — inspect the shape, columns, data types, missing values and target distribution.
3. **Preprocess** — fill missing numeric values with numeric column means as a safeguard and encode categorical columns using `LabelEncoder`.
4. **Split data** — separate `Target` from the input features and use an 80/20 train-test split with `random_state=42` and stratification.
5. **Train** — fit `RandomForestClassifier(n_estimators=100, random_state=42)`.
6. **Evaluate** — calculate accuracy, classification report and confusion matrix on the test split.
7. **Inspect feature importance** — plot the Random Forest feature importances.
8. **Save model** — export the fitted estimator to `model.pkl` using Joblib.

The notebook is the source for the training and evaluation procedure. Add the actual metric values from your executed notebook before publishing any performance claims; this README intentionally does not invent an accuracy score.

## 🖥️ Application walkthrough

1. Open the Gradio application.
2. Set the customer's age.
3. Choose a frequent-flyer category.
4. Choose an annual-income class.
5. Set the number of services opted.
6. Choose the social-media account sync status.
7. Choose the hotel booking status.
8. Click **Predict Churn**.
9. Review the predicted class, available probability estimates and input summary.

The app also provides a sample of the dataset and a target distribution chart in expandable sections.

## 🗂️ Project structure

```text
customer-churn/
├── app.py                  # Gradio web application
├── Customertravel.csv      # Tab-separated customer dataset
├── model.pkl               # Trained Random Forest model
├── customer_churn.ipynb    # EDA, preprocessing, training and evaluation
├── requirements.txt        # Python dependencies
├── README.md               # GitHub project documentation
└── .gitattributes          # Git attributes / large-file handling, if configured
```

The application expects `model.pkl` and `Customertravel.csv` to be in the same directory as `app.py`.

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/<YOUR-USERNAME>/<YOUR-REPOSITORY>.git
cd customer-churn
```

Replace the placeholder URL with your actual GitHub repository URL.

### 2. Create and activate a virtual environment

**Windows (PowerShell):**

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

**macOS / Linux:**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

The deployment requirements currently list:

```text
pandas
joblib
scikit-learn==1.7.2
```

The current `app.py` also imports `gradio` and `spaces`. Hugging Face Spaces may provide runtime-specific packages automatically. For a local run, install a compatible Gradio release and the appropriate Hugging Face `spaces` package/runtime support, or adapt the deployment-specific decorator for local development. Do not assume the ZeroGPU decorator works as a normal local GPU service.

### 4. Run the application

```bash
python app.py
```

If the local environment does not provide the Hugging Face `spaces` integration, the `import spaces` / `@spaces.GPU` deployment-specific code must be adapted for local execution. The app's Hugging Face deployment path is the primary supported target in this repository.

## 🤗 Hugging Face Spaces deployment

This project is configured for the Gradio SDK. To deploy:

1. Create or open a Hugging Face Space.
2. Select **Gradio** as the SDK.
3. Upload the application files to the Space repository.
4. Ensure the Space README begins with the Hugging Face YAML metadata (see the Space's own `README.md`).
5. Keep `app.py`, `Customertravel.csv`, `model.pkl` and `requirements.txt` in the Space root.
6. Wait for the build and inspect the Space logs if deployment fails.

The Hugging Face Space configuration currently specifies Gradio SDK 6.5.1, `app.py` as the entry point and Python 3.11. The GitHub README should remain ordinary Markdown; do not paste Hugging Face YAML metadata at the top of this GitHub README unless the repository itself is also being used as the Space repository.

## ⚠️ Important model compatibility note

The currently uploaded `model.pkl` generated a scikit-learn warning: it was serialized using **scikit-learn 1.8.0**, while the deployment environment shown in the logs uses **scikit-learn 1.7.2**.

Scikit-learn does not guarantee that models serialized with Joblib/pickle will remain compatible across different scikit-learn versions. A model may load while still producing warnings or unexpected behavior.

**Recommended before relying on predictions:**

1. Use the same scikit-learn version for training and inference.
2. Preferably retrain the model in the target environment and save a fresh `model.pkl`.
3. Test several known examples and compare predictions with the notebook.
4. Record the exact Python and scikit-learn versions used to train and serve the model.

Do not claim that the model is production-ready until this compatibility issue has been resolved and predictions have been validated.

## 🔍 Evaluation

The notebook calculates:
- Accuracy
- Precision, recall and F1-score through `classification_report`
- Confusion matrix
- Feature importances

Insert the actual output values from your notebook here when publishing a model-performance summary. Metrics should be calculated on held-out test data and should not be fabricated or inferred from the dataset size.

## 🔐 Data, privacy and responsible use

- Do not upload personal, confidential or identifiable customer records without permission.
- Confirm the dataset's source, license and redistribution permissions.
- Treat predictions as decision-support signals, not automatic grounds for denying services or treating customers differently.
- Check for bias, representativeness, data leakage and performance drift before real-world use.

## 🚧 Limitations and future improvements

- Reuse a single fitted preprocessing pipeline instead of rebuilding encoders from dataset categories at app startup.
- Add explicit handling for missing values and unseen categories.
- Add automated tests for feature order, preprocessing and prediction output.
- Report precision, recall, F1-score and confusion matrix in the app.
- Add model explainability (for example, feature importance or SHAP).
- Add model and dataset version information.
- Validate probability calibration and performance on newer, representative data.
- Add a reproducible training script and environment lock file.

## 🤝 Contributing

Suggestions and improvements are welcome. A useful contribution should include:
1. A clear description of the proposed change.
2. Reproducible steps or sample input where applicable.
3. Tests or notebook evidence for changes to preprocessing or model behavior.
4. Updated documentation when application behavior changes.

## 📄 License

No license is specified by this README. Add a license only after confirming that you have the rights to license the code and redistribute the dataset and model. Dataset attribution and usage terms should be documented separately where required.

## 👨‍💻 Author

**Neel Dongre**

If you publish this repository, update the author and repository links as appropriate.
