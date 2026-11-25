# Skin Lesion Classification – Streamlit

A Streamlit interface template for the HAM10000 skin-lesion classification
project. The intended pipeline uses HOG + HSV image features and a trained
classical machine-learning classifier (AdaBoost selected in the project
validation comparison).

## Project structure

```text
Skin_Lesion_Streamlit/
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
├── src/
│   ├── __init__.py
│   └── feature_extraction.py
└── models/
    └── (place final model artifacts here)
```

## Important before using predictions

The original training pipeline generated 6,132 features per image (5,940 HOG
and 192 HSV). The feature extraction file in this ZIP contains placeholder
HOG/image settings because the exact original HPC feature-extraction source
was not included when this template was generated.

Before making predictions, copy the exact preprocessing from the original
training pipeline into `src/feature_extraction.py`, including:

- image resizing and colour conversion
- HOG orientations, pixels-per-cell, cells-per-block and normalization
- HSV histogram bins, ranges and normalization
- feature concatenation order

The app checks that exactly 6,132 features are produced. This check alone does
not prove the feature values match training; the settings must be identical.

Also confirm whether your final saved estimator expects scaled features. If
your final artifact is a complete scikit-learn Pipeline containing its scaler,
do not load/apply a separate scaler. If it is a bare estimator trained on
scaled features, put the matching fitted scaler at `models/scaler.joblib`.

## Add model files

Put your final, test-evaluated AdaBoost model here:

`models/best_model.joblib`

If needed, put the fitted StandardScaler here:

`models/scaler.joblib`

Only use the final model after completing seed experiments and final evaluation.
Do not commit HAM10000 images, metadata, feature arrays, or temporary model
checkpoints.

## Run locally in VS Code

### Windows PowerShell

```powershell
py -m venv .venv
.venv\\Scripts\\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

If PowerShell blocks activation, use Command Prompt:

```bat
.venv\\Scripts\\activate.bat
```

## GitHub

1. Create a new GitHub repository.
2. Check the model file size before uploading.
3. Normal GitHub Git uploads have a 100 MB per-file limit. For larger models,
   use Git LFS or a model-hosting service and download the artifact securely
   at deployment time.
4. Do not commit secrets, datasets, patient information, or private files.

## Deploy on Streamlit Community Cloud

1. Push the app code to GitHub.
2. Open https://share.streamlit.io/
3. Connect GitHub and choose the repository and branch.
4. Set the main file path to `app.py`.
5. Deploy.

Make sure the model artifact is accessible to the deployed app. A model file
excluded by `.gitignore` will not automatically be present in the cloud.

## Medical disclaimer

This is an educational academic prototype, not a clinically validated medical
device. Predictions must not be used for diagnosis or treatment decisions.
