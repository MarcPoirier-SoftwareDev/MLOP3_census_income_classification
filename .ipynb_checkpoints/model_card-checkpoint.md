# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details

- Model Name: Census Income MLP
- Model Verion: 1.0
- Model Date: July 24; 2025
- Developers: Marc Poirier
- Description: A PyTorch-based Multi-Layer Perceptron (MLP) model for binary classification, predicting whether an individual's annual income exceeds $50,000 based on demographic and socioeconomic features from census data.
- Architecture
    - Input Layer: Linear layer mapping input features (default dimension: 108 after preprocessing) to hidden dimension.
    - Hidden Layers:  Configurable number of layers (default: 2), each consisting of a Linear layer (hidden_dim to hidden_dim), Dropout for regularization, and ReLU activation.
    - Output Layer:  Linear layer mapping hidden dimension to number of classes (default: 2), producing raw logits for CrossEntropyLoss.
- Hyperparameters (tunable via Optuna)
    - Batch Size: Tunable options [64, 128, 256, 512, 1024]; default: 1028 (note: default differs from tuning options).
    - Learning Rate: Tunable range 1e-5 to 1e-1 (log scale); default: 0.001.
    - Dropout Rate: Tunable options [0.3, 0.4, 0.5, 0.6, 0.7]; default: 0.5.
    - Number of Layers: Tunable options [1, 2, 3, 4, 5]; default: 2.
    - Hidden Dimension: Tunable options [5, 10, 25, 50]; default: 50.
    - Epochs: Default: 200 (150 during Optuna trials, 500 for final training).
- Training Environment: PyTorch (with Adam optimizer and CrossEntropyLoss); runs on GPU (CUDA) if available, otherwise CPU; uses Optuna for hyperparameter tuning (20 trials with TPESampler); additional libraries: NumPy, scikit-learn (for metrics, splitting, preprocessing), torch.utils.data for DataLoader.
- Preprocessing:
    - Categorical features (e.g., workclass, education, marital-status) one-hot encoded using OneHotEncoder, expanding to part of the 108 input dimensions.
    - Continuous features (e.g., age, hours-per-week) scaled using StandardScaler.
    - Labels binarized using LabelBinarizer (0: <=50K, 1: >50K).
- Model Files: Saved as mlp.pt (state dict), along with encoder.pkl, lb.pkl, scaler.pkl in the model/ directory.



## Intended Use

- Primary Use Case: This model is intended for educational and demonstrative purposes in machine learning operations (MLOps), such as predicting income brackets from demographic and socioeconomic census data to analyze income inequality or inform policy decisions.


- Users: Data scientists, ML engineers, researchers, or policymakers interested in socioeconomic predictions.

- In-Scope Applications: Batch inference on similar census-like datasets for binary income classification; fairness analysis via data slices (e.g., by education or race).

- Out-of-Scope Applications: Not for real-world deployment in high-stakes decisions like loan approvals, hiring, or welfare allocation without further validation, as it may perpetuate biases. Not suitable for time-series or non-tabular data.




## Training Data

- Dataset: UCI Adult Income Dataset (also known as Census Income), sourced from data/census.csv (cleaned version: census_clean.csv after removing whitespaces).
- Size: Approximately 48,842 instances (standard UCI Adult size), split 80% for training (~39,073 instances) and 20% for evaluation.
- Features: 14 attributes including:
    - Continuous: age, fnlwgt, education-num, capital-gain, capital-loss, hours-per-week.
    - Categorical: workclass, education, marital-status, occupation, relationship, race, sex, native-country.
- Label: Binary salary (<=50K or >50K).
- Preprocessing: As described in Model Details; training data processed with fit encoders/scalers.
- Imbalance: The dataset is imbalanced (~75% <=50K, ~25% >50K), but no explicit resampling in the code.
- Source: Public UCI Machine Learning Repository (1994 US Census extract).
- Random Seed: 42 for reproducibility in train-test split.





## Evaluation Data

- Dataset: 20% hold-out test set from the same UCI Adult dataset (~9,769 instances).
- Slice Evaluation: Additional evaluation on data slices grouped by categorical features (e.g., education level) to assess fairness. Slices are created post--split on the test set using get_data_slices().
    - Example Feature for Slicing: Education (values like 'Bachelors', 'HS-grad', 'Some-college', etc.).
    - Other Possible Slices: Race, sex, etc., though code defaults to education.

Preprocessing: Same as training, but using fitted encoders/scalers (transform only).




## Metrics

The model was evaluated using precision, recall and F1 score.

Overall test metrics: Precision=0.7489, Recall=0.6416, F1=0.6911


## Ethical Considerations

- Bias and Fairness: The dataset reflects 1994 US Census biases, potentially discriminating against protected groups (e.g., lower recall for minorities or females in high-income predictions). Slice evaluations help identify disparities, but the model may amplify societal biases in race, sex, or education.

- Privacy: Uses anonymized census data, but features like native-country could indirectly reveal sensitive info.

- Transparency: Model is interpretable via feature importances (though not implemented), but neural networks are black-box; use SHAP or LIME for explanations.

- Impact: Predictions could influence policy or resource allocation; misuse might exacerbate income inequality.

- Mitigation: Use fairness-aware training (not in code, but recommend adding e.g., adversarial debiasing). Monitor demographic parity and equalized odds.




## Caveats and Recommendations

- Caveats:
    - Data Age: Dataset from 1994; may not reflect current demographics/economics (e.g., post-COVID shifts).
    - Imbalance: Minority class (>50K) underrepresented, leading to lower recall.
    - Generalization: Trained on US data; poor performance on non-US or modern datasets.
    - Overfitting: Hyperparameter tuning helps, but monitor validation loss.
    - No Fairness Constraints: Code evaluates slices but doesn't mitigate bias during training.
    - Compute: GPU recommended for large epochs; CPU fallback may slow training.


- Recommendations:
    - Retraining: Update with newer data (e.g., recent census) and retrain periodically.
    - Fairness Enhancements: Integrate libraries like AIF360 for bias mitigation (e.g., reweighing samples).
    - Deployment: Use in low-stakes settings; add human oversight for decisions.
    - Monitoring: Track slice metrics in production; set thresholds (e.g., demographic parity <0.1).
    - Extensions: Add cross-validation instead of single split; experiment with more layers or ensembles.



