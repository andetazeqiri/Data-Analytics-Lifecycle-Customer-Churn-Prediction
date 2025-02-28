# -*- coding: utf-8 -*-


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score
import ipywidgets as widgets
from IPython.display import display

url = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
data = pd.read_csv(url)
data.head()

print(data.info())
print(data.describe())
print(data.isnull().sum())

data['TotalCharges'] = pd.to_numeric(data['TotalCharges'], errors='coerce')
data['TotalCharges'].fillna(data['TotalCharges'].median(), inplace=True)

data = data.drop(['customerID'], axis=1)

data = pd.get_dummies(data, drop_first=True)

plt.figure(figsize=(6, 4))
sns.countplot(data['Churn_Yes'])
plt.title("Churn Distribution")
plt.show()

X = data.drop('Churn_Yes', axis=1)
y = data['Churn_Yes']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

importances = model.feature_importances_
feature_names = X.columns
indices = np.argsort(importances)[-10:]

plt.figure(figsize=(8, 6))
plt.title("Feature Importances")
plt.barh(range(len(indices)), importances[indices], align="center")
plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
plt.xlabel("Relative Importance")
plt.show()

import matplotlib.pyplot as plt
import seaborn as sns
import ipywidgets as widgets
from IPython.display import display, clear_output, HTML

def predict_churn(tenure, MonthlyCharges, TotalCharges, Contract, PaymentMethod):
    user_data = {
        'tenure': [tenure],
        'MonthlyCharges': [MonthlyCharges],
        'TotalCharges': [TotalCharges],
        'Contract_Month-to-month': [1 if Contract == 'Month-to-month' else 0],
        'Contract_One year': [1 if Contract == 'One year' else 0],
        'Contract_Two year': [1 if Contract == 'Two year' else 0],
        'PaymentMethod_Bank transfer (automatic)': [1 if PaymentMethod == 'Bank transfer (automatic)' else 0],
        'PaymentMethod_Credit card (automatic)': [1 if PaymentMethod == 'Credit card (automatic)' else 0],
        'PaymentMethod_Electronic check': [1 if PaymentMethod == 'Electronic check' else 0],
        'PaymentMethod_Mailed check': [1 if PaymentMethod == 'Mailed check' else 0],
    }

    user_df = pd.DataFrame(user_data)

    for col in X.columns:
        if col not in user_df.columns:
            user_df[col] = 0
    user_df = user_df[X.columns]

    user_df_scaled = scaler.transform(user_df)


    prediction_probabilities = model.predict_proba(user_df_scaled)
    print("Prediction Probabilities:", prediction_probabilities)


    prediction = (prediction_probabilities[0][1] >= 0.5).astype(int)
    return prediction

def display_context_graphs(tenure, MonthlyCharges):
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))


    sns.histplot(data, x="tenure", hue="Churn_Yes", multiple="stack", ax=axs[0], palette="coolwarm")
    axs[0].axvline(tenure, color="lime", linestyle="--", linewidth=2, label=f"Selected tenure: {tenure}")
    axs[0].set_title("Tenure Distribution by Churn", fontsize=14, fontweight='bold')
    axs[0].legend()


    sns.histplot(data, x="MonthlyCharges", hue="Churn_Yes", multiple="stack", ax=axs[1], palette="coolwarm")
    axs[1].axvline(MonthlyCharges, color="lime", linestyle="--", linewidth=2, label=f"Selected Charges: ${MonthlyCharges}")
    axs[1].set_title("Monthly Charges Distribution by Churn", fontsize=14, fontweight='bold')
    axs[1].legend()

    plt.tight_layout()
    plt.show()

def on_button_clicked(b):
    with output:
        clear_output()

        result = predict_churn(
            tenure_slider.value,
            monthly_charges_slider.value,
            total_charges_slider.value,
            contract_dropdown.value,
            payment_method_dropdown.value
        )


        if result == 1:
            prediction_text = "<div style='color:red; font-size:28px; font-weight:bold; text-align:center;'>This customer is likely to churn.</div>"
        else:
            prediction_text = "<div style='color:green; font-size:28px; font-weight:bold; text-align:center;'>This customer is not likely to churn.</div>"


        display(HTML(prediction_text))


        with output:
            display_context_graphs(tenure_slider.value, monthly_charges_slider.value)

title = widgets.HTML("<h1 style='color:pink; text-align:center; font-size:36px; font-weight:bold;'> Data Analytics Lifecycle Customer Churn Prediction</h1>")
signature = widgets.HTML("<p style='text-align:center; font-style:italic; color:gray;'>Created by Andeta Zeqiri</p>")


tenure_slider = widgets.IntSlider(
    min=0, max=72, description="Tenure", style={'description_width': 'initial'},
    layout=widgets.Layout(width='80%', height='40px')
)
monthly_charges_slider = widgets.FloatSlider(
    min=10, max=200, description="Monthly Charges", style={'description_width': 'initial'},
    layout=widgets.Layout(width='80%', height='40px')
)
total_charges_slider = widgets.FloatSlider(
    min=10, max=10000, description="Total Charges", style={'description_width': 'initial'},
    layout=widgets.Layout(width='80%', height='40px')
)
contract_dropdown = widgets.Dropdown(
    options=['Month-to-month', 'One year', 'Two year'], description="Contract",
    style={'description_width': 'initial'}, layout=widgets.Layout(width='80%', height='40px')
)
payment_method_dropdown = widgets.Dropdown(
    options=['Bank transfer (automatic)', 'Credit card (automatic)', 'Electronic check', 'Mailed check'],
    description="Payment Method", style={'description_width': 'initial'}, layout=widgets.Layout(width='80%', height='40px')
)
output = widgets.Output(layout={'border': '1px solid black', 'padding': '10px'})


button = widgets.Button(
    description="Predict Churn", button_style='success',
    layout=widgets.Layout(width='80%', height='50px'),
    style={'font_weight': 'bold', 'font_size': '18px'}
)
button.on_click(on_button_clicked)


centered_layout = widgets.VBox([
    title,
    widgets.VBox([tenure_slider, monthly_charges_slider, total_charges_slider, contract_dropdown, payment_method_dropdown, button],
                 layout=widgets.Layout(align_items='center', width='100%')),
    output,
    signature
])


display(centered_layout)
