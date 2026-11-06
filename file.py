import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

@st.cache_resource
def train(method, sampling):
    xt, yt = x_train, y_train
    if sampling == 'SMOTE':
        xt, yt = SMOTE(random_state=42).fit_resample(x_train, y_train)
    elif sampling == 'Under-Sampling':
        xt, yt = RandomUnderSampler(random_state=42).fit_resample(x_train, y_train)
    
    if method == 'LogisticRegression':       
        model = LogisticRegression()
    elif method == 'Random Forest':
        model = RandomForestClassifier(n_estimators=300, class_weight="balanced_subsample",
                                      min_samples_leaf=3, random_state=42)
    elif method == 'Decision Tree':
        model = DecisionTreeClassifier(random_state=42)

    return model.fit(xt, yt)

st.title("Water Classifier App")

cols_per_row = 2
features = ['ph', 'hardness', 'solids', 'chloramines', 'sulfate', 'conductivity', 'organic_carbon', 'trihalomethanes', 'turbidity']
labels = ['pH', 'Hardness', 'Solids', 'Chloramines', 'Sulfate', 'Conductivity', 'Organic Carbon', 'Trihalomethanes', 'Turbidity']

st.text("""1. ph: pH of 1. water (0 to 14).
2. Hardness: Capacity of water to precipitate soap in mg/L.
3. Solids: Total dissolved solids in ppm.
4. Chloramines: Amount of Chloramines in ppm.
5. Sulfate: Amount of Sulfates dissolved in mg/L.
6. Conductivity: Electrical conductivity of water in μS/cm.
7. Organic_carbon: Amount of organic carbon in ppm.
8. Trihalomethanes: Amount of Trihalomethanes in μg/L.
9. Turbidity: Measure of light emiting property of water in NTU.
10. Potability: Indicates if water is safe for human consumption. Potable -1 and Not potable -0""")

st.markdown("## Example Dataset")
df = pd.read_csv("water_potability.csv")
df_water = df.fillna(df.mean())

x = df_water[df_water.columns[0:-1]]
y = df_water['Potability']


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)
method = st.selectbox("Select a model", ['LogisticRegression', 'Random Forest', 'Decision Tree'])
sampling = st.radio("Select a sampling method", ['None', 'SMOTE', 'Under-Sampling'], horizontal=True)
model = train(method, sampling)
y_pred = model.predict(x_test)
acc = accuracy_score(y_test, y_pred)
st.metric(f"{method} + {sampling} accuracy", f"{acc * 100:.2f}%")

event = st.dataframe(
    df_water,
    on_select='rerun',
    selection_mode='single-row'
)

rows = event.selection.rows

if rows and st.session_state.get('last_row') != rows[0]:
    row = df_water.iloc[rows[0]]
    for key, col in zip(features, x.columns):   # key = input key, col = CSV column
        st.session_state[key] = float(row[col])
    st.session_state['last_row'] = rows[0]
elif not rows:
    st.session_state['last_row'] = None

for label, key in zip(labels, features):
    st.number_input(f"{label} value", key=key, format="%.14f")

# ph = st.number_input("pH value", key='ph')
# hardness = st.number_input("Hardness value", key='hardness')
# solids = st.number_input("Solids value", key='solids')
# chloramines = st.number_input("Chloramines value", key='chloramines')
# sulfate = st.number_input("Sulfate value", key='sulfate')
# conductivity = st.number_input("Conductivity value", key='conductivity')
# organic_carbon = st.number_input("Organic Carbon value", key='organic_carbon')
# trihalomethanes = st.number_input("Trihalomethanes value", key='trihalomethanes')
# turbidity = st.number_input("Turbidity value", key='turbidity')

if st.button('Predict'):
    values = [st.session_state[k] for k in features]
    x_input = pd.DataFrame([values], columns=x.columns)
    st.write(x_input)
    pred = model.predict(x_input)[0]
    proba = model.predict_proba(x_input)[0][1]
    st.write( f'Prediction: {pred}')
    st.write(f'Probability of potable: {proba:.1%}')