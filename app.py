
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import urllib.request
import os
from sklearn.ensemble import IsolationForest

st.set_page_config(page_title="Energy Dashboard", layout="wide")

st.title("Energy Consumption Anomaly Detection")
st.write("Detect unusual energy consumption using Machine Learning.")

@st.cache_data
def load_data():
    url = "https://archive.ics.uci.edu/static/public/235/individual+household+electric+power+consumption.zip"
    zip_path = "/content/energy_data.zip"

    if not os.path.exists(zip_path):
        urllib.request.urlretrieve(url, zip_path)

    df = pd.read_csv(zip_path, sep=";", na_values="?")
    df = df[["Date", "Time", "Global_active_power"]].dropna()

    df["Global_active_power"] = pd.to_numeric(
        df["Global_active_power"], errors="coerce"
    )
    df = df.dropna()

    df["Datetime"] = pd.to_datetime(
        df["Date"] + " " + df["Time"],
        dayfirst=True,
        errors="coerce"
    )

    df = df.dropna(subset=["Datetime"]).set_index("Datetime")

    return df["Global_active_power"].resample("h").mean().dropna()


hourly_data = load_data()

contamination = st.sidebar.slider(
    "Anomaly Detection Level", 0.01, 0.10, 0.05, 0.01
)

X = hourly_data.values.reshape(-1, 1)

model = IsolationForest(
    contamination=contamination,
    random_state=42
)

predictions = model.fit_predict(X)

result = pd.DataFrame({
    "Energy Consumption": hourly_data.values,
    "Prediction": predictions
}, index=hourly_data.index)

result["Status"] = result["Prediction"].apply(
    lambda x: "Normal" if x == 1 else "Anomaly"
)

total = len(result)
normal = len(result[result["Status"] == "Normal"])
anomalies = len(result[result["Status"] == "Anomaly"])

average = result["Energy Consumption"].mean()
maximum = result["Energy Consumption"].max()
minimum = result["Energy Consumption"].min()

st.header("Energy Consumption Overview")

c1, c2, c3 = st.columns(3)
c1.metric("Total Readings", total)
c2.metric("Normal Readings", normal)
c3.metric("Anomalies", anomalies)

st.header("Energy Statistics")

c4, c5, c6 = st.columns(3)
c4.metric("Average", round(average, 3))
c5.metric("Maximum", round(maximum, 3))
c6.metric("Minimum", round(minimum, 3))

st.header("Energy Consumption Graph")

fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(result.index, result["Energy Consumption"], label="Consumption")

anomaly_data = result[result["Status"] == "Anomaly"]

ax.scatter(
    anomaly_data.index,
    anomaly_data["Energy Consumption"],
    color="red",
    label="Anomaly"
)

ax.set_xlabel("Date")
ax.set_ylabel("Energy Consumption (kW)")
ax.legend()
plt.xticks(rotation=45)
plt.tight_layout()

st.pyplot(fig)

st.header("Detected Anomalies")
st.dataframe(anomaly_data)

st.header("Ask Energy AI Assistant")

question = st.text_input("Ask a question:")

if question:
    q = question.lower()

    if "average" in q or "mean" in q:
        st.write("Average:", round(average, 3), "kW")
    elif "maximum" in q or "highest" in q:
        st.write("Maximum:", round(maximum, 3), "kW")
    elif "minimum" in q or "lowest" in q:
        st.write("Minimum:", round(minimum, 3), "kW")
    elif "anomal" in q or "abnormal" in q:
        st.write("Detected anomalies:", anomalies)
    elif "normal" in q:
        st.write("Normal readings:", normal)
    elif "total" in q:
        st.write("Total readings:", total)
    elif "status" in q or "condition" in q:
        st.write("Anomaly percentage:", round(anomalies / total * 100, 2), "%")
    else:
        st.write("Try asking about average, maximum, minimum, anomalies or total readings.")

st.header("Download Results")

csv = result.to_csv().encode("utf-8")

st.download_button(
    "Download CSV",
    csv,
    "energy_anomaly_results.csv",
    "text/csv"
)
