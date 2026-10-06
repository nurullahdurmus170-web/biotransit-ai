"""
BioTransit AI - Demo Arayuzu
=============================
Calistirmak icin:
    pip install streamlit scikit-learn pandas numpy joblib matplotlib
    cd demo
    streamlit run demo_app.py

Akis ("tek shipment verisi yukleme -> analiz -> LOW/MEDIUM/HIGH"):
  1) Hazir 600 gonderilik veri setinden bir shipment_id secilir, VEYA
  2) Kullanici kendi tek-gonderilik ham sensor CSV'sini yukler
     (kolonlar: shipment_id, elapsed_min, temperature_c, accel_rms_g,
      vibration_freq_hz, shock_peak_g)
  3) Ozellik cikarimi (notebook ile birebir ayni fonksiyon) + RF modeli
     calisir, risk sinifi ve aciklanabilir maruziyet ozeti gosterilir.

Not: "Ornek Gonderi Sec" sekmesi, bu depoda yer almayan
BioTransit_sensor_timeseries.csv dosyasi varsa ham sicaklik/sok grafigini
de gosterir; dosya yoksa grafik olmadan, yalnizca risk siniflandirmasiyla
calismaya devam eder (asagidaki load_raw_sensor bu durumu yakalar).
"""
import numpy as np
import pandas as pd
import joblib
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(page_title="BioTransit AI - Demo", layout="centered")

FEATURE_COLS = [
    "duration_h", "mean_temp_c", "max_temp_c", "temp_sd_c",
    "time_above_8c_min", "thermal_auc_above_8c_c_h",
    "mean_accel_rms_g", "max_accel_rms_g",
    "vibration_minutes_above_2g", "vibration_dose_g2_h",
    "dominant_vibration_freq_hz",
    "shock_count_ge_5g", "shock_count_ge_20g",
    "max_shock_g", "mean_shock_g",
]

RISK_STYLE = {
    "LOW":    {"color": "#2e7d32", "emoji": "🟢", "label": "DÜŞÜK RİSK"},
    "MEDIUM": {"color": "#f9a825", "emoji": "🟡", "label": "ORTA RİSK"},
    "HIGH":   {"color": "#c62828", "emoji": "🔴", "label": "YÜKSEK RİSK"},
}


def extract_shipment_features(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Notebook'taki extract_shipment_features ile birebir aynidir -
    demo ile CDR raporundaki sonuclarin tutarli olmasini saglar."""
    rows = []
    for shipment_id, x in raw_df.groupby("shipment_id", sort=False):
        x = x.sort_values("elapsed_min").copy()
        diffs = np.diff(x["elapsed_min"].to_numpy())
        dt_min = float(np.median(diffs)) if len(diffs) else 10.0
        shock_events = x.loc[x["shock_peak_g"] > 0, "shock_peak_g"]
        vibration_window = x.loc[x["accel_rms_g"] > 2, "vibration_freq_hz"]
        if len(vibration_window):
            dominant_freq = float(vibration_window.median())
        else:
            w = np.maximum(x["accel_rms_g"].to_numpy(), 1e-6) ** 2
            dominant_freq = float(np.average(x["vibration_freq_hz"].to_numpy(), weights=w))
        rows.append({
            "shipment_id": shipment_id,
            "duration_h": len(x) * dt_min / 60.0,
            "mean_temp_c": x["temperature_c"].mean(),
            "max_temp_c": x["temperature_c"].max(),
            "temp_sd_c": x["temperature_c"].std(ddof=0),
            "time_above_8c_min": (x["temperature_c"] > 8).sum() * dt_min,
            "thermal_auc_above_8c_c_h": np.maximum(x["temperature_c"] - 8, 0).sum() * dt_min / 60.0,
            "mean_accel_rms_g": x["accel_rms_g"].mean(),
            "max_accel_rms_g": x["accel_rms_g"].max(),
            "vibration_minutes_above_2g": (x["accel_rms_g"] > 2).sum() * dt_min,
            "vibration_dose_g2_h": (x["accel_rms_g"] ** 2).sum() * dt_min / 60.0,
            "dominant_vibration_freq_hz": dominant_freq,
            "shock_count_ge_5g": int((x["shock_peak_g"] >= 5).sum()),
            "shock_count_ge_20g": int((x["shock_peak_g"] >= 20).sum()),
            "max_shock_g": x["shock_peak_g"].max(),
            "mean_shock_g": shock_events.mean() if len(shock_events) else 0.0,
        })
    return pd.DataFrame(rows)


@st.cache_resource
def load_model():
    return joblib.load("rf_model.joblib")


@st.cache_data
def load_sample_library():
    return pd.read_csv("shipments_with_features.csv")


@st.cache_data
def load_raw_sensor():
    """Ham zaman serisi dosyası bu depoda yer almıyor; dosya yoksa None
    döner ve çağıran taraf grafik olmadan devam eder (çökme yok)."""
    try:
        return pd.read_csv("BioTransit_sensor_timeseries.csv")
    except FileNotFoundError:
        return None


def render_result(feat_row: pd.Series, raw_slice: pd.DataFrame, model):
    X_row = feat_row[FEATURE_COLS].to_frame().T
    pred = model.predict(X_row)[0]
    probs = pd.Series(model.predict_proba(X_row)[0], index=model.classes_)
    confidence = float(probs[pred])
    style = RISK_STYLE[pred]

    st.markdown(
        f"""
        <div style="border:2px solid {style['color']}; border-radius:10px; padding:16px; text-align:center;">
            <div style="font-size:14px; color:gray;">GÖNDERİ RİSK ÇIKTISI</div>
            <div style="font-size:34px; font-weight:800; color:{style['color']};">
                {style['emoji']} {style['label']}
            </div>
            <div style="font-size:14px; color:gray;">Model güveni: %{confidence*100:.0f}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("Maks. Sıcaklık", f"{feat_row['max_temp_c']:.1f} °C")
    c2.metric("8°C Üstü Süre", f"{feat_row['time_above_8c_min']:.0f} dk")
    c3.metric("Maks. Şok", f"{feat_row['max_shock_g']:.1f} G")

    if raw_slice is not None and len(raw_slice):
        fig, ax1 = plt.subplots(figsize=(7, 3.2))
        t = raw_slice["elapsed_min"] / 60.0
        ax1.plot(t, raw_slice["temperature_c"], color="#1f77b4", label="Sıcaklık (°C)")
        ax1.axhline(8, color="#1f77b4", linestyle="--", linewidth=0.8, alpha=0.6)
        ax1.set_xlabel("Geçen süre (saat)")
        ax1.set_ylabel("Sıcaklık (°C)", color="#1f77b4")
        ax2 = ax1.twinx()
        shocks = raw_slice[raw_slice["shock_peak_g"] > 0]
        ax2.scatter(shocks["elapsed_min"] / 60.0, shocks["shock_peak_g"], color="#d62728", s=18, label="Şok (G)")
        ax2.set_ylabel("Şok (G)", color="#d62728")
        fig.tight_layout()
        st.pyplot(fig)

    st.write("**Açıklanabilir maruziyet özeti:**")
    st.markdown(
        f"- Termal AUC (8°C üstü): **{feat_row['thermal_auc_above_8c_c_h']:.1f} °C·saat**\n"
        f"- 2G üstü titreşim süresi: **{feat_row['vibration_minutes_above_2g']:.0f} dk**\n"
        f"- ≥20G şok olayı sayısı: **{int(feat_row['shock_count_ge_20g'])}**\n"
    )
    st.caption(
        "⚠️ Bu çıktı literatürden ölçeklenmiş sentetik veriyle eğitilmiş bir modelin "
        "maruziyet-riski tahminidir; biyolojik aktivite ölçümü veya numune serbest "
        "bırakma kararı değildir."
    )


st.title("BioTransit AI")
st.caption("Sevkiyat risk sınıflandırma demosu")

model = load_model()
shipments = load_sample_library()

tab1, tab2 = st.tabs(["Örnek Gönderi Seç", "Kendi CSV'ni Yükle"])

with tab1:
    st.write("Hazır veri setinden 600 gönderiden birini seçin (LOW/MEDIUM/HIGH örnekleri dahil).")
    default_examples = ["BT0001 (LOW - stabil soğuk zincir)",
                         "BT0214 (MEDIUM - uzun sıcaklık sapması)",
                         "BT0220 (HIGH - uzun sıcaklık sapması)"]
    choice = st.selectbox("Hızlı örnekler", default_examples)
    shipment_id = choice.split(" ")[0]
    manual_id = st.text_input("veya bir shipment ID yazın (örn. BT0057)", "")
    if manual_id.strip():
        shipment_id = manual_id.strip()

    if st.button("Analiz Et", key="lib_btn"):
        row = shipments[shipments.shipment_id == shipment_id]
        if row.empty:
            st.error(f"'{shipment_id}' bulunamadı.")
        else:
            raw = load_raw_sensor()
            if raw is None:
                st.info(
                    "Ham sensör zaman serisi dosyası bu depoda yok; "
                    "sıcaklık/şok grafiği gösterilemiyor, ama risk "
                    "sınıflandırması aşağıda yine de çalışıyor."
                )
                raw_slice = None
            else:
                raw_slice = raw[raw.shipment_id == shipment_id]
            render_result(row.iloc[0], raw_slice, model)

with tab2:
    st.write(
        "Tek bir gönderiye ait ham sensör verisini yükleyin. "
        "Beklenen kolonlar: `shipment_id, elapsed_min, temperature_c, "
        "accel_rms_g, vibration_freq_hz, shock_peak_g`"
    )
    uploaded = st.file_uploader("CSV dosyası", type="csv")
    if uploaded is not None:
        try:
            user_raw = pd.read_csv(uploaded)
            feats = extract_shipment_features(user_raw)
            if len(feats) != 1:
                st.warning(f"Dosyada {len(feats)} farklı shipment_id bulundu; ilki kullanılacak.")
            row = feats.iloc[0]
            sid = row["shipment_id"]
            render_result(row, user_raw[user_raw.shipment_id == sid], model)
        except Exception as e:
            st.error(f"Dosya işlenemedi: {e}")

st.divider()
st.caption("BioTransit AI")
