# Jaguar Burger MX — BI & Analytics App (Streamlit)
# -------------------------------------------------
# Ejecuta:
#   pip install -r requirements.txt
#   streamlit run app.py

import io
import os
import time
from datetime import date
from functools import lru_cache

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import pydeck as pdk
import streamlit as st
from PIL import Image, ImageDraw, ImageFilter, ImageFont
from numpy.random import default_rng
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    accuracy_score,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.cluster import KMeans

# ──────────────────────────────────────────────────────────────────────────────
# Configuración base
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Jaguar Burger MX — BI",
    page_icon="🍔",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Estilos suaves (cards y tipografía más grande) — listo para modo oscuro.
st.markdown(
    """
    <style>
      html, body, [data-testid="stAppViewContainer"] * {font-size: 17px !important;}
      h1 {font-size: 2.0rem !important;}
      h2 {font-size: 1.6rem !important;}
      h3 {font-size: 1.25rem !important;}
      .card {
        padding: 0.8rem 1rem;
        border-radius: 14px;
        border: 1px solid rgba(255,255,255,.08);
        background: rgba(255,255,255,.03);
      }
      .tagline {opacity:.95}
      .pill {display:inline-block;padding:6px 10px;border-radius:999px;
             border:1px solid rgba(255,255,255,.12);background:rgba(255,255,255,.04);}
      .logo-wrap img {border-radius: 10px;}
    </style>
    """,
    unsafe_allow_html=True,
)

# ──────────────────────────────────────────────────────────────────────────────
# Rutas de imágenes (puedes subir tus archivos a assets/)
# ──────────────────────────────────────────────────────────────────────────────
ASSETS = {
    "logo": "assets/logo.png",
    "hero": "assets/hero.jpg",
    "tech": {
        "Streamlit": "assets/tech/streamlit.png",
        "scikit-learn": "assets/tech/sklearn.png",
        "Plotly": "assets/tech/plotly.png",
        "PyDeck": "assets/tech/pydeck.png",
        "Snowflake": "assets/tech/snowflake.png",
        "Gemini (ADK)": "assets/tech/gemini.png",
    },
}

# ──────────────────────────────────────────────────────────────────────────────
# Utilidades de imagen (fallback si falta el archivo)
# ──────────────────────────────────────────────────────────────────────────────
@lru_cache(maxsize=32)
def _font(size=64):
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except Exception:
        return ImageFont.load_default()

def make_brand_fallback(w=900, h=260, text="Jaguar Burger MX"):
    def _grad():
        base = Image.new("RGB", (w, h), (25, 25, 25))
        top = Image.new("RGB", (w, h), (255, 140, 0))
        mask = Image.linear_gradient("L").resize((w, h))
        return Image.composite(top, base, mask).filter(ImageFilter.GaussianBlur(0.3))

    img = _grad()
    d = ImageDraw.Draw(img)
    # hamburguesa simple
    cx, cy, r = int(h*0.33), int(h*0.55), int(h*0.28)
    d.rounded_rectangle([(cx-r, cy-r), (cx+r, cy-int(r*0.1))], radius=40, fill=(255,210,120))
    d.rounded_rectangle([(cx-r, cy), (cx+r, cy+int(r*0.15))], radius=20, fill=(90,40,30))
    d.rounded_rectangle([(cx-r, cy+int(r*0.3)), (cx+r, cy+int(r*0.5))], radius=40, fill=(255,210,120))
    f1, f2 = _font(90), _font(36)
    tx = cx + r + 28
    d.text((tx, cy-44), text, font=f1, fill=(255,255,255))
    d.text((tx, cy+36), "Business Intelligence & Analytics", font=f2, fill=(255,255,255,230))
    buf = io.BytesIO(); img.save(buf, format="PNG"); buf.seek(0)
    return buf

def load_image(path, fallback=None):
    if path and os.path.exists(path):
        return path
    return fallback() if callable(fallback) else None

# ──────────────────────────────────────────────────────────────────────────────
# Catálogo de plazas
# ──────────────────────────────────────────────────────────────────────────────
PLAZAS = [
    {"estado":"CDMX", "ciudad":"Ciudad de México", "lat":19.4326, "lon":-99.1332},
    {"estado":"Jalisco", "ciudad":"Guadalajara", "lat":20.6597, "lon":-103.3496},
    {"estado":"Nuevo León", "ciudad":"Monterrey", "lat":25.6866, "lon":-100.3161},
    {"estado":"Puebla", "ciudad":"Puebla", "lat":19.0414, "lon":-98.2063},
    {"estado":"Edomex", "ciudad":"Toluca", "lat":19.2826, "lon":-99.6557},
    {"estado":"Yucatán", "ciudad":"Mérida", "lat":20.9674, "lon":-89.5926},
    {"estado":"Baja California", "ciudad":"Tijuana", "lat":32.5149, "lon":-117.0382},
    {"estado":"Guanajuato", "ciudad":"León", "lat":21.1250, "lon":-101.6850},
    {"estado":"Querétaro", "ciudad":"Querétaro", "lat":20.5888, "lon":-100.3899},
]

# ──────────────────────────────────────────────────────────────────────────────
# Generación de datos de un año
# ──────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def generate_dataset(year=2024, seed=42) -> pd.DataFrame:
    rng = default_rng(seed)
    stores = []
    sid = 100
    for p in PLAZAS:
        for s in range(2):
            stores.append({
                "store_id": sid,
                "store_name": f"{p['ciudad']} — Sucursal {s+1}",
                "estado": p["estado"], "ciudad": p["ciudad"],
                "lat": p["lat"] + rng.normal(0, 0.02),
                "lon": p["lon"] + rng.normal(0, 0.02),
                "base_demand": rng.uniform(120, 240),
                "price": rng.uniform(80, 120),
                "cost_rate": rng.uniform(0.55, 0.62),
            }); sid += 1

    idx = pd.date_range(f"{year}-01-01", f"{year}-12-31", freq="D")
    holidays = {f"{year}-02-14", f"{year}-04-30", f"{year}-05-10", f"{year}-09-16",
                f"{year}-12-12", f"{year}-12-24", f"{year}-12-25", f"{year}-12-31"}
    rows = []

    for st_info in stores:
        monthly_amp = rng.uniform(0.06, 0.18)
        weekly_amp  = rng.uniform(0.12, 0.24)
        promo_freq  = rng.integers(10, 22)
        last_promo = rng.integers(0, 10)

        for d in idx:
            dow, month = d.dayofweek, d.month
            is_weekend = 1 if dow >= 5 else 0
            season = 1 + monthly_amp * np.sin(2*np.pi*(month/12))
            weekday_boost = 1 + weekly_amp * (1 if is_weekend else -0.5)

            last_promo += 1
            has_promo = 1 if last_promo >= promo_freq else 0
            discount = rng.choice([0,5,10,15], p=[0.6,0.2,0.15,0.05]) if has_promo else 0
            if has_promo: last_promo = 0

            marketing = max(0, rng.normal(1400, 450)) * (1.2 if has_promo else 1.0)
            holiday_boost = 1.35 if d.strftime("%Y-%m-%d") in holidays else 1.0

            demand = (st_info["base_demand"] * season * weekday_boost * holiday_boost
                      * rng.uniform(0.90, 1.10))
            orders = max(0, int(rng.normal(demand, demand*0.12)))
            price = max(65, st_info["price"] * (1 - discount/100))
            sales = orders * price
            cogs  = sales * st_info["cost_rate"]
            profit = sales - cogs - marketing
            ticket = sales/orders if orders>0 else price
            margin = profit/sales if sales>0 else 0

            rows.append({
                "date": d.date(),
                "store_id": st_info["store_id"], "store_name": st_info["store_name"],
                "estado": st_info["estado"], "ciudad": st_info["ciudad"],
                "lat": st_info["lat"], "lon": st_info["lon"],
                "orders": orders, "price": round(price,2), "discount": discount,
                "marketing_mxn": round(marketing,2),
                "sales_mxn": round(sales,2), "cogs_mxn": round(cogs,2),
                "profit_mxn": round(profit,2),
                "ticket_avg_mxn": round(ticket,2), "margin_pct": round(margin,4),
                "weekday": dow, "month": month, "is_weekend": is_weekend,
            })

    df = pd.DataFrame(rows)
    thr = df.groupby("store_id")["orders"].transform(lambda s: s.quantile(0.75))
    df["high_demand"] = (df["orders"] >= thr).astype(int)
    return df

# ──────────────────────────────────────────────────────────────────────────────
# Datos / estado / sidebar
# ──────────────────────────────────────────────────────────────────────────────
YEAR = 2024
if "seed" not in st.session_state:
    st.session_state.seed = 123
DATA = generate_dataset(YEAR, st.session_state.seed)
ALL_STATES = sorted(DATA["estado"].unique().tolist())
ALL_STORES = DATA[["store_id","store_name"]].drop_duplicates().sort_values("store_name")

with st.sidebar:
    st.markdown('<div class="logo-wrap">', unsafe_allow_html=True)
    st.image(load_image(ASSETS["logo"], fallback=make_brand_fallback), caption="Jaguar Burger MX", use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown('<div class="pill">Presencia: CDMX, Jalisco, Nuevo León, Puebla, Edomex, Yucatán, Baja California, Guanajuato, Querétaro</div>', unsafe_allow_html=True)

    st.divider()
    st.subheader("Filtros")
    estados = st.multiselect("Estados", ALL_STATES, default=ALL_STATES)
    names = ALL_STORES["store_name"].tolist()
    store_sel = st.multiselect("Sucursales", options=names, default=names[:min(6,len(names))])
    dmin, dmax = DATA["date"].min(), DATA["date"].max()
    date_range = st.date_input("Rango de fechas", value=(dmin, dmax), min_value=dmin, max_value=dmax)

    st.divider()
    s1, s2 = st.columns([3,1])
    with s1: seed_val = st.number_input("Semilla aleatoria", value=st.session_state.seed, step=1)
    with s2:
        if st.button("Regenerar"):
            st.session_state.seed = int(seed_val)
            st.cache_data.clear()
            st.rerun()

start_date, end_date = date_range
F = DATA[(DATA["estado"].isin(estados)) &
         (DATA["store_name"].isin(store_sel)) &
         (DATA["date"]>=start_date) & (DATA["date"]<=end_date)].copy()

# ──────────────────────────────────────────────────────────────────────────────
# Portada (logo + hero + descripción + tech logos)
# ──────────────────────────────────────────────────────────────────────────────
top_l, top_r = st.columns([1,2])
with top_l:
    st.image(load_image(ASSETS["logo"], fallback=make_brand_fallback), use_container_width=True)
with top_r:
    st.markdown("### Plataforma de Ventas, Rentabilidad y Modelos Predictivos")
    st.markdown(
        '<p class="tagline">Monitorea KPIs, explora tendencias, crea segmentos y entrena modelos de <b>regresión</b> y <b>clasificación</b> sobre el desempeño diario.</p>',
        unsafe_allow_html=True,
    )

hero_path = load_image(ASSETS["hero"], fallback=None)
if hero_path:
    st.image(hero_path, use_container_width=True)

# Logos de tecnologías (una sola vez)
st.markdown("##### Stack Tecnológico")
tcols = st.columns(len(ASSETS["tech"]))
for c, (label, path) in zip(tcols, ASSETS["tech"].items()):
    c.image(load_image(path, fallback=None) or load_image(ASSETS["logo"], fallback=make_brand_fallback),
            caption=label, use_container_width=True)

st.divider()

# ──────────────────────────────────────────────────────────────────────────────
# KPIs con micro-gráficas
# ──────────────────────────────────────────────────────────────────────────────
last_90 = F.sort_values("date").tail(90)
prev_90 = F.sort_values("date").iloc[-180:-90] if len(F) >= 180 else F.head(0)

sales_now = float(last_90["sales_mxn"].sum()) if len(last_90) else 0.0
sales_prev = float(prev_90["sales_mxn"].sum()) if len(prev_90) else 0.0
orders_now = int(last_90["orders"].sum()) if len(last_90) else 0
orders_prev = int(prev_90["orders"].sum()) if len(prev_90) else 0
margin_now = (last_90["profit_mxn"].sum() / last_90["sales_mxn"].sum() * 100) if sales_now>0 else 0
margin_prev = (prev_90["profit_mxn"].sum() / prev_90["sales_mxn"].sum() * 100) if sales_prev>0 else 0

trend_sales  = last_90.groupby("date")["sales_mxn"].sum().tolist()
trend_orders = last_90.groupby("date")["orders"].sum().tolist()
trend_margin = ((last_90.groupby("date")["profit_mxn"].sum() /
                 last_90.groupby("date")["sales_mxn"].sum()).fillna(0) * 100).tolist()

r1, r2, r3 = st.columns(3)
r1.metric("Ventas 90d (MXN)", f"{sales_now:,.0f}", round(sales_now - sales_prev, 2),
          chart_data=trend_sales, chart_type="area", border=True)
r2.metric("Pedidos 90d", f"{orders_now:,}", orders_now - orders_prev,
          chart_data=trend_orders, chart_type="bar", border=True)
r3.metric("Margen % 90d", f"{margin_now:,.1f}%", f"{(margin_now - margin_prev):+.2f} pp",
          chart_data=trend_margin, chart_type="line", border=True)

# ──────────────────────────────────────────────────────────────────────────────
# Tabs
# ──────────────────────────────────────────────────────────────────────────────
TAB1, TAB2, TAB3, TAB4, TAB5 = st.tabs([
    "📊 Explorador de Datos",
    "🧮 Modelos (Regresión)",
    "✅ Modelos (Clasificación)",
    "🧩 Clustering",
    "🗺️ Mapa & Gemini (ADK)",
])

# Explorador
with TAB1:
    st.subheader("Explorador de ventas y rentabilidad")
    with st.container():
        aggr = st.radio("Agrupar por", ["Mes", "Tienda"], horizontal=True)
    if aggr == "Mes":
        dfm = (
            F.assign(yyyy_mm=lambda d: d["date"].astype(str).str.slice(0,7))
             .groupby(["yyyy_mm"]).agg(
                ventas=("sales_mxn","sum"),
                pedidos=("orders","sum"),
                marketing=("marketing_mxn","sum"),
                utilidades=("profit_mxn","sum"),
                margen=("margin_pct","mean"),
                ticket=("ticket_avg_mxn","mean"),
             ).reset_index()
        )
        fig = px.line(dfm, x="yyyy_mm", y="ventas", markers=True)
        st.plotly_chart(fig, use_container_width=True)
        show_df = dfm
    else:
        dft = (
            F.groupby(["store_name"]).agg(
                ventas=("sales_mxn","sum"),
                pedidos=("orders","sum"),
                marketing=("marketing_mxn","sum"),
                utilidades=("profit_mxn","sum"),
                margen=("margin_pct","mean"),
                ticket=("ticket_avg_mxn","mean"),
            ).reset_index().sort_values("ventas", ascending=False)
        )
        fig = px.bar(dft, x="store_name", y="ventas")
        fig.update_layout(xaxis_title="Sucursal", yaxis_title="Ventas (MXN)")
        st.plotly_chart(fig, use_container_width=True)
        show_df = dft

    st.dataframe(
        show_df, use_container_width=True,
        column_config={
            "ventas": st.column_config.NumberColumn("Ventas (MXN)", format="MXN %,.0f"),
            "marketing": st.column_config.NumberColumn("Marketing (MXN)", format="MXN %,.0f"),
            "utilidades": st.column_config.NumberColumn("Utilidades (MXN)", format="MXN %,.0f"),
            "margen": st.column_config.NumberColumn("Margen %", format="%.2f"),
            "ticket": st.column_config.NumberColumn("Ticket Promedio (MXN)", format="MXN %,.0f"),
        },
        hide_index=True,
    )
    st.download_button("Descargar detalle (CSV)", F.to_csv(index=False).encode("utf-8"),
                       file_name="jaguar_burger_detalle.csv", mime="text/csv")

# Regresión
with TAB2:
    st.subheader("Modelos de Regresión (Ventas Diarias)")
    st.caption("Predicción de ventas por tienda a partir de precio, descuento, marketing, día de la semana y mes.")
    names = ALL_STORES["store_name"].tolist()
    stores_reg = st.multiselect("Tiendas a entrenar", names, default=names[:4])
    REG = F[F["store_name"].isin(stores_reg)].copy()
    if REG.empty:
        st.info("Selecciona al menos una sucursal para entrenar.")
    else:
        REG = REG.sort_values("date")
        cutoff = int(len(REG)*0.8)
        train, test = REG.iloc[:cutoff], REG.iloc[cutoff:]

        num_features = ["marketing_mxn", "discount", "price"]
        cat_features = ["weekday", "month", "store_name"]
        pre = ColumnTransformer([
            ("num", StandardScaler(), num_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_features),
        ])
        pipe = Pipeline([("pre", pre), ("lr", LinearRegression())])

        X_train, y_train = train[num_features+cat_features], train["sales_mxn"]
        X_test,  y_test  = test[num_features+cat_features],  test["sales_mxn"]

        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        r2  = r2_score(y_test, pred)
        rmse = mean_squared_error(y_test, pred, squared=False)

        c1, c2 = st.columns([1,2])
        with c1:
            st.metric("R²", f"{r2:.3f}", border=True)
            st.metric("RMSE (MXN)", f"{rmse:,.0f}", border=True)
        with c2:
            fig_sc = px.scatter(x=y_test, y=pred, labels={"x":"Real", "y":"Predicho"})
            fig_sc.add_trace(go.Scatter(x=[y_test.min(), y_test.max()],
                                        y=[y_test.min(), y_test.max()],
                                        mode="lines", name="45°"))
            st.plotly_chart(fig_sc, use_container_width=True)
        st.caption("Muestra de predicciones")
        st.dataframe(pd.DataFrame({"fecha": test["date"].values, "real": y_test.values, "pred": pred}).head(20),
                     use_container_width=True)

# Clasificación
with TAB3:
    st.subheader("Clasificación (Días de Alta Demanda)")
    st.caption("Clasificador logístico para anticipar días con pedidos por arriba del percentil 75 por tienda.")
    names = ALL_STORES["store_name"].tolist()
    stores_clf = st.multiselect("Tiendas a entrenar", names, default=names[2:8])
    CLF = F[F["store_name"].isin(stores_clf)].copy().sort_values("date")
    if CLF.empty:
        st.info("Selecciona al menos una sucursal para entrenar.")
    else:
        cutoff = int(len(CLF)*0.8)
        train, test = CLF.iloc[:cutoff], CLF.iloc[cutoff:]

        num_features = ["marketing_mxn", "discount", "price"]
        cat_features = ["weekday", "month", "store_name"]
        pre = ColumnTransformer([
            ("num", StandardScaler(), num_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_features),
        ])
        pipe = Pipeline([("pre", pre), ("clf", LogisticRegression(max_iter=200))])

        X_train, y_train = train[num_features+cat_features], train["high_demand"]
        X_test,  y_test  = test[num_features+cat_features],  test["high_demand"]

        pipe.fit(X_train, y_train)
        proba = pipe.predict_proba(X_test)[:,1]
        y_pred = (proba >= 0.5).astype(int)

        acc = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, proba)
        cm = confusion_matrix(y_test, y_pred)

        c1, c2 = st.columns([1,2])
        with c1:
            st.metric("Accuracy", f"{acc:.3f}", border=True)
            st.metric("ROC AUC", f"{auc:.3f}", border=True)
        with c2:
            z = cm
            fig_cm = go.Figure(data=go.Heatmap(z=z, x=["Pred 0","Pred 1"], y=["Real 0","Real 1"],
                                               text=z, texttemplate="%{text}"))
            fig_cm.update_layout(height=360)
            st.plotly_chart(fig_cm, use_container_width=True)

        fpr, tpr, _ = roc_curve(y_test, proba)
        fig_roc = go.Figure()
        fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name="ROC"))
        fig_roc.add_trace(go.Scatter(x=[0,1], y=[0,1], mode="lines", name="Azar"))
        fig_roc.update_layout(title="Curva ROC")
        st.plotly_chart(fig_roc, use_container_width=True)

# Clustering
with TAB4:
    st.subheader("Segmentación de Sucursales (K-Means)")
    agg = (
        F.groupby(["store_id","store_name","estado","ciudad","lat","lon"]).agg(
            ventas_mxn=("sales_mxn","sum"),
            pedidos=("orders","sum"),
            margen=("margin_pct","mean"),
            ticket=("ticket_avg_mxn","mean"),
            marketing=("marketing_mxn","sum"),
        ).reset_index()
    )
    k = st.slider("Número de clusters (K)", 2, 6, 3)
    X = agg[["ventas_mxn","pedidos","margen","ticket","marketing"]].copy()
    scaler = StandardScaler(); Xs = scaler.fit_transform(X)
    kmeans = KMeans(n_clusters=k, n_init=10, random_state=0); labels = kmeans.fit_predict(Xs)
    pca = PCA(n_components=2, random_state=0); coords = pca.fit_transform(Xs)
    agg["cluster"] = labels; agg["pc1"] = coords[:,0]; agg["pc2"] = coords[:,1]

    c1, c2 = st.columns([1.2,1])
    with c1:
        fig_sc = px.scatter(agg, x="pc1", y="pc2", color=agg["cluster"].astype(str),
                            hover_data=["store_name","ventas_mxn","margen","ticket"],
                            labels={"color":"Cluster"})
        st.plotly_chart(fig_sc, use_container_width=True)
    with c2:
        st.dataframe(
            agg[["store_name","estado","ciudad","ventas_mxn","margen","ticket","marketing","cluster"]]
              .sort_values(["cluster","ventas_mxn"], ascending=[True, False]),
            use_container_width=True,
            column_config={
                "ventas_mxn": st.column_config.NumberColumn("Ventas", format="MXN %,.0f"),
                "margen": st.column_config.NumberColumn("Margen %", format="%.2f"),
                "ticket": st.column_config.NumberColumn("Ticket", format="MXN %,.0f"),
                "marketing": st.column_config.NumberColumn("Marketing", format="MXN %,.0f"),
            },
            hide_index=True,
        )

# Mapa + “Gemini (ADK)” (explicación con streaming)
with TAB5:
    st.subheader("Mapa de sucursales y análisis asistido (Gemini — ADK)")

    stores_latest = (
        F.groupby(["store_id","store_name","estado","ciudad","lat","lon"]).agg(
            ventas_mxn=("sales_mxn","sum"),
            pedidos=("orders","sum"),
            margen=("margin_pct","mean"),
            ticket=("ticket_avg_mxn","mean"),
        ).reset_index()
    )
    stores_latest["size"] = (stores_latest["ventas_mxn"]/stores_latest["ventas_mxn"].max())*1000 + 300

    layer = pdk.Layer(
        "ScatterplotLayer",
        data=stores_latest,
        get_position="[lon, lat]",
        get_radius="size",
        get_fill_color="[255, 140, 0]",
        pickable=True, auto_highlight=True,
    )
    view_state = pdk.ViewState(latitude=23.6, longitude=-102.5, zoom=4.1, pitch=30)
    deck = pdk.Deck(layers=[layer], initial_view_state=view_state,
                    tooltip={"text": "{store_name}\nVentas: MXN {ventas_mxn}"})
    event = st.pydeck_chart(deck, on_select="rerun", selection_mode="single-object")

    sel_idx = None
    if event and hasattr(event, "selection") and event.selection:
        row = event.selection[0]
        sel_idx = int(row.get("index", 0)) if isinstance(row, dict) else None

    if sel_idx is not None and 0 <= sel_idx < len(stores_latest):
        selected_store = stores_latest.iloc[sel_idx]
        st.success(f"Sucursal seleccionada: {selected_store['store_name']} — {selected_store['ciudad']}, {selected_store['estado']}")
        F_sel = F[F["store_id"] == selected_store["store_id"]]
    else:
        selected_store = None
        F_sel = F.copy()

    st.markdown("#### Gemini (ADK) — Explicación de desempeño")
    user_goal = st.text_input("Pregunta o enfoque de análisis",
                              value="¿Cómo cerró el último trimestre en ventas y margen?")

    def stream_insight():
        df = F_sel.sort_values("date")
        last_q = df.tail(90); prev_q = df.iloc[-180:-90] if len(df) >= 180 else df.head(0)
        v, vp = last_q["sales_mxn"].sum(), prev_q["sales_mxn"].sum()
        m = (last_q["profit_mxn"].sum() / last_q["sales_mxn"].sum()) if v>0 else 0
        mp = (prev_q["profit_mxn"].sum() / prev_q["sales_mxn"].sum()) if vp>0 else 0
        dv, dm = v - vp, (m - mp)

        text = (f"{user_goal}\n\n"
                f"Últimos 90 días: ventas ≈ MXN {v:,.0f}, margen ≈ {m*100:,.1f}%. "
                f"Var. vs periodo previo: ventas {dv:+,.0f} MXN; margen {dm*100:+.2f} pp.\n")
        for w in text.split(" "):
            yield w + " "; time.sleep(0.01)

        out = pd.DataFrame({"Periodo":["T-1","T"], "Ventas (MXN)":[vp, v], "Margen %":[mp*100, m*100]})
        yield out

        extra = "Sugerencia: intensificar campañas en plazas con ticket alto y margen sobre la media; revisar descuentos >10%."
        for w in extra.split(" "):
            yield w + " "; time.sleep(0.01)

    if st.button("Generar explicación"):
        st.write_stream(stream_insight)

st.divider()
st.caption("© Jaguar Burger MX — Plataforma analítica integral.")
