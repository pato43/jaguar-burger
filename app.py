# Jaguar Burger MX — BI & Analytics App (Streamlit)
# ------------------------------------------------
# Cómo ejecutar:
#   1) Guarda este archivo como `app.py`
#   2) (Opcional) Crea .streamlit/config.toml con modo auto:
#        [theme]\nbase="auto"\n
#   3) Instala dependencias mínimas: 
#        pip install streamlit scikit-learn plotly pydeck pillow pandas numpy
#   4) Ejecuta: 
#        streamlit run app.py

import io
import time
from datetime import datetime, date
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

# Ajustes de tipografía y detalles visuales para verse bien en modo oscuro
st.markdown(
    """
    <style>
      html, body, [data-testid="stAppViewContainer"] * {font-size: 17px !important;}
      h1, .stMarkdown h1 {font-size: 2.0rem !important;}
      h2, .stMarkdown h2 {font-size: 1.6rem !important;}
      h3, .stMarkdown h3 {font-size: 1.25rem !important;}
      .metric-row .stMetric {padding: 0.25rem 0.5rem;}
      .tech-badges img {margin-right: 10px;}
      .tagline {opacity: .9}
      .pill {display:inline-block;padding:6px 10px;border-radius:999px;}
      .soft {background:rgba(255,255,255,.06);border:1px solid rgba(255,255,255,.1)}
    </style>
    """,
    unsafe_allow_html=True,
)

# ──────────────────────────────────────────────────────────────────────────────
# Utilidades de imagen (logo / badges) sin archivos externos
# ──────────────────────────────────────────────────────────────────────────────
@lru_cache(maxsize=32)
def _load_font(size: int = 64):
    # Usa la fuente por defecto del sistema si no hay TTF disponible
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except Exception:
        return ImageFont.load_default()


def _linear_gradient(w, h, c1, c2):
    base = Image.new("RGB", (w, h), c1)
    top = Image.new("RGB", (w, h), c2)
    mask = Image.linear_gradient("L").resize((w, h))
    return Image.composite(top, base, mask)


def make_brand_logo_png(text="Jaguar Burger MX", w=900, h=300):
    # Colores: naranja/ámbar y oscuro
    bg = _linear_gradient(w, h, (20, 20, 20), (245, 120, 37))
    bg = bg.filter(ImageFilter.GaussianBlur(radius=0.3))
    draw = ImageDraw.Draw(bg)

    # Ícono simple de hamburguesa
    cx, cy = int(h * 0.5), int(h * 0.50)
    r = int(h * 0.28)
    bun_top = [(cx - r, cy - r), (cx + r, cy - int(r * 0.1))]
    patty = [(cx - r, cy), (cx + r, cy + int(r * 0.15))]
    bun_bottom = [(cx - r, cy + int(r * 0.3)), (cx + r, cy + int(r * 0.5))]
    sesame = [(cx - int(r*0.6), cy - int(r*0.6)), (cx - int(r*0.2), cy - int(r*0.65)), (cx + int(r*0.1), cy - int(r*0.55)), (cx + int(r*0.5), cy - int(r*0.62))]

    draw.rounded_rectangle(bun_top, radius=40, fill=(255, 210, 120))
    draw.rounded_rectangle(patty, radius=20, fill=(90, 40, 30))
    draw.rounded_rectangle(bun_bottom, radius=40, fill=(255, 210, 120))
    for sx, sy in sesame:
        draw.ellipse((sx-8, sy-8, sx+8, sy+8), fill=(250, 235, 190))

    # Texto de marca
    f1 = _load_font(100)
    f2 = _load_font(40)
    text_x = cx + r + 30
    draw.text((text_x, cy - 40), text, font=f1, fill=(255, 255, 255))
    draw.text((text_x, cy + 50), "Business Intelligence & Analytics", font=f2, fill=(255, 255, 255, 230))

    buf = io.BytesIO()
    bg.save(buf, format="PNG")
    buf.seek(0)
    return buf


def make_badge_png(label: str, fg=(255, 255, 255), bg=(35, 35, 35)):
    pad_x, pad_y = 24, 10
    f = _load_font(28)
    w, h = f.getlength(label), 28
    img = Image.new("RGBA", (int(w) + pad_x * 2, h + pad_y * 2), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    draw.rounded_rectangle([0, 0, img.size[0], img.size[1]], radius=16, fill=bg)
    draw.text((pad_x, pad_y), label, font=f, fill=fg)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return buf


# ──────────────────────────────────────────────────────────────────────────────
# Catálogo de plazas y sucursales
# ──────────────────────────────────────────────────────────────────────────────
PLAZAS = [
    {"estado": "CDMX", "ciudad": "Ciudad de México", "lat": 19.4326, "lon": -99.1332},
    {"estado": "Jalisco", "ciudad": "Guadalajara", "lat": 20.6597, "lon": -103.3496},
    {"estado": "Nuevo León", "ciudad": "Monterrey", "lat": 25.6866, "lon": -100.3161},
    {"estado": "Puebla", "ciudad": "Puebla", "lat": 19.0414, "lon": -98.2063},
    {"estado": "Edomex", "ciudad": "Toluca", "lat": 19.2826, "lon": -99.6557},
    {"estado": "Yucatán", "ciudad": "Mérida", "lat": 20.9674, "lon": -89.5926},
    {"estado": "Baja California", "ciudad": "Tijuana", "lat": 32.5149, "lon": -117.0382},
    {"estado": "Guanajuato", "ciudad": "León", "lat": 21.1250, "lon": -101.6850},
    {"estado": "Querétaro", "ciudad": "Querétaro", "lat": 20.5888, "lon": -100.3899},
]


# ──────────────────────────────────────────────────────────────────────────────
# Generación de datos (un año)
# ──────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def generate_dataset(year: int = 2024, seed: int = 42) -> pd.DataFrame:
    rng = default_rng(seed)
    stores = []

    # Dos sucursales por ciudad
    store_id = 100
    for plaza in PLAZAS:
        for s in range(2):
            name = f"{plaza['ciudad']} — Sucursal {s+1}"
            stores.append({
                "store_id": store_id,
                "store_name": name,
                "estado": plaza["estado"],
                "ciudad": plaza["ciudad"],
                "lat": plaza["lat"] + rng.normal(0, 0.02),
                "lon": plaza["lon"] + rng.normal(0, 0.02),
                # Parámetros base de la tienda
                "base_demand": rng.uniform(120, 240),        # pedidos/día base
                "price": rng.uniform(80, 120),               # MXN
                "cost_rate": rng.uniform(0.55, 0.62),        # % costo de ingredientes
            })
            store_id += 1

    idx = pd.date_range(start=f"{year}-01-01", end=f"{year}-12-31", freq="D")
    rows = []
    holidays = set([
        f"{year}-02-14", f"{year}-04-30", f"{year}-05-10", f"{year}-09-16",
        f"{year}-12-12", f"{year}-12-24", f"{year}-12-25", f"{year}-12-31",
    ])

    for st_info in stores:
        # Componentes estacionales mensuales (alto verano/diciembre)
        monthly_amp = rng.uniform(0.06, 0.18)
        weekly_amp = rng.uniform(0.12, 0.24)
        promo_freq = rng.integers(10, 22)  # cada ~x días
        last_promo = rng.integers(0, 10)

        for d in idx:
            dow = d.dayofweek  # 0=Lunes
            month = d.month
            is_weekend = 1 if dow >= 5 else 0

            season = 1 + monthly_amp * np.sin(2 * np.pi * (month / 12.0))
            weekday_boost = 1 + weekly_amp * (1 if is_weekend else -0.5)

            # Promociones cada cierto tiempo
            last_promo += 1
            has_promo = 1 if last_promo >= promo_freq else 0
            discount = rng.choice([0, 5, 10, 15], p=[0.6, 0.2, 0.15, 0.05]) if has_promo else 0
            if has_promo:
                last_promo = 0

            # Gasto de marketing variable
            marketing = max(0, rng.normal(1400, 450)) * (1.2 if has_promo else 1.0)

            # Eventos especiales / días pico
            holiday_boost = 1.0
            if d.strftime('%Y-%m-%d') in holidays:
                holiday_boost = 1.35

            demand = (
                st_info["base_demand"]
                * season
                * weekday_boost
                * holiday_boost
                * rng.uniform(0.90, 1.10)
            )

            # Pedidos discretos
            orders = max(0, int(rng.normal(demand, demand * 0.12)))
            price = max(65, st_info["price"] * (1 - discount / 100))
            sales = orders * price
            cogs = sales * st_info["cost_rate"]
            profit = sales - cogs - marketing
            ticket_avg = sales / orders if orders > 0 else price
            margin_pct = profit / sales if sales > 0 else 0

            rows.append({
                "date": d.date(),
                "store_id": st_info["store_id"],
                "store_name": st_info["store_name"],
                "estado": st_info["estado"],
                "ciudad": st_info["ciudad"],
                "lat": st_info["lat"],
                "lon": st_info["lon"],
                "orders": orders,
                "price": round(price, 2),
                "discount": discount,
                "marketing_mxn": round(marketing, 2),
                "sales_mxn": round(sales, 2),
                "cogs_mxn": round(cogs, 2),
                "profit_mxn": round(profit, 2),
                "ticket_avg_mxn": round(ticket_avg, 2),
                "margin_pct": round(margin_pct, 4),
                "weekday": dow,
                "month": month,
                "is_weekend": is_weekend,
            })

    df = pd.DataFrame(rows)

    # Etiqueta binaria para modelado logístico (días de alta demanda, p75 por tienda)
    high_thr = df.groupby("store_id")["orders"].transform(lambda s: s.quantile(0.75))
    df["high_demand"] = (df["orders"] >= high_thr).astype(int)
    return df


# ──────────────────────────────────────────────────────────────────────────────
# Datos y estado
# ──────────────────────────────────────────────────────────────────────────────
YEAR_DEFAULT = 2024
if "seed" not in st.session_state:
    st.session_state.seed = 123

DATA = generate_dataset(year=YEAR_DEFAULT, seed=st.session_state.seed)
ALL_STATES = sorted(DATA["estado"].unique().tolist())
ALL_STORES = DATA[["store_id", "store_name"]].drop_duplicates().sort_values("store_name")

# ──────────────────────────────────────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image(make_brand_logo_png(), caption="Jaguar Burger MX", use_column_width=True)
    st.markdown("<div class='pill soft'>Presencia: CDMX, Jalisco, Nuevo León, Puebla, Edomex, Yucatán, Baja California, Guanajuato, Querétaro</div>", unsafe_allow_html=True)

    st.divider()
    st.subheader("Filtros")
    year = st.selectbox("Año", [YEAR_DEFAULT], index=0)
    estados = st.multiselect("Estados", ALL_STATES, default=ALL_STATES)

    stores_filtered = ALL_STORES[ALL_STORES["store_name"].str.contains("|")]
    store_names = stores_filtered["store_name"].tolist()
    store_sel = st.multiselect(
        "Sucursales",
        options=store_names,
        default=store_names[: min(6, len(store_names))],
    )

    date_min, date_max = DATA["date"].min(), DATA["date"].max()
    date_range = st.date_input(
        "Rango de fechas",
        value=(date_min, date_max),
        min_value=date_min,
        max_value=date_max,
    )

    st.divider()
    st.caption("Semilla aleatoria (para regenerar datos)")
    col_seed = st.columns([3, 1])
    with col_seed[0]:
        seed_input = st.number_input("seed", value=st.session_state.seed, step=1)
    with col_seed[1]:
        if st.button("Regenerar"):
            st.session_state.seed = int(seed_input)
            st.cache_data.clear()
            st.rerun()

# Subconjunto de datos según filtros
start_date, end_date = date_range
F = DATA[(DATA["estado"].isin(estados)) & (DATA["store_name"].isin(store_sel)) & (DATA["date"] >= start_date) & (DATA["date"] <= end_date)].copy()

# ──────────────────────────────────────────────────────────────────────────────
# Encabezado / Portada
# ──────────────────────────────────────────────────────────────────────────────
logo_col, title_col = st.columns([1, 2])
with logo_col:
    st.image(make_brand_logo_png(), use_column_width=True)
with title_col:
    st.markdown("### Plataforma de Ventas, Rentabilidad y Modelos Predictivos")
    st.markdown(
        "<p class='tagline'>Monitorea KPIs, explora tendencias, crea segmentos y entrena modelos de <b>regresión</b> y <b>clasificación</b> sobre el desempeño diario de cada sucursal.</p>",
        unsafe_allow_html=True,
    )

# Mini logos de tecnologías (solo una vez, en la pantalla principal)
tech_row = st.container()
with tech_row:
    st.markdown("**Stack Tecnológico**")
    bcols = st.columns(6)
    badges = [
        ("Streamlit", (230, 0, 80)),
        ("scikit-learn", (252, 146, 31)),
        ("Plotly", (0, 120, 210)),
        ("PyDeck", (0, 170, 140)),
        ("Snowflake", (0, 180, 225)),
        ("Gemini (ADK)", (80, 100, 255)),
    ]
    for col, (name, color) in zip(bcols, badges):
        with col:
            col.image(make_badge_png(name, bg=(color[0], color[1], color[2],)), use_column_width=True)

st.divider()

# ──────────────────────────────────────────────────────────────────────────────
# KPIs rápidos con micro-gráficas (nuevas APIs de st.metric)
# ──────────────────────────────────────────────────────────────────────────────
last_90 = F.sort_values("date").tail(90)
prev_90 = F.sort_values("date").iloc[-180:-90]

sales_now = float(last_90["sales_mxn"].sum()) if len(last_90) else 0.0
sales_prev = float(prev_90["sales_mxn"].sum()) if len(prev_90) else 0.0
sales_delta = round((sales_now - sales_prev), 2)

orders_now = int(last_90["orders"].sum()) if len(last_90) else 0
orders_prev = int(prev_90["orders"].sum()) if len(prev_90) else 0
orders_delta = orders_now - orders_prev

margin_now = (
    (last_90["profit_mxn"].sum() / last_90["sales_mxn"].sum()) * 100 if last_90["sales_mxn"].sum() > 0 else 0
)
margin_prev = (
    (prev_90["profit_mxn"].sum() / prev_90["sales_mxn"].sum()) * 100 if sales_prev > 0 else 0
)
margin_delta = round(margin_now - margin_prev, 2)

trend_sales = last_90.groupby("date")["sales_mxn"].sum().tolist()
trend_orders = last_90.groupby("date")["orders"].sum().tolist()
trend_margin = (
    (last_90.groupby("date")["profit_mxn"].sum() / last_90.groupby("date")["sales_mxn"].sum()).fillna(0) * 100
).tolist()

row = st.container(border=False, height=None)
with row:
    r1, r2, r3 = st.columns(3)
    with r1:
        st.metric("Ventas 90d (MXN)", f"{sales_now:,.0f}", sales_delta, chart_data=trend_sales, chart_type="area", border=True)
    with r2:
        st.metric("Pedidos 90d", f"{orders_now:,}", orders_delta, chart_data=trend_orders, chart_type="bar", border=True)
    with r3:
        st.metric("Margen % 90d", f"{margin_now:,.1f}%", f"{margin_delta:+.2f} pp", chart_data=trend_margin, chart_type="line", border=True)

# ──────────────────────────────────────────────────────────────────────────────
# Tabs principales
# ──────────────────────────────────────────────────────────────────────────────
TAB1, TAB2, TAB3, TAB4, TAB5 = st.tabs([
    "📊 Explorador de Datos",
    "🧮 Modelos (Regresión)",
    "✅ Modelos (Clasificación)",
    "🧩 Clustering",
    "🗺️ Mapa & Gemini (ADK)",
])

# ----------------------------------------------------------------------------
# Explorador de datos
# ----------------------------------------------------------------------------
with TAB1:
    st.subheader("Explorador de ventas y rentabilidad")

    # Resumen por mes/tienda
    aggr_type = st.radio("Agrupar por", ["Mes", "Tienda"], horizontal=True)
    if aggr_type == "Mes":
        dfm = (
            F.assign(yyyy_mm=lambda d: d["date"].astype(str).str.slice(0, 7))
             .groupby(["yyyy_mm"]).agg(
                 ventas=("sales_mxn", "sum"),
                 pedidos=("orders", "sum"),
                 marketing=("marketing_mxn", "sum"),
                 utilidades=("profit_mxn", "sum"),
                 margen=("margin_pct", "mean"),
                 ticket=("ticket_avg_mxn", "mean"),
             ).reset_index()
        )
        fig = px.line(dfm, x="yyyy_mm", y="ventas", markers=True)
        st.plotly_chart(fig, use_container_width=True)
        show_df = dfm
    else:
        dft = (
            F.groupby(["store_name"]).agg(
                ventas=("sales_mxn", "sum"),
                pedidos=("orders", "sum"),
                marketing=("marketing_mxn", "sum"),
                utilidades=("profit_mxn", "sum"),
                margen=("margin_pct", "mean"),
                ticket=("ticket_avg_mxn", "mean"),
            ).reset_index().sort_values("ventas", ascending=False)
        )
        fig = px.bar(dft, x="store_name", y="ventas")
        fig.update_layout(xaxis_title="Sucursal", yaxis_title="Ventas (MXN)")
        st.plotly_chart(fig, use_container_width=True)
        show_df = dft

    st.dataframe(
        show_df,
        use_container_width=True,
        column_config={
            "ventas": st.column_config.NumberColumn("Ventas (MXN)", format="MXN %,.0f"),
            "marketing": st.column_config.NumberColumn("Marketing (MXN)", format="MXN %,.0f"),
            "utilidades": st.column_config.NumberColumn("Utilidades (MXN)", format="MXN %,.0f"),
            "margen": st.column_config.NumberColumn("Margen %", format="%.2f"),
            "ticket": st.column_config.NumberColumn("Ticket Promedio (MXN)", format="MXN %,.0f"),
        },
        hide_index=True,
    )

    csv = F.to_csv(index=False).encode("utf-8")
    st.download_button("Descargar detalle (CSV)", csv, file_name="jaguar_burger_detalle.csv", mime="text/csv")

# ----------------------------------------------------------------------------
# Modelos — Regresión lineal (ventas)
# ----------------------------------------------------------------------------
with TAB2:
    st.subheader("Modelos de Regresión (Ventas Diarias)")
    st.caption("Predicción de ventas por tienda a partir de precio, descuento, marketing, día de la semana y mes.")

    stores_reg = st.multiselect("Tiendas a entrenar", store_names, default=store_names[:4])
    REG = F[F["store_name"].isin(stores_reg)].copy()

    if REG.empty:
        st.info("Selecciona al menos una sucursal para entrenar.")
    else:
        # Separación temporal 80/20 (train primeros 80% de días)
        REG = REG.sort_values("date")
        cutoff = int(len(REG) * 0.8)
        train, test = REG.iloc[:cutoff], REG.iloc[cutoff:]

        num_features = ["marketing_mxn", "discount", "price"]
        cat_features = ["weekday", "month", "store_name"]

        pre = ColumnTransformer([
            ("num", StandardScaler(), num_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_features),
        ])
        pipe = Pipeline([
            ("pre", pre),
            ("lr", LinearRegression()),
        ])

        X_train = train[num_features + cat_features]
        y_train = train["sales_mxn"]
        X_test = test[num_features + cat_features]
        y_test = test["sales_mxn"]

        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)

        r2 = r2_score(y_test, pred)
        rmse = mean_squared_error(y_test, pred, squared=False)

        c1, c2 = st.columns([1, 2])
        with c1:
            st.metric("R²", f"{r2:.3f}", border=True)
            st.metric("RMSE (MXN)", f"{rmse:,.0f}", border=True)

        with c2:
            fig_sc = px.scatter(x=y_test, y=pred, labels={"x": "Real", "y": "Predicho"})
            fig_sc.add_trace(go.Scatter(x=[y_test.min(), y_test.max()], y=[y_test.min(), y_test.max()], mode="lines", name="45°"))
            st.plotly_chart(fig_sc, use_container_width=True)

        st.caption("Muestra de predicciones")
        preview = pd.DataFrame({"fecha": test["date"].values, "real": y_test.values, "pred": pred}).head(20)
        st.dataframe(preview, use_container_width=True)

# ----------------------------------------------------------------------------
# Modelos — Clasificación (Logística)
# ----------------------------------------------------------------------------
with TAB3:
    st.subheader("Clasificación (Días de Alta Demanda)")
    st.caption("Clasificador logístico para anticipar días con pedidos por arriba del percentil 75 por tienda.")

    stores_clf = st.multiselect("Tiendas a entrenar", store_names, default=store_names[2:8])
    CLF = F[F["store_name"].isin(stores_clf)].copy().sort_values("date")

    if CLF.empty:
        st.info("Selecciona al menos una sucursal para entrenar.")
    else:
        cutoff = int(len(CLF) * 0.8)
        train, test = CLF.iloc[:cutoff], CLF.iloc[cutoff:]

        num_features = ["marketing_mxn", "discount", "price"]
        cat_features = ["weekday", "month", "store_name"]

        pre = ColumnTransformer([
            ("num", StandardScaler(), num_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_features),
        ])
        pipe = Pipeline([
            ("pre", pre),
            ("clf", LogisticRegression(max_iter=200)),
        ])

        X_train = train[num_features + cat_features]
        y_train = train["high_demand"]
        X_test = test[num_features + cat_features]
        y_test = test["high_demand"]

        pipe.fit(X_train, y_train)
        proba = pipe.predict_proba(X_test)[:, 1]
        y_pred = (proba >= 0.5).astype(int)

        acc = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, proba)
        cm = confusion_matrix(y_test, y_pred)

        c1, c2 = st.columns([1, 2])
        with c1:
            st.metric("Accuracy", f"{acc:.3f}", border=True)
            st.metric("ROC AUC", f"{auc:.3f}", border=True)
        with c2:
            z = cm
            fig_cm = go.Figure(data=go.Heatmap(z=z, x=["Pred 0", "Pred 1"], y=["Real 0", "Real 1"], text=z, texttemplate="%{text}"))
            fig_cm.update_layout(height=360)
            st.plotly_chart(fig_cm, use_container_width=True)

        fpr, tpr, thr = roc_curve(y_test, proba)
        fig_roc = go.Figure()
        fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name="ROC"))
        fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Azar"))
        fig_roc.update_layout(title="Curva ROC")
        st.plotly_chart(fig_roc, use_container_width=True)

# ----------------------------------------------------------------------------
# Clustering de sucursales
# ----------------------------------------------------------------------------
with TAB4:
    st.subheader("Segmentación de Sucursales (K-Means)")
    agg = (
        F.groupby(["store_id", "store_name", "estado", "ciudad", "lat", "lon"]).agg(
            ventas_mxn=("sales_mxn", "sum"),
            pedidos=("orders", "sum"),
            margen=("margin_pct", "mean"),
            ticket=("ticket_avg_mxn", "mean"),
            marketing=("marketing_mxn", "sum"),
        ).reset_index()
    )

    k = st.slider("Número de clusters (K)", 2, 6, 3)

    X = agg[["ventas_mxn", "pedidos", "margen", "ticket", "marketing"]].copy()
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=k, n_init=10, random_state=0)
    labels = kmeans.fit_predict(Xs)

    pca = PCA(n_components=2, random_state=0)
    coords = pca.fit_transform(Xs)

    agg["cluster"] = labels
    agg["pc1"] = coords[:, 0]
    agg["pc2"] = coords[:, 1]

    c1, c2 = st.columns([1.2, 1])
    with c1:
        fig_sc = px.scatter(
            agg, x="pc1", y="pc2", color=agg["cluster"].astype(str), hover_data=["store_name", "ventas_mxn", "margen", "ticket"],
            labels={"color": "Cluster"}
        )
        st.plotly_chart(fig_sc, use_container_width=True)
    with c2:
        st.dataframe(
            agg[["store_name", "estado", "ciudad", "ventas_mxn", "margen", "ticket", "marketing", "cluster"]]
              .sort_values(["cluster", "ventas_mxn"], ascending=[True, False]),
            use_container_width=True,
            column_config={
                "ventas_mxn": st.column_config.NumberColumn("Ventas", format="MXN %,.0f"),
                "margen": st.column_config.NumberColumn("Margen %", format="%.2f"),
                "ticket": st.column_config.NumberColumn("Ticket", format="MXN %,.0f"),
                "marketing": st.column_config.NumberColumn("Marketing", format="MXN %,.0f"),
            },
            hide_index=True,
        )

# ----------------------------------------------------------------------------
# Mapa + Gemini (ADK) — selección en mapa + explicación tipo chat
# ----------------------------------------------------------------------------
with TAB5:
    st.subheader("Mapa de sucursales y análisis asistido (Gemini — ADK)")

    # Vista de mapa con selección
    stores_latest = (
        F.groupby(["store_id", "store_name", "estado", "ciudad", "lat", "lon"]).agg(
            ventas_mxn=("sales_mxn", "sum"),
            pedidos=("orders", "sum"),
            margen=("margin_pct", "mean"),
            ticket=("ticket_avg_mxn", "mean"),
        ).reset_index()
    )
    stores_latest["size"] = (stores_latest["ventas_mxn"] / stores_latest["ventas_mxn"].max()) * 1000 + 300

    layer = pdk.Layer(
        "ScatterplotLayer",
        data=stores_latest,
        get_position="[lon, lat]",
        get_radius="size",
        get_fill_color="[255, 140, 0]",
        pickable=True,
        auto_highlight=True,
    )
    view_state = pdk.ViewState(latitude=23.6, longitude=-102.5, zoom=4.1, pitch=30)
    deck = pdk.Deck(layers=[layer], initial_view_state=view_state, tooltip={"text": "{store_name}\nVentas: MXN {ventas_mxn}"})

    event = st.pydeck_chart(deck, on_select="rerun", selection_mode="single-object")

    sel_idx = None
    if event and hasattr(event, "selection") and event.selection:
        # Cuando hay selección, filtra por tienda
        row = event.selection[0]
        sel_idx = int(row.get("index", 0)) if isinstance(row, dict) else None

    if sel_idx is not None and 0 <= sel_idx < len(stores_latest):
        selected_store = stores_latest.iloc[sel_idx]
        st.success(f"Sucursal seleccionada: {selected_store['store_name']} — {selected_store['ciudad']}, {selected_store['estado']}")
        F_sel = F[F["store_id"] == selected_store["store_id"]]
    else:
        selected_store = None
        F_sel = F.copy()

    # Panel: explicación tipo chat con streaming
    st.markdown("#### Gemini (ADK) — Explicación de desempeño")
    user_goal = st.text_input("Pregunta o enfoque de análisis", value="¿Cómo cerró el último trimestre en ventas y margen?")

    def stream_insight():
        # Construye un pequeño insight con datos agregados recientes
        df = F_sel.sort_values("date")
        last_q = df.tail(90)
        prev_q = df.iloc[-180:-90]
        v, vp = last_q["sales_mxn"].sum(), prev_q["sales_mxn"].sum()
        m = (last_q["profit_mxn"].sum() / last_q["sales_mxn"].sum()) if last_q["sales_mxn"].sum() > 0 else 0
        mp = (prev_q["profit_mxn"].sum() / prev_q["sales_mxn"].sum()) if vp > 0 else 0
        dv = v - vp
        dm = m - mp

        text = (
            f"{user_goal}\n\n"
            f"Últimos 90 días: ventas ≈ MXN {v:,.0f}, margen ≈ {m*100:,.1f}%. "
            f"Var. vs periodo previo: ventas {dv:+,.0f} MXN; margen {dm*100:+.2f} pp.\n"
        )
        for word in text.split(" "):
            yield word + " "
            time.sleep(0.01)

        # Pequeña tabla de apoyo
        out = pd.DataFrame({
            "Periodo": ["T-1", "T"],
            "Ventas (MXN)": [vp, v],
            "Margen %": [mp * 100, m * 100],
        })
        yield out

        extra = "Sugerencia: intensificar campañas en plazas con ticket alto y margen superior a la media; revisar descuentos >10% que erosionan rentabilidad."
        for word in extra.split(" "):
            yield word + " "
            time.sleep(0.01)

    if st.button("Generar explicación"):
        st.write_stream(stream_insight)

# ──────────────────────────────────────────────────────────────────────────────
# Pie de página
# ──────────────────────────────────────────────────────────────────────────────
st.divider()
st.caption("© Jaguar Burger MX — Plataforma analítica integral para ventas y rentabilidad.")
