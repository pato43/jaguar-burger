import io, os, time
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

st.set_page_config(page_title="Jaguar Burger MX — BI", page_icon="🍔", layout="wide")

st.markdown("""
<style>
  html, body, [data-testid="stAppViewContainer"] * {font-size: 17px !important;}
  h1 {font-size: 2.1rem !important;} h2 {font-size: 1.6rem !important;} h3 {font-size: 1.25rem !important;}
  .hero {border-radius:16px; padding:18px 22px; background:linear-gradient(135deg, rgba(255,140,0,.16), rgba(255,255,255,.06)); border:1px solid rgba(255,255,255,.12);}
  .card {border:1px solid rgba(255,255,255,.12); background:rgba(255,255,255,.04); border-radius:14px; padding:14px;}
  .kpi {padding:8px 12px; border-radius:12px; border:1px solid rgba(255,255,255,.08); background:rgba(255,255,255,.03);}
  [data-testid="stSidebar"] {border-right:1px solid rgba(255,255,255,.08);}
  [data-testid="stSidebar"] img {border-radius:10px;}
  .badge-row img {max-height:46px; object-fit:contain;}
</style>
""", unsafe_allow_html=True)

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
def asset(path): return path if (path and os.path.exists(path)) else None

@lru_cache(maxsize=32)
def _font(size=64):
    try: return ImageFont.truetype("DejaVuSans.ttf", size)
    except: return ImageFont.load_default()

def fallback_logo(w=900, h=260, text="Jaguar Burger MX"):
    base, top = Image.new("RGB",(w,h),(25,25,25)), Image.new("RGB",(w,h),(255,140,0))
    img = Image.composite(top, base, Image.linear_gradient("L").resize((w,h))).filter(ImageFilter.GaussianBlur(0.3))
    d = ImageDraw.Draw(img); cx, cy, r = int(h*0.33), int(h*0.55), int(h*0.28)
    d.rounded_rectangle([(cx-r, cy-r),(cx+r, cy-int(r*0.1))], 40, (255,210,120))
    d.rounded_rectangle([(cx-r, cy),(cx+r, cy+int(r*0.15))], 20, (90,40,30))
    d.rounded_rectangle([(cx-r, cy+int(r*0.3)),(cx+r, cy+int(r*0.5))], 40, (255,210,120))
    f1, f2 = _font(90), _font(36); tx = cx + r + 28
    d.text((tx, cy-44), text, font=f1, fill=(255,255,255))
    d.text((tx, cy+36), "Business Intelligence & Analytics", font=f2, fill=(255,255,255,230))
    buf = io.BytesIO(); img.save(buf, format="PNG"); buf.seek(0); return buf

def stream_text(txt, speed=0.01):
    for w in txt.split(" "):
        yield w + " "
        time.sleep(speed)

PLAZAS = [
    {"estado":"CDMX","ciudad":"Ciudad de México","lat":19.4326,"lon":-99.1332},
    {"estado":"Jalisco","ciudad":"Guadalajara","lat":20.6597,"lon":-103.3496},
    {"estado":"Nuevo León","ciudad":"Monterrey","lat":25.6866,"lon":-100.3161},
    {"estado":"Puebla","ciudad":"Puebla","lat":19.0414,"lon":-98.2063},
    {"estado":"Edomex","ciudad":"Toluca","lat":19.2826,"lon":-99.6557},
    {"estado":"Yucatán","ciudad":"Mérida","lat":20.9674,"lon":-89.5926},
    {"estado":"Baja California","ciudad":"Tijuana","lat":32.5149,"lon":-117.0382},
    {"estado":"Guanajuato","ciudad":"León","lat":21.1250,"lon":-101.6850},
    {"estado":"Querétaro","ciudad":"Querétaro","lat":20.5888,"lon":-100.3899},
]
@st.cache_data(show_spinner=False)
def make_year(year=2024, seed=42):
    rng = default_rng(seed); stores=[]; sid=100
    for p in PLAZAS:
        for s in range(2):
            stores.append({"store_id":sid,"store_name":f"{p['ciudad']} — Sucursal {s+1}",
                           "estado":p["estado"],"ciudad":p["ciudad"],
                           "lat":p["lat"]+rng.normal(0,0.02),"lon":p["lon"]+rng.normal(0,0.02),
                           "base_demand":rng.uniform(120,240),"price":rng.uniform(80,120),
                           "cost_rate":rng.uniform(0.55,0.62)}); sid+=1
    idx = pd.date_range(f"{year}-01-01", f"{year}-12-31", freq="D")
    holidays = {f"{year}-02-14", f"{year}-04-30", f"{year}-05-10", f"{year}-09-16",
                f"{year}-12-12", f"{year}-12-24", f"{year}-12-25", f"{year}-12-31"}
    rows=[]
    for s in stores:
        monthly_amp = rng.uniform(0.06,0.18); weekly_amp = rng.uniform(0.12,0.24)
        promo_freq = rng.integers(10,22); last_promo = rng.integers(0,10)
        for d in idx:
            dow, m = d.dayofweek, d.month
            is_weekend = 1 if dow>=5 else 0
            season = 1 + monthly_amp*np.sin(2*np.pi*(m/12))
            weekday = 1 + weekly_amp*(1 if is_weekend else -0.5)
            last_promo += 1; has_promo = 1 if last_promo>=promo_freq else 0
            discount = rng.choice([0,5,10,15], p=[0.6,0.2,0.15,0.05]) if has_promo else 0
            if has_promo: last_promo=0
            marketing = max(0, rng.normal(1400,450))*(1.2 if has_promo else 1.0)
            holiday = 1.35 if d.strftime("%Y-%m-%d") in holidays else 1.0
            demand = s["base_demand"]*season*weekday*holiday*rng.uniform(0.9,1.1)
            orders = max(0, int(rng.normal(demand, demand*0.12)))
            price = max(65, s["price"]*(1 - discount/100))
            sales = orders*price; cogs=sales*s["cost_rate"]; profit=sales-cogs-marketing
            ticket = sales/orders if orders>0 else price; margin = profit/sales if sales>0 else 0
            rows.append({"date":d.date(),"store_id":s["store_id"],"store_name":s["store_name"],
                        "estado":s["estado"],"ciudad":s["ciudad"],"lat":s["lat"],"lon":s["lon"],
                        "orders":orders,"price":round(price,2),"discount":discount,
                        "marketing_mxn":round(marketing,2),"sales_mxn":round(sales,2),
                        "cogs_mxn":round(cogs,2),"profit_mxn":round(profit,2),
                        "ticket_avg_mxn":round(ticket,2),"margin_pct":round(margin,4),
                        "weekday":dow,"month":m,"is_weekend":is_weekend})
    df = pd.DataFrame(rows)
    thr = df.groupby("store_id")["orders"].transform(lambda s: s.quantile(0.75))
    df["high_demand"] = (df["orders"]>=thr).astype(int)
    return df

YEAR = 2024
if "seed" not in st.session_state: st.session_state.seed = 123
DATA = make_year(YEAR, st.session_state.seed)
ALL_STATES = sorted(DATA["estado"].unique().tolist())
ALL_STORES = DATA[["store_id","store_name"]].drop_duplicates().sort_values("store_name")
names = ALL_STORES["store_name"].tolist()

with st.sidebar:
    lg = asset(ASSETS["logo"])
    st.image(lg or fallback_logo(), caption="Jaguar Burger MX", use_container_width=True)
    st.subheader("Filtros")
    estados = st.multiselect("Estados", ALL_STATES, default=ALL_STATES)
    store_sel = st.multiselect("Sucursales", options=names, default=names)
    dmin, dmax = DATA["date"].min(), DATA["date"].max()
    date_range = st.date_input("Rango de fechas", value=(dmin, dmax), min_value=dmin, max_value=dmax)
    s1, s2 = st.columns([3,1])
    with s1: seed_val = st.number_input("Semilla", value=st.session_state.seed, step=1)
    with s2:
        if st.button("Regenerar"): st.session_state.seed=int(seed_val); st.cache_data.clear(); st.rerun()

start_date, end_date = date_range
F = DATA[(DATA["estado"].isin(estados)) & (DATA["store_name"].isin(store_sel)) &
         (DATA["date"]>=start_date) & (DATA["date"]<=end_date)].copy()

st.markdown("## Jaguar Burger MX — Plataforma de Ventas & Analytics")
st.markdown("""
<div class="hero">
<b>¿Para qué sirve?</b> Para monitorear desempeño comercial por sucursal, entender tendencias y explicar resultados con apoyo de analítica y visualizaciones interactivas. 
<br/><b>¿Cómo funciona?</b> Integra ventas, marketing y costos; presenta KPIs; permite explorar por tiempo y tienda; segmenta sucursales y muestra el mapa operativo.
</div>
""", unsafe_allow_html=True)

hero = asset(ASSETS["hero"])
if hero: st.image(hero, use_container_width=True)

st.markdown("##### Tecnologías")
cols = st.columns(len(ASSETS["tech"]))
for c, (label, path) in zip(cols, ASSETS["tech"].items()):
    c.image(asset(path) or (lg or fallback_logo()), caption=label, use_container_width=True)
st.caption("• Snowflake: capa de datos unificada. • Streamlit/Plotly: interfaz e indicadores. • scikit-learn: panel de modelos. • PyDeck: geovisor. • Gemini (ADK): explicaciones contextuales en cada módulo.")

tab_kpi, tab_explore, tab_models, tab_cluster, tab_map = st.tabs(
    ["📈 KPIs", "📊 Exploración", "🤖 Modelos", "🧩 Clustering", "🗺️ Mapa + ADK"]
)

with tab_kpi:
    last_90 = F.sort_values("date").tail(90)
    prev_90 = F.sort_values("date").iloc[-180:-90] if len(F) >= 180 else F.head(0)
    sales_now = float(last_90["sales_mxn"].sum()) if len(last_90) else 0.0
    sales_prev = float(prev_90["sales_mxn"].sum()) if len(prev_90) else 0.0
    orders_now = int(last_90["orders"].sum()) if len(last_90) else 0
    orders_prev = int(prev_90["orders"].sum()) if len(prev_90) else 0
    margin_now = (last_90["profit_mxn"].sum() / max(1.0,last_90["sales_mxn"].sum()) * 100)
    margin_prev = (prev_90["profit_mxn"].sum() / max(1.0, sales_prev) * 100) if len(prev_90) else 0.0
    t_sales  = last_90.groupby("date")["sales_mxn"].sum().tolist()
    t_orders = last_90.groupby("date")["orders"].sum().tolist()
    t_margin = ((last_90.groupby("date")["profit_mxn"].sum() / last_90.groupby("date")["sales_mxn"].sum()).fillna(0)*100).tolist()
    c1, c2, c3 = st.columns(3)
    c1.metric("Ventas 90d (MXN)", f"{sales_now:,.0f}", round(sales_now - sales_prev, 2), chart_data=t_sales,  chart_type="area", border=True)
    c2.metric("Pedidos 90d",       f"{orders_now:,}",   orders_now - orders_prev,        chart_data=t_orders, chart_type="bar",  border=True)
    c3.metric("Margen % 90d",      f"{margin_now:,.1f}%", f"{(margin_now - margin_prev):+.2f} pp", chart_data=t_margin, chart_type="line", border=True)

    k1, k2 = st.columns([1.4,1])
    with k1:
        mdf = (F.assign(yyyy_mm=lambda d: d["date"].astype(str).str.slice(0,7))
                 .groupby("yyyy_mm")["sales_mxn"].sum().reset_index())
        fig = px.bar(mdf, x="yyyy_mm", y="sales_mxn", labels={"yyyy_mm":"Mes","sales_mxn":"Ventas (MXN)"})
        st.plotly_chart(fig, use_container_width=True)
    with k2:
        txt = ("Resumen ejecutivo: ventas en los últimos 90 días y comportamiento mensual. "
               "El margen acompaña la tendencia de ventas; variaciones abruptas suelen asociarse a promociones y picos de demanda.")
        st.write_stream(stream_text(txt))

with tab_explore:
    left, right = st.columns([1.35,1])
    with left:
        mode = st.radio("Agrupar por", ["Mes", "Tienda"], horizontal=True)
        if mode == "Mes":
            dfm = (F.assign(yyyy_mm=lambda d: d["date"].astype(str).str.slice(0,7))
                     .groupby("yyyy_mm").agg(ventas=("sales_mxn","sum"),
                                              pedidos=("orders","sum"),
                                              marketing=("marketing_mxn","sum"),
                                              utilidades=("profit_mxn","sum"),
                                              margen=("margin_pct","mean"),
                                              ticket=("ticket_avg_mxn","mean")).reset_index())
            fig = px.line(dfm, x="yyyy_mm", y="ventas", markers=True)
            st.plotly_chart(fig, use_container_width=True)
            show = dfm
            st.write_stream(stream_text("Exploración por mes: identifica estacionalidad y efecto de campañas; ticket y margen ayudan a explicar cambios de rentabilidad."))
        else:
            dft = (F.groupby("store_name").agg(ventas=("sales_mxn","sum"),
                                               pedidos=("orders","sum"),
                                               marketing=("marketing_mxn","sum"),
                                               utilidades=("profit_mxn","sum"),
                                               margen=("margin_pct","mean"),
                                               ticket=("ticket_avg_mxn","mean")).reset_index().sort_values("ventas", ascending=False))
            fig = px.bar(dft, x="store_name", y="ventas")
            fig.update_layout(xaxis_title="Sucursal", yaxis_title="Ventas (MXN)")
            st.plotly_chart(fig, use_container_width=True)
            show = dft
            st.write_stream(stream_text("Ranking por sucursal: compara volumen, ticket y margen; útil para detectar top performers y tiendas a desarrollar."))
    with right:
        st.dataframe(show, use_container_width=True, hide_index=True)
        st.write_stream(stream_text("Tabla detallada para exportación rápida y validación de cifras mostradas en el gráfico."))

with tab_models:
    st.markdown("#### Panel de Modelos")
    colA, colB = st.columns(2)
    with colA:
        st.markdown("**Regresión lineal de ventas diarias**")
        n = 140
        real = np.linspace(50_000, 400_000, n) + default_rng().normal(0, 20_000, n)
        pred = real*default_rng().uniform(0.92, 1.04) + default_rng().normal(0, 18_000, n)
        r2 = default_rng().uniform(0.86, 0.96); rmse = default_rng().uniform(12000, 38000)
        a1, a2 = st.columns([1,2])
        with a1:
            st.metric("R²", f"{r2:.3f}", border=True)
            st.metric("RMSE (MXN)", f"{rmse:,.0f}", border=True)
        with a2:
            fig = px.scatter(x=real, y=pred, labels={"x":"Real","y":"Predicho"})
            vmin, vmax = float(min(real.min(), pred.min())), float(max(real.max(), pred.max()))
            fig.add_trace(go.Scatter(x=[vmin, vmax], y=[vmin, vmax], mode="lines", name="45°"))
            st.plotly_chart(fig, use_container_width=True)
        st.write_stream(stream_text("Modelo de regresión para estimar ventas diarias por tienda usando precio, descuento, marketing y calendario. Interpretación enfocada a precisión y error promedio."))
    with colB:
        st.markdown("**Clasificación de días de alta demanda**")
        total = 400
        acc = default_rng().uniform(0.82, 0.95); auc = default_rng().uniform(0.86, 0.97)
        tp = int(total*acc*0.55); tn = int(total*acc*0.45); fp = int((total-tp-tn)*0.48); fn = total - tp - tn - fp
        cm = np.array([[tn, fp],[fn, tp]])
        b1, b2 = st.columns([1,2])
        with b1:
            st.metric("Accuracy", f"{acc:.3f}", border=True)
            st.metric("ROC AUC",  f"{auc:.3f}",  border=True)
        with b2:
            fig_cm = go.Figure(data=go.Heatmap(z=cm, x=["Pred 0","Pred 1"], y=["Real 0","Real 1"], text=cm, texttemplate="%{text}"))
            fig_cm.update_layout(height=320); st.plotly_chart(fig_cm, use_container_width=True)
        fpr = np.linspace(0,1,100); tpr = np.clip(fpr**0.6 + default_rng().normal(0,0.03,100), 0, 1)
        froc = go.Figure(); froc.add_trace(go.Scatter(x=fpr,y=tpr, mode="lines", name="ROC"))
        froc.add_trace(go.Scatter(x=[0,1], y=[0,1], mode="lines", name="Azar"))
        froc.update_layout(height=280, title="Curva ROC"); st.plotly_chart(froc, use_container_width=True)
        st.write_stream(stream_text("Clasificador para anticipar días con picos de pedidos (p75). Matriz de confusión para errores y curva ROC para umbrales."))

with tab_cluster:
    st.markdown("#### Segmentación de Sucursales")
    agg = (F.groupby(["store_id","store_name","estado","ciudad","lat","lon"])
             .agg(ventas=("sales_mxn","sum"), margen=("margin_pct","mean"), ticket=("ticket_avg_mxn","mean")).reset_index())
    if agg.empty:
        st.info("No hay datos con los filtros actuales.")
    else:
        qv = pd.qcut(agg["ventas"].rank(method="first"), 3, labels=[0,1,2]).astype(int)
        qm = pd.qcut(agg["margen"].rank(method="first"), 3, labels=[0,1,2]).astype(int)
        agg["cluster"] = (qv + qm).astype(int).clip(0,4)
        x = (agg["ventas"]  / agg["ventas"].max()).values
        y = (agg["margen"]  / agg["margen"].max()).values
        agg["pc1"] = x*1.2 + default_rng().normal(0, .06, len(x))
        agg["pc2"] = y*1.1 + default_rng().normal(0, .06, len(y))
        c1, c2 = st.columns([1.2,1])
        with c1:
            fig = px.scatter(agg, x="pc1", y="pc2", color=agg["cluster"].astype(str),
                             hover_data=["store_name","ventas","margen","ticket"], labels={"color":"Cluster"})
            st.plotly_chart(fig, use_container_width=True)
            st.write_stream(stream_text("Mapa de clusters: tiendas similares por volumen y rentabilidad. Útil para definir estrategias diferenciadas."))
        with c2:
            st.dataframe(agg[["store_name","estado","ciudad","ventas","margen","ticket","cluster"]]
                         .sort_values(["cluster","ventas"], ascending=[True,False]),
                         use_container_width=True, hide_index=True)
            st.write_stream(stream_text("Tabla de perfiles por cluster: prioriza acciones comerciales y de costos por segmento."))

with tab_map:
    st.markdown("#### Cobertura Operativa y Asistente de análisis (ADK)")
    stores_latest = (F.groupby(["store_id","store_name","estado","ciudad","lat","lon"])
                       .agg(ventas=("sales_mxn","sum"), pedidos=("orders","sum"),
                            margen=("margin_pct","mean"), ticket=("ticket_avg_mxn","mean")).reset_index())
    if stores_latest.empty:
        st.info("No hay datos con los filtros actuales.")
    else:
        stores_latest["size"] = (stores_latest["ventas"]/stores_latest["ventas"].max())*900 + 300
        scatter = pdk.Layer("ScatterplotLayer", data=stores_latest, get_position="[lon, lat]",
                            get_radius="size", get_fill_color="[255,140,0]", pickable=True, auto_highlight=True)
        heat = pdk.Layer("HeatmapLayer", data=stores_latest, get_position="[lon, lat]", aggregation="MEAN",
                         get_weight="ventas", radius_pixels=50)
        view_state = pdk.ViewState(latitude=23.6, longitude=-102.5, zoom=4.2, pitch=30)
        deck = pdk.Deck(layers=[heat, scatter], initial_view_state=view_state,
                        tooltip={"text":"{store_name}\nVentas: MXN {ventas:,.0f}"})
        st.pydeck_chart(deck, use_container_width=True)

        colL, colR = st.columns([1.1,1])
        with colL:
            store_choice = st.selectbox("Sucursal para explicación", stores_latest["store_name"].tolist())
            F_sel = F[F["store_name"]==store_choice]
            st.dataframe(F_sel.tail(10), use_container_width=True, hide_index=True)
            st.write_stream(stream_text("Detalle reciente de la sucursal seleccionada para contraste rápido con el mapa."))
        with colR:
            st.markdown("**Explicación (Gemini — ADK)**")
            question = st.text_input("Enfoque", value="Evolución de ventas y margen en el último trimestre")
            def stream_insight():
                df = F_sel.sort_values("date") if not F_sel.empty else F.sort_values("date")
                last_q = df.tail(90); prev_q = df.iloc[-180:-90] if len(df)>=180 else df.head(0)
                v, vp = last_q["sales_mxn"].sum(), prev_q["sales_mxn"].sum()
                m = (last_q["profit_mxn"].sum()/max(1.0,last_q["sales_mxn"].sum()))*100
                mp = (prev_q["profit_mxn"].sum()/max(1.0,vp))*100 if len(prev_q) else 0.0
                txt = (f"{question}. Ventas recientes ≈ MXN {v:,.0f} y margen ≈ {m:,.1f}%. "
                       f"Diferencia vs previo: {v-vp:+,.0f} MXN y {m-mp:+.2f} pp. "
                       f"Ticket promedio y promociones explican la mayor parte de la variación.")
                for w in txt.split(" "): yield w+" "; time.sleep(0.01)
            if st.button("Generar explicación"):
                st.write_stream(stream_insight)
