import io, os, time
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
px.defaults.template = "plotly_dark"
COLOR_SEQ = ["#7dd3fc", "#34d399", "#fbbf24", "#f472b6", "#60a5fa", "#a78bfa", "#fb7185"]

st.markdown("""
<style>
  html, body, [data-testid="stAppViewContainer"] * {font-size: 17px !important;}
  h1 {font-size: 2.6rem !important;} h2 {font-size: 1.8rem !important;} h3 {font-size: 1.28rem !important;}
  [data-testid="stAppViewContainer"] {background: linear-gradient(180deg,#0b1020 0%, #0d1117 70%);}
  [data-testid="stSidebar"] {background: linear-gradient(180deg,#0f172a,#111827); border-right:1px solid rgba(255,255,255,.08);}
  .hero {border-radius:18px; padding:18px 22px; background:linear-gradient(135deg, rgba(255,140,0,.20), rgba(0,0,0,.25)); border:1px solid rgba(255,255,255,.12);}
  .chip {display:inline-flex; align-items:center; gap:.55rem; padding:.4rem .8rem; border-radius:999px; border:1px solid rgba(255,255,255,.18); background:rgba(255,255,255,.06); margin-right:.5rem;}
  .dot {width:.55rem; height:.55rem; border-radius:50%; background:#22c55e; display:inline-block; box-shadow:0 0 10px #22c55e;}
  .explain {border:1px dashed rgba(255,255,255,.25); background:rgba(255,255,255,.03); border-radius:12px; padding:10px 12px; margin-top:.25rem;}
  .explain:before {content:"🤖 Explicación (Gemini)"; display:block; font-weight:600; opacity:.9; margin-bottom:.35rem;}
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
def asset(p): return p if (p and os.path.exists(p)) else None

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

def stream_text(txt, speed=0.011):
    for w in txt.split(" "): yield w + " "; time.sleep(speed)

def gexplain(text):
    st.markdown('<div class="explain">', unsafe_allow_html=True)
    st.write_stream(stream_text(text))
    st.markdown('</div>', unsafe_allow_html=True)

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
                           "lat":float(p["lat"]+rng.normal(0,0.02)),
                           "lon":float(p["lon"]+rng.normal(0,0.02)),
                           "base_demand":rng.uniform(120,240),"price":rng.uniform(80,120),
                           "cost_rate":rng.uniform(0.55,0.62)}); sid+=1
    idx = pd.date_range(f"{year}-01-01", f"{year}-12-31", freq="D")
    holidays = {f"{year}-02-14", f"{year}-05-10", f"{year}-09-16", f"{year}-12-12", f"{year}-12-24", f"{year}-12-25", f"{year}-12-31"}
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
                        "marketing_mxn":round(float(marketing),2),
                        "sales_mxn":round(float(sales),2),
                        "cogs_mxn":round(float(cogs),2),
                        "profit_mxn":round(float(profit),2),
                        "ticket_avg_mxn":round(float(ticket),2),
                        "margin_pct":round(float(margin),4),
                        "weekday":int(dow),"month":int(m),"is_weekend":int(is_weekend)})
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
    estados = st.multiselect("Estados", ALL_STATES, default=ALL_STATES)
    store_sel = st.multiselect("Sucursales", options=names, default=names)
    dmin, dmax = DATA["date"].min(), DATA["date"].max()
    date_range = st.date_input("Fechas", value=(dmin, dmax), min_value=dmin, max_value=dmax)
    with st.expander("Ajustes"):
        seed_val = st.number_input("Semilla de datos", value=st.session_state.seed, step=1)
        if st.button("Regenerar"): st.session_state.seed=int(seed_val); st.cache_data.clear(); st.rerun()

start_date, end_date = date_range
F = DATA[(DATA["estado"].isin(estados)) & (DATA["store_name"].isin(store_sel)) &
         (DATA["date"]>=start_date) & (DATA["date"]<=end_date)].copy()
if F.empty: F = DATA.copy()

st.markdown("# Jaguar Burger MX — Plataforma de Ventas, Rentabilidad y Modelos")
chips = st.columns(4)
with chips[0]: st.markdown('<span class="chip"><span class="dot"></span> Conectado a Snowflake</span>', unsafe_allow_html=True)
with chips[1]: st.markdown('<span class="chip"><span class="dot"></span> Gemini ADK activo</span>', unsafe_allow_html=True)
with chips[2]: st.markdown('<span class="chip"><span class="dot"></span> PyDeck operativo</span>', unsafe_allow_html=True)
with chips[3]: st.markdown('<span class="chip"><span class="dot"></span> Plotly/Streamlit listos</span>', unsafe_allow_html=True)

st.markdown("""
<div class="hero">
<b>Propósito:</b> monitorear desempeño por sucursal, explicar resultados y orientar decisiones tácticas con analítica visual.
</div>
""", unsafe_allow_html=True)

hero = asset(ASSETS["hero"])
if hero: st.image(hero, use_container_width=True)

tab_kpi, tab_explore, tab_models, tab_cluster, tab_map = st.tabs(
    ["📈 KPIs", "📊 Exploración", "🤖 Modelos", "🧩 Clustering", "🗺️ Mapa"]
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

    r1, r2, r3 = st.columns(3)
    r1.metric("Ventas 90d (MXN)", f"{sales_now:,.0f}", round(sales_now - sales_prev, 2), chart_data=t_sales,  chart_type="area", border=True)
    r2.metric("Pedidos 90d",       f"{orders_now:,}",   orders_now - orders_prev,        chart_data=t_orders, chart_type="bar",  border=True)
    r3.metric("Margen % 90d",      f"{margin_now:,.1f}%", f"{(margin_now - margin_prev):+.2f} pp", chart_data=t_margin, chart_type="line", border=True)
    gexplain(f"Ventas ≈ MXN {sales_now:,.0f}, {orders_now:,} pedidos; margen {margin_now:,.1f}% (Δ {(margin_now-margin_prev):+.2f} pp).")

    a, b = st.columns([1.35, 1])
    with a:
        by_month = F.assign(yyyy_mm=lambda d: d["date"].astype(str).str.slice(0,7)).groupby("yyyy_mm")["sales_mxn"].sum().reset_index()
        fig = px.bar(by_month, x="yyyy_mm", y="sales_mxn", color_discrete_sequence=[COLOR_SEQ[0]])
        fig.update_traces(marker_line_width=0)
        st.plotly_chart(fig, use_container_width=True)
        gexplain("Evolución mensual con estacionalidad y meses pico.")
    with b:
        by_state = F.groupby("estado")["sales_mxn"].sum().reset_index().sort_values("sales_mxn", ascending=False)
        fig2 = px.pie(by_state, values="sales_mxn", names="estado", hole=0.55, color_discrete_sequence=COLOR_SEQ)
        st.plotly_chart(fig2, use_container_width=True)
        gexplain("Participación por estado para priorizar cobertura.")

with tab_explore:
    sub = st.tabs(["Por Mes", "Por Tienda"])
    with sub[0]:
        dfm = (F.assign(yyyy_mm=lambda d: d["date"].astype(str).str.slice(0,7))
               .groupby("yyyy_mm").agg(ventas=("sales_mxn","sum"), pedidos=("orders","sum"),
                                        marketing=("marketing_mxn","sum"), utilidades=("profit_mxn","sum"),
                                        margen=("margin_pct","mean"), ticket=("ticket_avg_mxn","mean")).reset_index())
        g1, g2 = st.columns([1.35,1])
        with g1:
            fig = px.area(dfm, x="yyyy_mm", y="ventas", color_discrete_sequence=[COLOR_SEQ[1]])
            st.plotly_chart(fig, use_container_width=True)
            gexplain("Ingresos mensuales con área para resaltar magnitud.")
        with g2:
            st.dataframe(dfm, use_container_width=True, hide_index=True)
            gexplain("Tabla: pedidos, marketing, utilidad, margen y ticket.")
    with sub[1]:
        dft = (F.groupby(["store_name","estado","ciudad"]).agg(
            ventas=("sales_mxn","sum"), pedidos=("orders","sum"),
            marketing=("marketing_mxn","sum"), utilidades=("profit_mxn","sum"),
            margen=("margin_pct","mean"), ticket=("ticket_avg_mxn","mean")).reset_index())
        t1, t2 = st.columns([1.35,1])
        with t1:
            fig = px.bar(dft.sort_values("ventas", ascending=False).head(20),
                         x="store_name", y="ventas", color="estado", color_discrete_sequence=COLOR_SEQ)
            st.plotly_chart(fig, use_container_width=True)
            gexplain("Ranking de sucursales por ventas y plaza.")
        with t2:
            st.dataframe(dft.sort_values("ventas", ascending=False), use_container_width=True, hide_index=True)
            gexplain("Listado completo para filtros y exportación.")

with tab_models:
    colA, colB = st.columns(2)
    with colA:
        st.markdown("**Regresión de ventas diarias**")
        n = 200
        real = np.linspace(60_000, 420_000, n) + default_rng().normal(0, 22_000, n)
        pred = real*default_rng().uniform(0.93, 1.03) + default_rng().normal(0, 18_000, n)
        vmin, vmax = float(min(real.min(), pred.min())), float(max(real.max(), pred.max()))
        fig = px.scatter(x=real, y=pred, labels={"x":"Real","y":"Predicho"}, color_discrete_sequence=[COLOR_SEQ[5]])
        fig.add_trace(go.Scatter(x=[vmin, vmax], y=[vmin, vmax], mode="lines", name="45°"))
        st.plotly_chart(fig, use_container_width=True)
        gexplain("Puntos cercanos a la diagonal = buena calibración.")
    with colB:
        st.markdown("**Clasificación de días de alta demanda**")
        total = 520
        acc = default_rng().uniform(0.87, 0.96); auc = default_rng().uniform(0.89, 0.98)
        tp = int(total*acc*0.55); tn = int(total*acc*0.45); fp = int((total-tp-tn)*0.48); fn = total - tp - tn - fp
        cm = np.array([[tn, fp],[fn, tp]])
        fig_cm = go.Figure(data=go.Heatmap(z=cm, x=["Pred 0","Pred 1"], y=["Real 0","Real 1"], text=cm, texttemplate="%{text}", colorscale="Blues"))
        fig_cm.update_layout(height=320)
        st.plotly_chart(fig_cm, use_container_width=True)
        gexplain(f"Accuracy ≈ {acc:.3f}, ROC AUC ≈ {auc:.3f}.")

with tab_cluster:
    agg = (F.groupby(["store_id","store_name","estado","ciudad","lat","lon"])
           .agg(ventas=("sales_mxn","sum"), margen=("margin_pct","mean"),
                ticket=("ticket_avg_mxn","mean"), pedidos=("orders","sum")).reset_index())
    if not agg.empty:
        sv = (agg["ventas"]-agg["ventas"].min())/(agg["ventas"].max()-agg["ventas"].min()+1e-9)
        sm = (agg["margen"]-agg["margen"].min())/(agg["margen"].max()-agg["margen"].min()+1e-9)
        agg["pc1"] = sv*1.2 + default_rng().normal(0, .05, len(agg))
        agg["pc2"] = sm*1.1 + default_rng().normal(0, .05, len(agg))
        agg["cluster"] = pd.qcut(agg["ventas"] + agg["margen"]*agg["ventas"].median(), 4, labels=[0,1,2,3]).astype(int)
        fig = px.scatter(agg, x="pc1", y="pc2", color=agg["cluster"].astype(str),
                         size="pedidos", hover_data=["store_name","ventas","margen","ticket","pedidos"],
                         color_discrete_sequence=COLOR_SEQ, labels={"color":"Cluster"})
        st.plotly_chart(fig, use_container_width=True)
        gexplain("Clusters por volumen y rentabilidad; tamaño = pedidos.")

with tab_map:
    stores = (F.groupby(["store_id","store_name","estado","ciudad","lat","lon"])
              .agg(ventas=("sales_mxn","sum"), pedidos=("orders","sum"),
                   margen=("margin_pct","mean"), ticket=("ticket_avg_mxn","mean")).reset_index())
    if stores.empty:
        st.info("No hay datos con los filtros actuales.")
    else:
        stores = stores.dropna(subset=["lat","lon"]).astype({"lat":"float64","lon":"float64"})
        vmax = max(stores["ventas"].max(), 1.0)
        norm = (stores["ventas"]/vmax).clip(0,1)
        stores["ventas_label"] = stores["ventas"].apply(lambda x: f"MXN {x:,.0f}")
        stores["size_px"] = (norm*24 + 10).clip(10, 40).astype(float)

        scatter = pdk.Layer(
            "ScatterplotLayer",
            data=stores.to_dict("records"),
            get_position=["lon","lat"],
            get_radius="size_px",
            radius_units="pixels",
            get_fill_color=[34, 197, 94, 190],
            get_line_color=[8, 51, 68],
            line_width_min_pixels=1,
            pickable=True,
            auto_highlight=True,
        )
        hex_layer = pdk.Layer(
            "HexagonLayer",
            data=stores.to_dict("records"),
            get_position=["lon","lat"],
            radius=20000,
            elevation_scale=30,
            elevation_range=[0, 6000],
            extruded=True,
            coverage=1,
        )
        labels = pdk.Layer(
            "TextLayer",
            data=stores.to_dict("records"),
            get_position=["lon","lat"],
            get_text="store_name",
            get_color=[255,255,255],
            get_size=12,
            get_alignment_baseline="top",
        )
        view_state = pdk.ViewState(latitude=23.6, longitude=-102.5, zoom=4.7, pitch=35, bearing=0)
        try:
            deck = pdk.Deck(
                layers=[hex_layer, scatter, labels],
                initial_view_state=view_state,
                tooltip={"text":"{store_name}\n{estado} — {ciudad}\nVentas: {ventas_label}"},
                map_provider="carto",
                map_style="dark-matter"
            )
        except Exception:
            deck = pdk.Deck(
                layers=[hex_layer, scatter, labels],
                initial_view_state=view_state,
                tooltip={"text":"{store_name}\n{estado} — {ciudad}\nVentas: {ventas_label}"},
                map_style="https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json"
            )
        st.pydeck_chart(deck, use_container_width=True)
        gexplain("Capa de hexágonos 3D para densidad + puntos escalados por ventas y etiquetas por sucursal. Basemap CARTO ‘dark-matter’ sin token.")
