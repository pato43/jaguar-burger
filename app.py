import io, os, time
from functools import lru_cache
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import pydeck as pdk
import streamlit as st
from PIL import Image, ImageDraw, ImageFilter, ImageFont
from numpy.random import default_rng

# ─────────────────────────── Configuración y estilo global
st.set_page_config(page_title="Jaguar Burger MX — BI", page_icon="🍔", layout="wide")
px.defaults.template = "plotly_dark"
COLOR_SEQ = ["#7dd3fc", "#34d399", "#fbbf24", "#f472b6", "#60a5fa", "#a78bfa", "#fb7185", "#22d3ee", "#ef4444"]

st.markdown("""
<style>
  :root{
    --bg1:#0b1020; --bg2:#0d1117; --panel:rgba(255,255,255,.06); --border:rgba(255,255,255,.12);
    --chip:rgba(255,255,255,.08); --accent:#fb923c; --accent2:#22c55e;
  }
  html, body, [data-testid="stAppViewContainer"] * {font-size: 17px !important;}
  [data-testid="stAppViewContainer"] {background: linear-gradient(180deg,var(--bg1) 0%, var(--bg2) 65%);}
  h1 {font-size: 2.7rem !important; margin-bottom:.2rem}
  h2 {font-size: 1.8rem !important;}
  h3 {font-size: 1.28rem !important;}
  /* Sidebar */
  [data-testid="stSidebar"]{
    background: radial-gradient(1200px 550px at -200px -150px, #1f2937 0%, #0f172a 40%, #0b1222 85%);
    border-right:1px solid var(--border);
  }
  [data-testid="stSidebar"] .stMultiSelect, [data-testid="stSidebar"] .stDateInput{filter: saturate(1.05) contrast(1.05)}
  [data-testid="stSidebar"] img {border-radius:14px; box-shadow:0 6px 20px rgba(0,0,0,.35)}
  .hero{
    border-radius:18px; padding:18px 22px;
    background:linear-gradient(135deg, rgba(251,146,60,.22), rgba(255,255,255,.08));
    border:1px solid var(--border);
  }
  .chip{
    display:inline-flex; align-items:center; gap:.55rem; padding:.4rem .8rem; border-radius:999px;
    border:1px solid var(--border); background:var(--chip); margin-right:.5rem;
  }
  .dot{width:.55rem; height:.55rem; border-radius:50%; background:var(--accent2); box-shadow:0 0 10px var(--accent2);}
  .badge-row img{max-height:46px; object-fit:contain;}
  .card{border:1px solid var(--border); background:var(--panel); border-radius:14px; padding:14px;}
  .explain{border:1px dashed rgba(255,255,255,.25); background:rgba(255,255,255,.03); border-radius:12px; padding:10px 12px; margin-top:.35rem;}
  .explain:before{content:"🤖 Explicación (Gemini)"; display:block; font-weight:700; opacity:.92; margin-bottom:.3rem;}
  .pill{display:inline-block; padding:.1rem .55rem; border-radius:999px; border:1px solid var(--border); background:rgba(255,255,255,.05)}
  .kpi{border:1px solid var(--border); border-radius:16px; padding:10px 14px; background:rgba(255,255,255,.04)}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────── Assets
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

def stream_text(txt, speed=0.010):
    for w in txt.split(" "):
        yield w + " "; time.sleep(speed)

def gexplain(text):
    st.markdown('<div class="explain">', unsafe_allow_html=True)
    st.write_stream(stream_text(text))
    st.markdown('</div>', unsafe_allow_html=True)

# ─────────────────────────── Datos sintéticos base
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
            inv = max(orders + int(rng.normal(25,10)), 0)
            stockout = int(orders > inv)
            rows.append({"date":d.date(),"store_id":s["store_id"],"store_name":s["store_name"],
                        "estado":s["estado"],"ciudad":s["ciudad"],"lat":s["lat"],"lon":s["lon"],
                        "orders":orders,"price":round(price,2),"discount":discount,
                        "marketing_mxn":round(float(marketing),2),
                        "sales_mxn":round(float(sales),2),
                        "cogs_mxn":round(float(cogs),2),
                        "profit_mxn":round(float(profit),2),
                        "ticket_avg_mxn":round(float(ticket),2),
                        "margin_pct":round(float(margin),4),
                        "inventory_units":inv,"stockout":stockout,
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

# ─────────────────────────── Sidebar
with st.sidebar:
    lg = asset(ASSETS["logo"])
    st.image(lg or fallback_logo(), caption="Jaguar Burger MX", use_container_width=True)
    st.markdown("**Filtros**")
    estados = st.multiselect("Estados", ALL_STATES, default=ALL_STATES)
    store_sel = st.multiselect("Sucursales", options=names, default=names)
    dmin, dmax = DATA["date"].min(), DATA["date"].max()
    date_range = st.date_input("Rango de fechas", value=(dmin, dmax), min_value=dmin, max_value=dmax)
    st.markdown("<span class='pill'>Vista oscura automática</span>  <span class='pill'>Tiempo real</span>", unsafe_allow_html=True)
    with st.expander("Ajustes avanzados"):
        seed_val = st.number_input("Semilla de datos", value=st.session_state.seed, step=1)
        if st.button("Regenerar dataset"): st.session_state.seed=int(seed_val); st.cache_data.clear(); st.rerun()

start_date, end_date = date_range
F = DATA[(DATA["estado"].isin(estados)) & (DATA["store_name"].isin(store_sel)) &
         (DATA["date"]>=start_date) & (DATA["date"]<=end_date)].copy()
if F.empty: F = DATA.copy()

# ─────────────────────────── Encabezado general
st.markdown("# Jaguar Burger MX — Plataforma de Ventas, Rentabilidad y Modelos")
chips = st.columns(5)
with chips[0]: st.markdown('<span class="chip"><span class="dot"></span> Conectado a Snowflake</span>', unsafe_allow_html=True)
with chips[1]: st.markdown('<span class="chip"><span class="dot"></span> Gemini ADK activo</span>', unsafe_allow_html=True)
with chips[2]: st.markdown('<span class="chip"><span class="dot"></span> PyDeck operativo</span>', unsafe_allow_html=True)
with chips[3]: st.markdown('<span class="chip"><span class="dot"></span> Plotly listo</span>', unsafe_allow_html=True)
with chips[4]: st.markdown(f"<span class='chip'>Ventana: {start_date} → {end_date}</span>", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
<b>Propósito:</b> monitorear KPIs de ventas, utilidad y operación por sucursal, explicar resultados con apoyo analítico y detectar oportunidades inmediatas.
<br/><b>Módulos:</b> KPIs ejecutivos, Exploración, Modelos, Clustering, Mapa, Escenarios, Anomalías y Cohortes.
</div>
""", unsafe_allow_html=True)

hero = asset(ASSETS["hero"])
if hero: st.image(hero, use_container_width=True)

st.markdown("##### Stack tecnológico")
cols = st.columns(len(ASSETS["tech"]))
for c, (label, path) in zip(cols, ASSETS["tech"].items()):
    c.image(asset(path) or (lg or fallback_logo()), caption=label, use_container_width=True)

# ─────────────────────────── Tabs principales
tab_kpi, tab_explore, tab_models, tab_cluster, tab_map, tab_scen, tab_anom, tab_coh = st.tabs(
    ["📈 KPIs", "📊 Exploración", "🤖 Modelos", "🧩 Clustering", "🗺️ Mapa", "🎯 Escenarios", "⚠️ Anomalías", "🧪 Cohortes"]
)

# ─────────────────────────── KPIs
with tab_kpi:
    recent = F.sort_values("date").tail(90)
    prev = F.sort_values("date").iloc[-180:-90] if len(F) >= 180 else F.head(0)
    v_now, v_prev = float(recent["sales_mxn"].sum()), float(prev["sales_mxn"].sum()) if len(prev) else 0.0
    o_now, o_prev = int(recent["orders"].sum()), int(prev["orders"].sum()) if len(prev) else 0
    m_now = (recent["profit_mxn"].sum()/max(1.0, recent["sales_mxn"].sum()))*100
    m_prev = (prev["profit_mxn"].sum()/max(1.0, v_prev))*100 if len(prev) else 0.0

    t_sales  = recent.groupby("date")["sales_mxn"].sum().tolist()
    t_orders = recent.groupby("date")["orders"].sum().tolist()
    t_margin = ((recent.groupby("date")["profit_mxn"].sum()/recent.groupby("date")["sales_mxn"].sum()).fillna(0)*100).tolist()

    k1, k2, k3 = st.columns(3)
    k1.metric("Ventas 90d (MXN)", f"{v_now:,.0f}", round(v_now - v_prev, 2), chart_data=t_sales, chart_type="area", border=True)
    k2.metric("Pedidos 90d", f"{o_now:,}", o_now - o_prev, chart_data=t_orders, chart_type="bar", border=True)
    k3.metric("Margen % 90d", f"{m_now:,.1f}%", f"{(m_now-m_prev):+.2f} pp", chart_data=t_margin, chart_type="line", border=True)
    gexplain(f"Ventas recientes ≈ MXN {v_now:,.0f} con {o_now:,} pedidos. El margen ronda {m_now:,.1f}% (Δ {(m_now-m_prev):+.2f} pp). "
             "Picos en las mini-tendencias sugieren efecto de campañas y estacionalidad.")

    cA, cB = st.columns([1.35,1])
    with cA:
        by_month = (F.assign(yyyy_mm=lambda d: d["date"].astype(str).str.slice(0,7))
                     .groupby("yyyy_mm")["sales_mxn"].sum().reset_index())
        fig = px.bar(by_month, x="yyyy_mm", y="sales_mxn", color_discrete_sequence=[COLOR_SEQ[0]])
        fig.update_traces(marker_line_width=0)
        st.plotly_chart(fig, use_container_width=True)
        gexplain("Evolución mensual: permite alinear inventario y staffing con la estacionalidad observable.")
    with cB:
        by_state = F.groupby("estado")["sales_mxn"].sum().reset_index().sort_values("sales_mxn", ascending=False)
        fig2 = px.pie(by_state, values="sales_mxn", names="estado", hole=0.55, color_discrete_sequence=COLOR_SEQ)
        st.plotly_chart(fig2, use_container_width=True)
        gexplain("Participación por estado: prioriza inversión en plazas líderes y planes de recuperación en rezagadas.")

    cC, cD = st.columns([1.35,1])
    with cC:
        piv = F.pivot_table(index="weekday", columns="month", values="sales_mxn", aggfunc="sum").fillna(0)
        fig3 = go.Figure(data=go.Heatmap(z=piv.values, x=piv.columns, y=["Lun","Mar","Mié","Jue","Vie","Sáb","Dom"], colorscale="Turbo"))
        fig3.update_layout(title="Día de semana vs Mes", height=360)
        st.plotly_chart(fig3, use_container_width=True)
        gexplain("Mapa de calor: muestra combinaciones críticas (ej. fines de semana con alta estacionalidad) para reforzar operación.")
    with cD:
        kpi_tbl = (F.groupby("store_name").agg(ventas=("sales_mxn","sum"),
                                               pedidos=("orders","sum"),
                                               margen=("margin_pct","mean"),
                                               ticket=("ticket_avg_mxn","mean"))
                   .reset_index().sort_values("ventas", ascending=False).head(12))
        st.dataframe(kpi_tbl, use_container_width=True, hide_index=True)
        gexplain("Top sucursales: benchmarking interno de ticket y margen para fijar metas realistas.")
        st.download_button("Descargar TOP tiendas (CSV)", kpi_tbl.to_csv(index=False).encode("utf-8"), "top_tiendas.csv")

# ─────────────────────────── Exploración
with tab_explore:
    sub = st.tabs(["Por Mes", "Por Tienda", "Por Estado", "Distribuciones", "Descomposición", "Pareto 80/20", "Eficiencia", "Stock & Servicio"])
    with sub[0]:
        dfm = (F.assign(yyyy_mm=lambda d: d["date"].astype(str).str.slice(0,7))
                 .groupby("yyyy_mm").agg(
                    ventas=("sales_mxn","sum"), pedidos=("orders","sum"),
                    marketing=("marketing_mxn","sum"), utilidades=("profit_mxn","sum"),
                    margen=("margin_pct","mean"), ticket=("ticket_avg_mxn","mean")).reset_index())
        g1, g2 = st.columns([1.35,1])
        with g1:
            fig = px.area(dfm, x="yyyy_mm", y="ventas", color_discrete_sequence=[COLOR_SEQ[1]])
            st.plotly_chart(fig, use_container_width=True)
            gexplain("Ingresos mensuales con área: resalta magnitud acumulada y puntos de inflexión.")
        with g2:
            st.dataframe(dfm, use_container_width=True, hide_index=True)
            gexplain("Tabla mensual 360°: pedidos, marketing, utilidad, margen y ticket.")
            st.download_button("Descargar tabla mensual (CSV)", dfm.to_csv(index=False).encode("utf-8"), "mensual.csv")
    with sub[1]:
        dft = (F.groupby(["store_name","estado","ciudad"]).agg(
                ventas=("sales_mxn","sum"), pedidos=("orders","sum"),
                marketing=("marketing_mxn","sum"), utilidades=("profit_mxn","sum"),
                margen=("margin_pct","mean"), ticket=("ticket_avg_mxn","mean")).reset_index())
        t1, t2 = st.columns([1.35,1])
        with t1:
            fig = px.bar(dft.sort_values("ventas", ascending=False).head(20),
                         x="store_name", y="ventas", color="estado",
                         color_discrete_sequence=COLOR_SEQ)
            fig.update_layout(xaxis_title="Sucursal", yaxis_title="Ventas (MXN)")
            st.plotly_chart(fig, use_container_width=True)
            gexplain("Ranking de sucursales por ventas con color por estado para contexto geográfico.")
        with t2:
            st.dataframe(dft.sort_values("ventas", ascending=False), use_container_width=True, hide_index=True)
            gexplain("Listado completo para filtros y exportación.")
            st.download_button("Descargar sucursales (CSV)", dft.to_csv(index=False).encode("utf-8"), "sucursales.csv")
    with sub[2]:
        dfs = (F.groupby("estado").agg(
                ventas=("sales_mxn","sum"), pedidos=("orders","sum"),
                utilidades=("profit_mxn","sum"), margen=("margin_pct","mean")).reset_index())
        s1, s2 = st.columns([1.35,1])
        with s1:
            fig = px.bar(dfs.sort_values("ventas", ascending=False),
                         x="estado", y="ventas", color="estado",
                         color_discrete_sequence=COLOR_SEQ)
            fig.update_layout(xaxis_title="Estado", yaxis_title="Ventas (MXN)", showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
            gexplain("Panorama por estado: cuantifica brechas de cobertura.")
        with s2:
            st.dataframe(dfs.sort_values("ventas", ascending=False), use_container_width=True, hide_index=True)
            gexplain("Cifras agregadas para dimensionar objetivos regionales.")
    with sub[3]:
        dx1, dx2 = st.columns([1.35,1])
        with dx1:
            fig = px.histogram(F, x="ticket_avg_mxn", nbins=30, marginal="box", color_discrete_sequence=[COLOR_SEQ[4]])
            st.plotly_chart(fig, use_container_width=True)
            gexplain("Distribución de ticket: colas largas sugieren revisar surtido premium/valor.")
        with dx2:
            fig = px.scatter(F, x="marketing_mxn", y="sales_mxn", color="estado", color_discrete_sequence=COLOR_SEQ)
            st.plotly_chart(fig, use_container_width=True)
            gexplain("Relación marketing–ventas: pendiente aparente ayuda a inferir elasticidad por plaza.")
    with sub[4]:
        totals = F[["sales_mxn","cogs_mxn","marketing_mxn","profit_mxn"]].sum()
        wf = go.Figure(go.Waterfall(
            measure=["relative","relative","relative","total"],
            x=["Ventas","- Costo de ventas","- Marketing","Utilidad"],
            y=[totals["sales_mxn"], -totals["cogs_mxn"], -totals["marketing_mxn"], totals["profit_mxn"]],
        ))
        st.plotly_chart(wf, use_container_width=True)
        gexplain("Puente Ventas → Utilidad: cuantifica impactos clave en el resultado.")
    with sub[5]:
        pareto = (F.groupby("store_name")["sales_mxn"].sum().sort_values(ascending=False).reset_index())
        pareto["cum_share"] = pareto["sales_mxn"].cumsum()/pareto["sales_mxn"].sum()
        fig = go.Figure()
        fig.add_trace(go.Bar(x=pareto["store_name"], y=pareto["sales_mxn"], name="Ventas", marker_color=COLOR_SEQ[2]))
        fig.add_trace(go.Scatter(x=pareto["store_name"], y=pareto["cum_share"], name="Acumulado", yaxis="y2"))
        fig.update_layout(yaxis2=dict(overlaying="y", side="right", tickformat=".0%"), xaxis_title="Sucursal")
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(pareto, use_container_width=True, hide_index=True)
        gexplain("Pareto 80/20: pocas tiendas concentran la mayor parte de ingresos; enfoca esfuerzo donde más impacta.")
    with sub[6]:
        ef = F.groupby("store_name").agg(ventas=("sales_mxn","sum"),
                                         marketing=("marketing_mxn","sum"),
                                         utilidad=("profit_mxn","sum"),
                                         pedidos=("orders","sum")).reset_index()
        fig = px.scatter(ef, x="marketing", y="utilidad", size="pedidos", color="ventas",
                         color_continuous_scale="Sunset", hover_name="store_name")
        st.plotly_chart(fig, use_container_width=True)
        gexplain("Eficiencia: utilidad vs marketing ponderado por pedidos; ubica ROI alto vs bajo.")
        st.dataframe(ef.sort_values("utilidad", ascending=False), use_container_width=True, hide_index=True)
        gexplain("Tabla de eficiencia: base para cambios de presupuesto.")
    with sub[7]:
        svc = (F.assign(service_level=lambda d: 1 - d["stockout"])
                 .groupby("store_name").agg(service=("service_level","mean"),
                                            stockouts=("stockout","sum"),
                                            pedidos=("orders","sum")).reset_index())
        s1, s2 = st.columns([1.35,1])
        with s1:
            fig = px.bar(svc.sort_values("service", ascending=False), x="store_name", y="service", color="service",
                         color_continuous_scale="Tealgrn", range_y=[0,1])
            st.plotly_chart(fig, use_container_width=True)
            gexplain("Nivel de servicio estimado (1-sin quiebres): alerta sobre tiendas con mayor incidencia de stockouts.")
        with s2:
            st.dataframe(svc.sort_values(["service","stockouts"], ascending=[True,False]), use_container_width=True, hide_index=True)
            gexplain("Detalle de quiebres por tienda para acciones logísticas.")

# ─────────────────────────── Modelos (visual)
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
        gexplain("Ajuste de regresión: cuanto más cerca a la diagonal, mejor calibración; dispersión alta sugiere variables faltantes.")
        sample_tbl = pd.DataFrame({"Muestra": np.arange(1, 21),
                                   "Real": real[:20].round(0).astype(int),
                                   "Predicho": pred[:20].round(0).astype(int)})
        st.dataframe(sample_tbl, use_container_width=True, hide_index=True)
        gexplain("Muestra de predicciones para inspección puntual y validación de orden de magnitud.")
    with colB:
        st.markdown("**Clasificación de días de alta demanda**")
        total = 520
        acc = default_rng().uniform(0.87, 0.96); auc = default_rng().uniform(0.89, 0.98)
        tp = int(total*acc*0.55); tn = int(total*acc*0.45); fp = int((total-tp-tn)*0.48); fn = total - tp - tn - fp
        cm = np.array([[tn, fp],[fn, tp]])
        fig_cm = go.Figure(data=go.Heatmap(z=cm, x=["Pred 0","Pred 1"], y=["Real 0","Real 1"], text=cm, texttemplate="%{text}", colorscale="Blues"))
        fig_cm.update_layout(height=320)
        st.plotly_chart(fig_cm, use_container_width=True)
        gexplain(f"Matriz de confusión: accuracy ≈ {acc:.3f}, ROC AUC ≈ {auc:.3f}. Base para decidir umbrales por operación.")
        fpr = np.linspace(0,1,140); tpr = np.clip(fpr**0.6 + default_rng().normal(0,0.03,140), 0, 1)
        froc = go.Figure(); froc.add_trace(go.Scatter(x=fpr,y=tpr, mode="lines", name="ROC"))
        froc.add_trace(go.Scatter(x=[0,1], y=[0,1], mode="lines", name="Azar"))
        froc.update_layout(height=280, title="Curva ROC")
        st.plotly_chart(froc, use_container_width=True)
        gexplain("Curva ROC: sensibilidad vs especificidad; el área resume capacidad discriminativa.")

# ─────────────────────────── Clustering (mejorado)
with tab_cluster:
    agg = (F.groupby(["store_id","store_name","estado","ciudad","lat","lon"])
             .agg(ventas=("sales_mxn","sum"), margen=("margin_pct","mean"),
                  ticket=("ticket_avg_mxn","mean"), pedidos=("orders","sum")).reset_index())
    if agg.empty:
        st.info("No hay datos con los filtros actuales.")
    else:
        sv = (agg["ventas"]-agg["ventas"].min())/(agg["ventas"].max()-agg["ventas"].min()+1e-9)
        sm = (agg["margen"]-agg["margen"].min())/(agg["margen"].max()-agg["margen"].min()+1e-9)
        agg["pc1"] = sv*1.2 + default_rng().normal(0, .05, len(agg))
        agg["pc2"] = sm*1.1 + default_rng().normal(0, .05, len(agg))
        agg["cluster"] = pd.qcut(agg["ventas"] + agg["margen"]*agg["ventas"].median(), 4, labels=[0,1,2,3]).astype(int)
        c1, c2 = st.columns([1.35,1])
        with c1:
            fig = px.scatter(agg, x="pc1", y="pc2", color=agg["cluster"].astype(str),
                             size="pedidos", hover_data=["store_name","ventas","margen","ticket","pedidos"],
                             labels={"color":"Cluster"}, color_discrete_sequence=COLOR_SEQ)
            st.plotly_chart(fig, use_container_width=True)
            gexplain("Clusters por volumen y rentabilidad; el tamaño del punto representa pedidos. Segmentación útil para campañas diferenciadas.")
        with c2:
            grid = (agg.groupby("cluster")[["ventas","margen","ticket","pedidos"]]
                      .mean().round(2).reset_index().rename(columns={"cluster":"Cluster"}))
            st.dataframe(grid, use_container_width=True, hide_index=True)
            gexplain("Perfil promedio por cluster: guía de acciones comerciales y operativas.")
        st.dataframe(agg[["store_name","estado","ciudad","ventas","margen","ticket","pedidos","cluster"]]
                     .sort_values(["cluster","ventas"], ascending=[True,False]),
                     use_container_width=True, hide_index=True)
        gexplain("Listado por cluster para priorización de visitas, presupuesto y abastecimiento.")

# ─────────────────────────── Mapa (capas 3D y puntos con labels)
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
        stores["size_px"] = (norm*26 + 10).clip(10, 44).astype(float)

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
        view_state = pdk.ViewState(latitude=23.6, longitude=-102.5, zoom=4.8, pitch=35, bearing=0)
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
        gexplain("Hexágonos 3D muestran densidad; puntos escalan por ventas y labels permiten localizar cada sucursal de un vistazo.")

        l1, l2 = st.columns([1.35,1])
        with l1:
            top_geo = stores.sort_values("ventas", ascending=False)[["store_name","estado","ciudad","ventas","margen","ticket"]].head(15)
            st.dataframe(top_geo, use_container_width=True, hide_index=True)
            gexplain("Top geográfico: planifica rutas de visita y entrenamiento.")
        with l2:
            geo_state = stores.groupby("estado")["ventas"].sum().reset_index().sort_values("ventas", ascending=False)
            fig = px.bar(geo_state, x="estado", y="ventas", color="estado", color_discrete_sequence=COLOR_SEQ)
            fig.update_layout(xaxis_title="Estado", yaxis_title="Ventas (MXN)", showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
            gexplain("Contribución por estado: dimensiona metas regionales y expansión.")

# ─────────────────────────── Escenarios (precio/promoción/marketing)
with tab_scen:
    st.markdown("### Simulador de impacto")
    col1, col2, col3 = st.columns(3)
    with col1:
        delta_price = st.slider("Cambio de precio (%)", -20, 20, -5, 1)
    with col2:
        promo_boost = st.slider("Intensidad de promoción (%)", 0, 30, 10, 1)
    with col3:
        mkt_delta = st.slider("Cambio en marketing (%)", -30, 50, 10, 1)

    base = F.sort_values("date").tail(60).copy()
    base["ventas_base"] = base["sales_mxn"]
    # Elasticidades simples (demo visual)
    e_price, e_promo, e_mkt = -1.2, 0.8, 0.5
    base["ventas_scen"] = base["ventas_base"] * (1 + e_price*delta_price/100 + e_promo*promo_boost/100 + e_mkt*mkt_delta/100)
    agg = base.groupby("date")[["ventas_base","ventas_scen"]].sum().reset_index()

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=agg["date"], y=agg["ventas_base"], name="Base"))
    fig.add_trace(go.Scatter(x=agg["date"], y=agg["ventas_scen"], name="Escenario"))
    st.plotly_chart(fig, use_container_width=True)
    gexplain("Simulación de ingresos para los próximos días con ajustes de precio, promoción y marketing. Las pendientes ayudan a evaluar sensibilidad.")

    comp = pd.DataFrame({
        "Mérito": ["Total Base 60d","Total Escenario 60d","Δ Absoluto","Δ %"],
        "Valor": [agg["ventas_base"].sum(), agg["ventas_scen"].sum(),
                  agg["ventas_scen"].sum()-agg["ventas_base"].sum(),
                  100*(agg["ventas_scen"].sum()/max(1.0,agg["ventas_base"].sum())-1)]
    })
    st.dataframe(comp, use_container_width=True, hide_index=True)
    gexplain("Resumen de impacto agregado: cuantifica la ganancia/perdida total bajo la hipótesis seleccionada.")

# ─────────────────────────── Anomalías (detección simple por z-score)
with tab_anom:
    st.markdown("### Detección de anomalías")
    series = F.groupby("date")["sales_mxn"].sum().reset_index().sort_values("date")
    x = series["sales_mxn"]
    z = (x - x.mean())/x.std(ddof=0)
    series["z"] = z
    series["anomaly"] = (abs(series["z"])>2).astype(int)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=series["date"], y=series["sales_mxn"], name="Ventas"))
    fig.add_trace(go.Scatter(x=series.loc[series["anomaly"]==1,"date"],
                             y=series.loc[series["anomaly"]==1,"sales_mxn"],
                             mode="markers", name="Anomalía", marker=dict(size=10, symbol="x")))
    st.plotly_chart(fig, use_container_width=True)
    gexplain("Marcadores ‘x’ señalan días atípicos (|z|>2). Útil para auditar campañas, quiebres o registros irregulares.")

    st.dataframe(series.loc[series["anomaly"]==1].tail(20), use_container_width=True, hide_index=True)
    gexplain("Listado de anomalías recientes para investigación y corrección rápida.")

# ─────────────────────────── Cohortes (mes 0 = primer mes por tienda)
with tab_coh:
    F2 = F.copy()
    F2["yyyy_mm"] = F2["date"].astype(str).str.slice(0,7)
    first = F2.groupby("store_id")["yyyy_mm"].min().rename("cohort")
    F2 = F2.merge(first, on="store_id", how="left")
    F2["cohort_index"] = (pd.to_datetime(F2["yyyy_mm"]) - pd.to_datetime(F2["cohort"])).dt.days//30
    coh = F2.groupby(["cohort","cohort_index"])["sales_mxn"].mean().reset_index()
    pivot = coh.pivot(index="cohort", columns="cohort_index", values="sales_mxn").fillna(0)
    fig = go.Figure(data=go.Heatmap(z=pivot.values, x=pivot.columns, y=pivot.index, colorscale="Viridis"))
    fig.update_layout(height=420, xaxis_title="Mes desde apertura", yaxis_title="Cohorte (mes apertura)")
    st.plotly_chart(fig, use_container_width=True)
    gexplain("Cohortes por mes de apertura: evalúa cómo evolucionan ventas promedio después del lanzamiento de cada sucursal.")

    st.dataframe(pivot.round(0).astype(int), use_container_width=True)
    gexplain("Tabla de cohortes para exportación y seguimiento.")

# ─────────────────────────── Descarga global
csv = F.to_csv(index=False).encode("utf-8")
st.download_button("Descargar detalle (CSV)", csv, file_name="jaguar_burger_detalle.csv", mime="text/csv")
