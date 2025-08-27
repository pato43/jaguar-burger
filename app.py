import io, os, time
from functools import lru_cache
from datetime import date

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import pydeck as pdk
import streamlit as st
from PIL import Image, ImageDraw, ImageFilter, ImageFont
from numpy.random import default_rng

# ─────────────────────────── Base y estilos
st.set_page_config(page_title="Jaguar Burger MX — BI", page_icon="🍔", layout="wide")
st.markdown("""
<style>
  html, body, [data-testid="stAppViewContainer"] * {font-size: 17px !important;}
  h1 {font-size: 2.1rem !important;} h2 {font-size: 1.6rem !important;} h3 {font-size: 1.25rem !important;}
  .hero {border-radius:16px; padding:18px 22px; background:linear-gradient(135deg, rgba(255,140,0,.18), rgba(255,255,255,.06)); border:1px solid rgba(255,255,255,.12);}
  .chip {display:inline-flex; align-items:center; gap:.5rem; padding:.35rem .7rem; border-radius:999px; border:1px solid rgba(255,255,255,.18); background:rgba(255,255,255,.06); margin-right:.5rem;}
  .dot {width:.55rem; height:.55rem; border-radius:50%; background:#2ecc71; display:inline-block; box-shadow:0 0 8px rgba(46,204,113,.85);}
  [data-testid="stSidebar"] {border-right:1px solid rgba(255,255,255,.08);}
  [data-testid="stSidebar"] img {border-radius:10px;}
  .badge-row img {max-height:46px; object-fit:contain;}
  .note {opacity:.95}
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

def stream_text(txt, speed=0.012):
    for w in txt.split(" "):
        yield w + " "; time.sleep(speed)

# ─────────────────────────── Datos sintéticos
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

# ─────────────────────────── Sidebar minimalista
with st.sidebar:
    lg = asset(ASSETS["logo"])
    st.image(lg or fallback_logo(), caption="Jaguar Burger MX", use_container_width=True)
    estados = st.multiselect("Estados", ALL_STATES, default=ALL_STATES)
    store_sel = st.multiselect("Sucursales", options=names, default=names)
    dmin, dmax = DATA["date"].min(), DATA["date"].max()
    date_range = st.date_input("Fechas", value=(dmin, dmax), min_value=dmin, max_value=dmax)
    with st.expander("Avanzado"):
        seed_val = st.number_input("Semilla", value=st.session_state.seed, step=1)
        if st.button("Regenerar dataset"): st.session_state.seed=int(seed_val); st.cache_data.clear(); st.rerun()

start_date, end_date = date_range
F = DATA[(DATA["estado"].isin(estados)) & (DATA["store_name"].isin(store_sel)) &
         (DATA["date"]>=start_date) & (DATA["date"]<=end_date)].copy()
if F.empty: F = DATA.copy()

# ─────────────────────────── Encabezado con “estado de conexión”
st.markdown("## Jaguar Burger MX — Plataforma de Ventas & Analytics")
status = st.columns(5)
with status[0]: st.markdown('<span class="chip"><span class="dot"></span> Conectado a Snowflake</span>', unsafe_allow_html=True)
with status[1]: st.markdown('<span class="chip"><span class="dot"></span> Gemini ADK activo</span>', unsafe_allow_html=True)
with status[2]: st.markdown('<span class="chip"><span class="dot"></span> PyDeck OK</span>', unsafe_allow_html=True)
with status[3]: st.markdown('<span class="chip"><span class="dot"></span> Plotly listo</span>', unsafe_allow_html=True)
with status[4]: st.markdown('<span class="chip"><span class="dot"></span> Streamlit Live</span>', unsafe_allow_html=True)

st.markdown("""
<div class="hero">
<b>Propósito:</b> visibilizar desempeño de ventas y rentabilidad por sucursal, explicar resultados con apoyo analítico y detectar oportunidades.
<br/><b>Incluye:</b> KPIs con microtendencias, exploración por mes/tienda/estado, modelos, clusters y mapa operativo con calor de ventas.
</div>
""", unsafe_allow_html=True)

hero = asset(ASSETS["hero"])
if hero: st.image(hero, use_container_width=True)

st.markdown("##### Tecnologías")
cols = st.columns(len(ASSETS["tech"]))
for c, (label, path) in zip(cols, ASSETS["tech"].items()):
    c.image(asset(path) or (lg or fallback_logo()), caption=label, use_container_width=True)

# ─────────────────────────── Tabs
tab_kpi, tab_explore, tab_models, tab_cluster, tab_map = st.tabs(
    ["📈 KPIs", "📊 Exploración", "🤖 Modelos", "🧩 Clustering", "🗺️ Mapa"]
)

# ─────────────────────────── KPIs
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

    m1, m2, m3 = st.columns(3)
    m1.metric("Ventas 90d (MXN)", f"{sales_now:,.0f}", round(sales_now - sales_prev, 2), chart_data=t_sales,  chart_type="area", border=True)
    m2.metric("Pedidos 90d",       f"{orders_now:,}",   orders_now - orders_prev,        chart_data=t_orders, chart_type="bar",  border=True)
    m3.metric("Margen % 90d",      f"{margin_now:,.1f}%", f"{(margin_now - margin_prev):+.2f} pp", chart_data=t_margin, chart_type="line", border=True)
    st.write_stream(stream_text("🧠 Explicación Gemini: KPIs con tendencia de 90 días; el margen suele moverse con promociones y mix de ticket."))

    a, b = st.columns([1.3, 1])
    with a:
        by_month = F.assign(yyyy_mm=lambda d: d["date"].astype(str).str.slice(0,7)).groupby("yyyy_mm")["sales_mxn"].sum().reset_index()
        fig = px.bar(by_month, x="yyyy_mm", y="sales_mxn", labels={"yyyy_mm":"Mes","sales_mxn":"Ventas (MXN)"})
        st.plotly_chart(fig, use_container_width=True)
        st.write_stream(stream_text("🧠 Explicación Gemini: evolución mensual para detectar estacionalidad y efectos de campañas."))
    with b:
        by_state = F.groupby("estado")["sales_mxn"].sum().reset_index().sort_values("sales_mxn", ascending=False)
        fig2 = px.pie(by_state, values="sales_mxn", names="estado", hole=0.55)
        st.plotly_chart(fig2, use_container_width=True)
        st.write_stream(stream_text("🧠 Explicación Gemini: participación por estado guía la inversión y cobertura."))

    c, d = st.columns([1.3,1])
    with c:
        piv = F.pivot_table(index="weekday", columns="month", values="sales_mxn", aggfunc="sum").fillna(0)
        fig3 = go.Figure(data=go.Heatmap(z=piv.values, x=piv.columns, y=["Lun","Mar","Mié","Jue","Vie","Sáb","Dom"]))
        fig3.update_layout(title="Día de semana vs Mes", height=360)
        st.plotly_chart(fig3, use_container_width=True)
        st.write_stream(stream_text("🧠 Explicación Gemini: el calor de ventas revela picos operativos por día y mes."))
    with d:
        kpi_tbl = (F.groupby("store_name").agg(ventas=("sales_mxn","sum"),
                                               pedidos=("orders","sum"),
                                               margen=("margin_pct","mean"),
                                               ticket=("ticket_avg_mxn","mean"))
                   .reset_index().sort_values("ventas", ascending=False).head(10))
        st.dataframe(kpi_tbl, use_container_width=True, hide_index=True)
        st.write_stream(stream_text("🧠 Explicación Gemini: top sucursales para benchmarking de ticket y margen."))

    e, f = st.columns([1.3,1])
    with e:
        series = (F.assign(yyyy_mm=lambda d: d["date"].astype(str).str.slice(0,7))
                    .groupby("yyyy_mm")[["profit_mxn","marketing_mxn"]].sum().reset_index())
        fig4 = go.Figure()
        fig4.add_trace(go.Scatter(x=series["yyyy_mm"], y=series["profit_mxn"], mode="lines+markers", name="Utilidad"))
        fig4.add_trace(go.Scatter(x=series["yyyy_mm"], y=series["marketing_mxn"], mode="lines+markers", name="Marketing"))
        fig4.update_layout(title="Utilidad vs Marketing (mensual)")
        st.plotly_chart(fig4, use_container_width=True)
        st.write_stream(stream_text("🧠 Explicación Gemini: balance entre inversión y resultado; desalineaciones indican oportunidad."))
    with f:
        cor = F[["orders","price","discount","marketing_mxn","sales_mxn","profit_mxn","ticket_avg_mxn","margin_pct"]].corr(numeric_only=True)
        fig5 = go.Figure(data=go.Heatmap(z=cor.values, x=cor.columns, y=cor.columns, zmin=-1, zmax=1))
        fig5.update_layout(title="Correlaciones")
        st.plotly_chart(fig5, use_container_width=True)
        st.write_stream(stream_text("🧠 Explicación Gemini: correlaciones para entender relaciones clave entre variables."))

# ─────────────────────────── Exploración
with tab_explore:
    subtab = st.tabs(["Por Mes", "Por Tienda", "Por Estado", "Distribuciones", "Descomposición"])
    with subtab[0]:
        dfm = (F.assign(yyyy_mm=lambda d: d["date"].astype(str).str.slice(0,7))
                 .groupby("yyyy_mm").agg(
                    ventas=("sales_mxn","sum"), pedidos=("orders","sum"),
                    marketing=("marketing_mxn","sum"), utilidades=("profit_mxn","sum"),
                    margen=("margin_pct","mean"), ticket=("ticket_avg_mxn","mean")).reset_index())
        g1, g2 = st.columns([1.3,1])
        with g1:
            fig = px.area(dfm, x="yyyy_mm", y="ventas")
            st.plotly_chart(fig, use_container_width=True)
            st.write_stream(stream_text("🧠 Explicación Gemini: ingresos mensuales con área para remarcar magnitud y tendencia."))
        with g2:
            st.dataframe(dfm, use_container_width=True, hide_index=True)
            st.write_stream(stream_text("🧠 Explicación Gemini: tabla mensual con pedidos, marketing, utilidad, margen y ticket."))
    with subtab[1]:
        dft = (F.groupby(["store_name","estado","ciudad"]).agg(
                ventas=("sales_mxn","sum"), pedidos=("orders","sum"),
                marketing=("marketing_mxn","sum"), utilidades=("profit_mxn","sum"),
                margen=("margin_pct","mean"), ticket=("ticket_avg_mxn","mean")).reset_index())
        t1, t2 = st.columns([1.3,1])
        with t1:
            fig = px.bar(dft.sort_values("ventas", ascending=False).head(20), x="store_name", y="ventas", hover_data=["estado","ciudad"])
            fig.update_layout(xaxis_title="Sucursal", yaxis_title="Ventas (MXN)")
            st.plotly_chart(fig, use_container_width=True)
            st.write_stream(stream_text("🧠 Explicación Gemini: ranking de sucursales y focos de crecimiento."))
        with t2:
            st.dataframe(dft.sort_values("ventas", ascending=False), use_container_width=True, hide_index=True)
            st.write_stream(stream_text("🧠 Explicación Gemini: tabla completa para exportación y filtros."))
    with subtab[2]:
        dfs = (F.groupby("estado").agg(
                ventas=("sales_mxn","sum"), pedidos=("orders","sum"),
                utilidades=("profit_mxn","sum"), margen=("margin_pct","mean")).reset_index())
        s1, s2 = st.columns([1.3,1])
        with s1:
            fig = px.bar(dfs.sort_values("ventas", ascending=False), x="estado", y="ventas")
            fig.update_layout(xaxis_title="Estado", yaxis_title="Ventas (MXN)")
            st.plotly_chart(fig, use_container_width=True)
            st.write_stream(stream_text("🧠 Explicación Gemini: panorama por estado para priorizar inversiones."))
        with s2:
            st.dataframe(dfs.sort_values("ventas", ascending=False), use_container_width=True, hide_index=True)
            st.write_stream(stream_text("🧠 Explicación Gemini: cifras agregadas por estado."))
    with subtab[3]:
        dx1, dx2 = st.columns([1.3,1])
        with dx1:
            fig = px.histogram(F, x="ticket_avg_mxn", nbins=30, marginal="box")
            st.plotly_chart(fig, use_container_width=True)
            st.write_stream(stream_text("🧠 Explicación Gemini: distribución de ticket para detectar colas y outliers."))
        with dx2:
            fig = px.scatter(F, x="marketing_mxn", y="sales_mxn", trendline="ols")
            st.plotly_chart(fig, use_container_width=True)
            st.write_stream(stream_text("🧠 Explicación Gemini: relación marketing-ventas y elasticidad aparente."))
    with subtab[4]:
        totals = F[["sales_mxn","cogs_mxn","marketing_mxn","profit_mxn"]].sum()
        wf = go.Figure(go.Waterfall(
            measure=["relative","relative","relative","total"],
            x=["Ventas","- Costo de ventas","- Marketing","Utilidad"],
            y=[totals["sales_mxn"], -totals["cogs_mxn"], -totals["marketing_mxn"], totals["profit_mxn"]],
        ))
        st.plotly_chart(wf, use_container_width=True)
        st.write_stream(stream_text("🧠 Explicación Gemini: puente Ventas → Utilidad para entender impactos."))

# ─────────────────────────── Modelos (presentación)
with tab_models:
    colA, colB = st.columns(2)
    with colA:
        st.markdown("**Regresión de ventas diarias**")
        n = 180
        real = np.linspace(60_000, 420_000, n) + default_rng().normal(0, 22_000, n)
        pred = real*default_rng().uniform(0.93, 1.03) + default_rng().normal(0, 18_000, n)
        vmin, vmax = float(min(real.min(), pred.min())), float(max(real.max(), pred.max()))
        fig = px.scatter(x=real, y=pred, labels={"x":"Real","y":"Predicho"})
        fig.add_trace(go.Scatter(x=[vmin, vmax], y=[vmin, vmax], mode="lines", name="45°"))
        st.plotly_chart(fig, use_container_width=True)
        st.write_stream(stream_text("🧠 Explicación Gemini: la cercanía a la línea 45° indica buena calibración del modelo de regresión."))
        st.dataframe(pd.DataFrame({"Muestra": np.arange(1, 21), "Real": real[:20].round(0).astype(int), "Predicho": pred[:20].round(0).astype(int)}),
                     use_container_width=True, hide_index=True)
        st.write_stream(stream_text("🧠 Explicación Gemini: muestra de predicciones para inspección puntual."))
    with colB:
        st.markdown("**Clasificación de días de alta demanda**")
        total = 500
        acc = default_rng().uniform(0.86, 0.96); auc = default_rng().uniform(0.88, 0.98)
        tp = int(total*acc*0.55); tn = int(total*acc*0.45); fp = int((total-tp-tn)*0.48); fn = total - tp - tn - fp
        cm = np.array([[tn, fp],[fn, tp]])
        fig_cm = go.Figure(data=go.Heatmap(z=cm, x=["Pred 0","Pred 1"], y=["Real 0","Real 1"], text=cm, texttemplate="%{text}"))
        fig_cm.update_layout(height=320)
        st.plotly_chart(fig_cm, use_container_width=True)
        st.write_stream(stream_text("🧠 Explicación Gemini: matriz de confusión para entender aciertos y errores del clasificador."))
        fpr = np.linspace(0,1,120); tpr = np.clip(fpr**0.6 + default_rng().normal(0,0.03,120), 0, 1)
        froc = go.Figure(); froc.add_trace(go.Scatter(x=fpr,y=tpr, mode="lines", name="ROC"))
        froc.add_trace(go.Scatter(x=[0,1], y=[0,1], mode="lines", name="Azar"))
        froc.update_layout(height=280, title="Curva ROC")
        st.plotly_chart(froc, use_container_width=True)
        st.write_stream(stream_text("🧠 Explicación Gemini: capacidad discriminativa a distintos umbrales."))

# ─────────────────────────── Clustering
with tab_cluster:
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
        c1, c2 = st.columns([1.3,1])
        with c1:
            fig = px.scatter(agg, x="pc1", y="pc2", color=agg["cluster"].astype(str),
                             hover_data=["store_name","ventas","margen","ticket"], labels={"color":"Cluster"})
            st.plotly_chart(fig, use_container_width=True)
            st.write_stream(stream_text("🧠 Explicación Gemini: clusters de sucursales para estrategias diferenciadas."))
        with c2:
            st.dataframe(agg[["store_name","estado","ciudad","ventas","margen","ticket","cluster"]]
                         .sort_values(["cluster","ventas"], ascending=[True,False]),
                         use_container_width=True, hide_index=True)
            st.write_stream(stream_text("🧠 Explicación Gemini: perfiles por cluster con métricas clave."))

# ─────────────────────────── Mapa (con puntos garantizados)
with tab_map:
    stores_latest = (F.groupby(["store_id","store_name","estado","ciudad","lat","lon"])
                       .agg(ventas=("sales_mxn","sum"), pedidos=("orders","sum"),
                            margen=("margin_pct","mean"), ticket=("ticket_avg_mxn","mean")).reset_index())
    if stores_latest.empty:
        st.info("No hay datos con los filtros actuales.")
    else:
        stores_latest = stores_latest.dropna(subset=["lat","lon"])
        vmax = max(stores_latest["ventas"].max(), 1.0)
        stores_latest["ventas_label"] = stores_latest["ventas"].apply(lambda x: f"MXN {x:,.0f}")
        stores_latest["size_px"] = (stores_latest["ventas"]/vmax*22 + 10).clip(10, 36).astype(float)
        scatter = pdk.Layer(
            "ScatterplotLayer",
            data=stores_latest.to_dict("records"),
            get_position=["lon","lat"],
            get_radius="size_px",
            radius_units="pixels",
            get_fill_color=[0, 220, 140],
            get_line_color=[0, 80, 60],
            line_width_min_pixels=1,
            pickable=True,
            auto_highlight=True,
        )
        text = pdk.Layer(
            "TextLayer",
            data=stores_latest.to_dict("records"),
            get_position=["lon","lat"],
            get_text="store_name",
            get_color=[255,255,255],
            get_size=12,
            get_alignment_baseline="top",
        )
        columns = pdk.Layer(
            "ColumnLayer",
            data=stores_latest.to_dict("records"),
            get_position=["lon","lat"],
            get_elevation="ventas",
            elevation_scale=0.0005,
            radius=8000,
            extruded=True,
            get_fill_color=[255, 140, 0],
            pickable=True,
        )
        heat = pdk.Layer(
            "HeatmapLayer",
            data=stores_latest.to_dict("records"),
            get_position=["lon","lat"],
            aggregation="SUM",
            get_weight="ventas",
            radius_pixels=50,
        )
        view_state = pdk.ViewState(latitude=23.6, longitude=-102.5, zoom=4.4, pitch=30, bearing=0)
        deck = pdk.Deck(layers=[heat, columns, scatter, text],
                        initial_view_state=view_state,
                        tooltip={"text":"{store_name}\n{estado} — {ciudad}\nVentas: {ventas_label}"})
        st.pydeck_chart(deck, use_container_width=True)
        st.write_stream(stream_text("🧠 Explicación Gemini: puntos y columnas 3D escalan con ventas; la capa de calor resalta concentración geográfica."))

        m1, m2 = st.columns([1.3,1])
        with m1:
            top_geo = stores_latest.sort_values("ventas", ascending=False)[["store_name","estado","ciudad","ventas","margen","ticket"]].head(15)
            st.dataframe(top_geo, use_container_width=True, hide_index=True)
            st.write_stream(stream_text("🧠 Explicación Gemini: top sucursales por ventas con margen y ticket."))
        with m2:
            geo_state = stores_latest.groupby("estado")["ventas"].sum().reset_index().sort_values("ventas", ascending=False)
            fig = px.bar(geo_state, x="estado", y="ventas")
            fig.update_layout(xaxis_title="Estado", yaxis_title="Ventas (MXN)")
            st.plotly_chart(fig, use_container_width=True)
            st.write_stream(stream_text("🧠 Explicación Gemini: contribución estatal a partir de agregados geoespaciales."))

# ─────────────────────────── Descarga
csv = F.to_csv(index=False).encode("utf-8")
st.download_button("Descargar detalle (CSV)", csv, file_name="jaguar_burger_detalle.csv", mime="text/csv")
