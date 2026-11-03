"""Ocazionally Intelligence Engine: project dashboard (Streamlit). Reads only the small files in ./data."""
import json
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

D = Path(__file__).parent / "data"
TYPICAL = dict(wedding=300, navratri_society=400, diwali=80, birthday=40, college_fest=600,
               corporate=80, uttarayan=70, other=60)
GOLD, GREY, RED = "#F2A900", "#A3A3A3", "#EF4444"
nice = lambda s: " ".join(("DJ" if w == "dj" else w.capitalize()) for w in str(s).split("_"))
inr = lambda x: f"₹{int(x):,}"


def vendor_name(v):
    """Show synthetic vendors as 'Decor vendor #94' instead of 'Synthetic decor 94'."""
    parts = str(v).split(" ")
    if len(parts) == 3 and parts[0] == "Synthetic" and parts[2].isdigit():
        return f"{nice(parts[1])} vendor #{parts[2]}"
    return v


st.set_page_config(page_title="Ocazionally Intelligence Engine", page_icon="🎪", layout="centered")

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html {font-size: 18px;}
.stApp {background: #000000; font-family: 'Inter', 'Segoe UI', sans-serif;}
[data-testid="stHeader"] {background: rgba(0,0,0,.85);}
.stApp p, .stApp label, .stApp li, .stApp td, .stApp th, .stApp h1, .stApp h2, .stApp h3, .stApp h4,
.stApp button, .stApp input {font-family: 'Inter', 'Segoe UI', sans-serif;}
.block-container {max-width: 1100px !important; padding-top: 5rem; padding-bottom: 4rem;}

/* hero */
.hero {text-align: center; padding: 2.4rem 2rem; border-radius: 14px; margin-bottom: 1.6rem;
       background: linear-gradient(180deg, #17120A 0%, #050505 100%);
       border: 1px solid #2B2B2B; border-top: 4px solid #F2A900;}
.hero h1 {font-weight: 800; font-size: 2.5rem; letter-spacing: -.5px; margin: 0 0 .7rem 0; padding: 0; color: #FFFFFF;}
.hero p {margin: 0 auto; max-width: 52ch; font-size: 1.1rem; line-height: 1.6; color: #D4D4D4;}

/* headings and body text */
h2, h3 {text-align: center; font-weight: 700 !important; color: #FFFFFF; margin-top: 2rem !important;}
h3 {font-size: 1.6rem !important;}
.lead {text-align: center; font-size: 1.1rem; line-height: 1.65; color: #D4D4D4; max-width: 70ch; margin: 0 auto 1.4rem auto;}
.stApp [data-testid="stCaptionContainer"], .stApp [data-testid="stCaptionContainer"] p {text-align: center; font-size: 1rem; color: #A3A3A3;}
[data-testid="stMarkdownContainer"] li {font-size: 1.05rem; line-height: 1.7;}
code, pre {font-size: .95rem !important;}

/* tabs */
.stTabs [data-baseweb="tab-list"] {justify-content: center; gap: .3rem; border-bottom: 1px solid #262626; flex-wrap: wrap;}
.stTabs [data-baseweb="tab"] {padding: .8rem 1.2rem; height: auto;}
.stTabs [data-baseweb="tab"] p {font-size: 1.05rem; font-weight: 600;}

/* metric cards */
[data-testid="stMetric"] {background: #0D0D0D; border: 1px solid #2B2B2B; border-top: 3px solid #F2A900;
       padding: 1.1rem 1rem; border-radius: 12px;}
[data-testid="stMetric"] {min-height: 8rem; display: flex; flex-direction: column; justify-content: center; align-items: center;}
[data-testid="stMetric"] * {justify-content: center; text-align: center;}
[data-testid="stMetricLabel"] {width: 100%;}
[data-testid="stMetricLabel"] p {font-size: 1rem; color: #A3A3A3; white-space: nowrap;}
[data-testid="stMetricValue"], [data-testid="stMetricValue"] div {width: 100%; font-size: 1.9rem !important; font-weight: 800; color: #FFFFFF;
       white-space: nowrap; overflow: visible !important; text-overflow: clip !important;}

/* step cards */
.step {text-align: center; padding: 1.3rem 1.1rem; border-radius: 12px; min-height: 13rem;
       display: flex; flex-direction: column; justify-content: center;
       background: #0D0D0D; border: 1px solid #2B2B2B;}
.step b {display: block; margin-bottom: .5rem; font-size: 1.1rem; color: #F2A900;}
.step span {font-size: 1rem; line-height: 1.55; color: #D4D4D4;}

/* inputs */
[data-testid="stWidgetLabel"] p {font-size: 1rem; font-weight: 600; color: #E5E5E5;}
[data-baseweb="select"] div, [data-testid="stNumberInput"] input {font-size: 1.05rem;}
[data-testid="stVerticalBlockBorderWrapper"] {background: #0A0A0A; border-color: #2B2B2B !important; border-radius: 12px;}
[data-testid="stExpander"] summary p {font-size: 1.05rem; font-weight: 600;}

/* tables */
.twrap {overflow: auto; border: 1px solid #2B2B2B; border-radius: 12px; margin: .4rem 0 1.2rem 0;}
.otable {width: 100%; border-collapse: collapse; font-size: 1rem;}
.otable thead th {position: sticky; top: 0; background: #151515; color: #F2A900; font-weight: 700;
       text-align: left !important; padding: .85rem 1rem; border-bottom: 1px solid #333;}
.otable td {padding: .8rem 1rem; border-bottom: 1px solid #1E1E1E; color: #F0F0F0; line-height: 1.45; vertical-align: top;}
.otable tbody tr:last-child td {border-bottom: none;}
.otable tbody tr:hover td {background: #0F0F0F;}
.foot {text-align: center; font-size: .9rem; color: #737373; margin-top: 3rem;}
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_data
def csv(name):
    p = D / name
    return pd.read_csv(p) if p.exists() else pd.DataFrame()


@st.cache_data
def text(name):
    p = D / name
    return p.read_text(encoding="utf-8") if p.exists() else "(not available)"


def lead(msg):
    st.markdown(f'<p class="lead">{msg}</p>', unsafe_allow_html=True)


def table(df, max_height=480):
    """Readable HTML table (scales with the page font), or a friendly message if the data is missing."""
    if df is None or len(df) == 0:
        st.info("This result file is not available in the public data folder.")
        return
    fmt = lambda x: f"{x:,.0f}" if abs(x) >= 100 else f"{x:,.2f}"
    html = df.to_html(index=False, border=0, classes="otable", float_format=fmt, na_rep="").replace("\n", "")
    st.markdown(f'<div class="twrap" style="max-height:{max_height}px">{html}</div>', unsafe_allow_html=True)


def style_chart(ch):
    return (ch.configure_axis(labelFontSize=14, titleFontSize=14, labelColor="#D4D4D4", titleColor="#D4D4D4",
                              gridColor="#1F1F1F", domainColor="#333333", tickColor="#333333")
            .configure_legend(labelFontSize=14, labelColor="#D4D4D4", symbolStrokeWidth=3)
            .configure_view(strokeWidth=0))


head = json.loads((D / "headline.json").read_text(encoding="utf-8"))

st.markdown(
    """
<div class="hero">
  <h1>🎪 Ocazionally Intelligence Engine</h1>
  <p>Plan an event with the right vendors, forecast the demand, and make sure new vendors get a fair chance.<br>
  Big Data Analytics project, Adani University.</p>
</div>
""",
    unsafe_allow_html=True,
)

tabs = st.tabs(["Overview", "Package planner", "Demand forecast", "Fairness", "Engineering", "About & ethics"])

# ---------------------------------------------------------------------------------------------------- Overview
with tabs[0]:
    c = st.columns(5)
    for col, (k, lab) in zip(c, [("vendors", "Vendors"), ("customers", "Customers"), ("bookings", "Bookings"),
                                 ("reviews", "Reviews"), ("events", "Platform events")]):
        col.metric(lab, f"{head[k]:,}" if head[k] < 1_000_000 else f"{head[k] / 1e6:.1f}M")
    st.caption(f"Cities covered: {', '.join(head['cities'])}. All numbers come from a simulated marketplace.")

    st.subheader("What the engine does")
    s = st.columns(4)
    steps = [("Learns what events need", "Each event type has a usual set of services, such as food, decor and photography."),
             ("Picks reliable vendors", "For every service it chooses the vendor with the best rating and fewest cancellations."),
             ("Forecasts demand", "It predicts bookings four weeks ahead so vendors can be recruited in time."),
             ("Keeps it fair", "It reserves a slot for new vendors so a lack of history does not lock them out.")]
    for col, (t, d) in zip(s, steps):
        col.markdown(f'<div class="step"><b>{t}</b><span>{d}</span></div>', unsafe_allow_html=True)

    st.subheader("How the project covers the syllabus")
    table(pd.DataFrame([
        ["Introduction to Big Data", "Volume, variety and velocity of a marketplace: 4 million platform events, structured and semi-structured data, a live stream", "Overview, Engineering"],
        ["Frameworks and architectures", "Parquet data lake; the MapReduce programming model (map, shuffle, reduce) benchmarked against Spark DataFrames. HDFS is explained, not run (8 GB laptop)", "Engineering"],
        ["NoSQL", "MongoDB document model, indexes (COLLSCAN to IXSCAN), 3-server replica set with failover, CAP demonstration, shard-key analysis", "Engineering"],
        ["Spark", "DataFrames, SparkSQL, lazy evaluation and the DAG, local Spark with fault-tolerant execution", "Engineering"],
        ["Analytics", "MLlib demand forecasting (GBT), K-Means vendor segments, FP-Growth bundles, Structured Streaming alerts", "Demand forecast, Package planner"],
        ["Ethics and case studies", "Fairness audit and fix for new vendors, privacy rules, honest limits of simulated data", "Fairness, About & ethics"]],
        columns=["Unit", "What was built", "Where to see it"]), max_height=900)

# ------------------------------------------------------------------------------------------- Package planner
with tabs[1]:
    lead("Choose a city and event type. The engine takes the services that event usually books, then picks the most "
         "reliable vendor for each. Costs scale with guest count, and one alternative per service is a new vendor.")
    pk = csv("packages_all.csv")
    if pk.empty:
        st.info("Package data not available.")
    else:
        with st.container(border=True):
            a, b = st.columns(2)
            city = a.selectbox("City", sorted(pk["city"].unique()))
            et = b.selectbox("Event type", sorted(pk["event_type"].unique()), format_func=nice)
            c2, d = st.columns(2)
            guests = c2.number_input("Guests", 10, 5000, TYPICAL.get(et, 60), step=10)
            budget = d.number_input("Budget in ₹ (0 = none)", 0, 5_000_000, 0, step=10_000)
        sub = pk[(pk["city"] == city) & (pk["event_type"] == et)].copy()

        def cost(r):
            f = guests if r["unit"] == "per_plate" else min(2.0, max(0.5, (guests / TYPICAL.get(et, 60)) ** 0.4))
            return int(round((r["price_min"] + r["price_max"]) / 2 * f / 500) * 500)
        sub["cost"] = sub.apply(cost, axis=1)
        services = list(dict.fromkeys(sub["service"]))
        pick = {s: sub[(sub["service"] == s)].sort_values("rank").iloc[0].name for s in services}
        total = lambda: sum(sub.loc[i, "cost"] for i in pick.values())
        while budget and total() > budget:                      # swap in a cheaper, still-reliable vendor
            best, move = 0, None
            for s in services:
                cur = sub.loc[pick[s]]
                for i, r in sub[(sub["service"] == s) & (sub["rank"] == 2)].iterrows():
                    if cur["cost"] - r["cost"] > best:
                        best, move = cur["cost"] - r["cost"], (s, i)
            if not move:
                break
            pick[move[0]] = move[1]

        st.write("")
        m1, m2, m3 = st.columns(3)
        m1.metric("Estimated total", inr(total()))
        m2.metric("Services covered", len(pick))
        if budget:
            gap = budget - total()
            m3.metric("Against budget", inr(abs(gap)), "under budget" if gap >= 0 else "over budget",
                      delta_color="normal" if gap >= 0 else "inverse")
        else:
            m3.metric("Cost per guest", inr(total() / max(guests, 1)))

        st.subheader("Recommended package")
        rows = [{"Service": nice(s), "Vendor": vendor_name(sub.loc[i, "vendor"]), "Record": sub.loc[i, "label"],
                 "Rating": f"{float(sub.loc[i, 'rating']):.2f}", "Cancels": f"{sub.loc[i, 'cancel_rate']:.0%}",
                 "Est. cost": inr(sub.loc[i, "cost"])} for s, i in pick.items()]
        table(pd.DataFrame(rows))
        with st.expander("Alternatives, including one new vendor per service"):
            alt_df = sub[~sub.index.isin(pick.values())][["service", "vendor", "label", "rating", "cost"]]
            alt_df = alt_df.assign(service=alt_df["service"].map(nice), vendor=alt_df["vendor"].map(vendor_name),
                                   rating=alt_df["rating"].map(lambda x: f"{float(x):.2f}"), cost=alt_df["cost"].map(inr))
            table(alt_df.rename(columns=str.capitalize))

# ------------------------------------------------------------------------------------------ Demand forecast
with tabs[2]:
    lead("Weekly bookings predicted four weeks ahead, for a full year the model never saw. Gradient-boosted "
         "trees (Spark MLlib) are compared with two simple baselines.")
    fm, fp = csv("forecast_metrics.csv"), csv("forecast_predictions.csv")
    if not fm.empty:
        st.subheader("Model accuracy against baselines")
        table(fm)
    if not fp.empty:
        st.subheader("Actual vs forecast")
        with st.container(border=True):
            a, b = st.columns(2)
            city = a.selectbox("City", ["All cities"] + sorted(fp["city"].unique()), key="fc_city")
            cat = b.selectbox("Category", ["All categories"] + sorted(fp["category"].unique()), key="fc_cat", format_func=nice)
        g = fp.copy()
        g = g if city == "All cities" else g[g["city"] == city]
        g = g if cat == "All categories" else g[g["category"] == cat]
        w = g.groupby("week")[["y", "pred_gbt"]].sum().rename(columns={"y": "Actual", "pred_gbt": "Forecast"})
        w = w.reset_index().melt("week", var_name="Series", value_name="Bookings")
        w["week"] = pd.to_datetime(w["week"])
        line = (alt.Chart(w).mark_line(strokeWidth=3)
                .encode(x=alt.X("week:T", title=None),
                        y=alt.Y("Bookings:Q", title="Bookings per week"),
                        color=alt.Color("Series:N", scale=alt.Scale(domain=["Actual", "Forecast"], range=[GREY, GOLD]),
                                        legend=alt.Legend(orient="top", title=None)),
                        strokeDash=alt.StrokeDash("Series:N", scale=alt.Scale(domain=["Actual", "Forecast"], range=[[1, 0], [6, 3]]), legend=None),
                        tooltip=["week:T", "Series:N", alt.Tooltip("Bookings:Q", format=",.0f")])
                .properties(height=360, width="container"))
        st.altair_chart(style_chart(line))

        st.subheader("Vendor-recruitment alerts")
        st.caption("Weeks where forecast demand is closest to (or above) the capacity of all verified vendors.")
        t = fp.sort_values("utilisation", ascending=False).head(8)[["week", "city", "category", "pred_gbt", "weekly_capacity", "utilisation"]]
        t = t.assign(category=t["category"].map(nice), utilisation=(t["utilisation"] * 100).round().astype(int).astype(str) + "%",
                     pred_gbt=t["pred_gbt"].round().astype(int), weekly_capacity=t["weekly_capacity"].round().astype(int))
        table(t.rename(columns={"week": "Week", "city": "City", "category": "Category", "pred_gbt": "Forecast",
                                "weekly_capacity": "Capacity", "utilisation": "Used"}))
    img = D / "figures" / "forecast_actual_vs_predicted.png"
    if img.exists():
        with st.expander("Test year: actual vs forecast (full figure)"):
            st.image(str(img))

# ------------------------------------------------------------------------------------------------- Fairness
with tabs[3]:
    lead("A ranking built on track record can lock newcomers out: they are not shown because they have no history, "
         "and they get no history because they are not shown. We measured that, then tested a fix.")
    fa, fo = csv("fairness_audit.csv"), csv("fairness_policies.csv")
    if not fa.empty:
        st.subheader("Exposure of new vs established vendors")
        st.caption("Vendors of the same quality. 1.0 is fair. Below 0.8 is flagged (the 'four-fifths rule' from employment law).")
        fd = fa.rename(columns={"ratio_new_vs_established": "ratio"})[["quality_band", "ratio"]]
        bars = (alt.Chart(fd).mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
                .encode(x=alt.X("quality_band:N", title="Vendor quality band", sort=None, axis=alt.Axis(labelAngle=0)),
                        y=alt.Y("ratio:Q", title="Exposure ratio (new / established)"),
                        color=alt.condition(alt.datum.ratio < 0.8, alt.value(RED), alt.value(GOLD)),
                        tooltip=["quality_band:N", alt.Tooltip("ratio:Q", format=".2f")]))
        rule = alt.Chart(pd.DataFrame({"y": [0.8]})).mark_rule(strokeDash=[5, 4], color=GREY).encode(y="y:Q")
        st.altair_chart(style_chart((bars + rule).properties(height=340, width="container")))
    if not fo.empty:
        st.subheader("Who the top-5 recommendation lists show")
        show = pd.DataFrame({"Policy": fo["policy"], "Share to new vendors": fo["share_new"].map("{:.1%}".format),
                             "Fairness ratio": fo["di_ratio"].map("{:.2f}".format),
                             "Vendors shown": fo["distinct_vendors"], "Avg true quality": fo["true_quality"].map("{:.3f}".format)})
        table(show)
        st.success("Reserving one of five slots for the best newcomer costs about 1% in average vendor quality and "
                   "lifts new vendors from a few percent of slots to a fair share.")
    st.warning("The bias was built into the simulation so the audit had something to find. This shows the method works, "
               "not that a real platform behaves this way. 'True quality' exists only because the data is simulated.")
    img = D / "figures" / "fairness_audit.png"
    if img.exists():
        with st.expander("Fairness audit figure"):
            st.image(str(img))

# --------------------------------------------------------------------------------------------- Engineering
with tabs[4]:
    e1, e2, e3, e4 = st.tabs(["Spark vs MapReduce", "MongoDB", "Streaming", "Segments & bundles"])
    with e1:
        st.subheader("MapReduce-style vs Spark DataFrame")
        st.caption("The same job, run both ways, with the same answer.")
        table(csv("results.csv"))
    with e2:
        st.subheader("Queries without and with an index")
        table(csv("mongo_index_results.csv"))
        st.subheader("Shard-key analysis on 3 shards")
        table(csv("shard_key_analysis.csv"))
        st.subheader("Replication, failover and the CAP theorem")
        st.caption("3-server replica set demonstration log.")
        st.code(text("replica_set_demo_log.txt"), language=None)
    with e3:
        st.subheader("Live booking stream")
        st.caption("Spark Structured Streaming output.")
        st.code(text("streaming_results.txt")[-1800:], language=None)
    with e4:
        st.subheader("Vendor segments (K-Means)")
        table(csv("vendor_segment_profile.csv"))
        st.subheader("Bundles booked together (FP-Growth)")
        table(csv("bundles_by_event_type.csv"))

# ----------------------------------------------------------------------------------------- About & ethics
with tabs[5]:
    st.subheader("Honest limits")
    st.markdown(
        "- **Simulated data.** Ocazionally has not launched. Bookings, prices, ratings and cancellations are generated "
        "to test the engine; accuracy here shows the pipeline works, not how it will perform on real events.\n"
        "- **Real vendor directory, anonymised.** Vendors came from public Google listings in Ahmedabad. Their names are "
        "never shown with simulated numbers.\n"
        "- **Scaled prototype.** Everything ran on one 8 GB laptop. The same Spark code runs unchanged on a cluster.\n"
        "- **HDFS** is described, not run. The MapReduce comparison measures the programming model on Spark's engine.")
    st.subheader("Privacy")
    lead("No customer names or contact details exist in the analytics tables; ids are hashed. Vendor phone numbers "
         "stay in a private, git-ignored folder and are not part of this dashboard.")
    st.subheader("Next step")
    lead("A customer-facing app with sign-up, login and a chat assistant that turns a typed request, such as "
         "“birthday for 60 people in Rajkot, ₹80,000”, into a plan using this engine, for any city that has vendors.")

st.markdown('<div class="foot">Ocazionally Intelligence Engine · Adani University · simulated data</div>', unsafe_allow_html=True)