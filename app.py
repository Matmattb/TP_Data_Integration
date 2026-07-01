"""
Data Quality Dashboard — Road-traffic personal-injury accidents (BAAC 2024)
TP Data Integration & Applications (EFREI ST2DLDI)

Run with:
    pip install streamlit pandas plotly numpy
    streamlit run app.py

Put the 4 CSVs (caract / lieux / vehicules / usagers -2024.csv) in the same folder,
or set the path in the sidebar.
"""
import os
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------
FILES = {
    "caract": "caract-2024.csv",
    "lieux": "lieux-2024.csv",
    "vehicules": "vehicules-2024.csv",
    "usagers": "usagers-2024.csv",
}

SENTINELS = {"", " ", "-1", "N/A", "n/a", "nan", "NaN", "#N/A", "null", "NULL"}

DOMAINS = {
    ("usagers", "catu"): {"1", "2", "3"},
    ("usagers", "sexe"): {"1", "2"},
    ("usagers", "grav"): {"1", "2", "3", "4"},
    ("usagers", "trajet"): {"0", "1", "2", "3", "4", "5", "9"},
    ("caract", "lum"): {"1", "2", "3", "4", "5"},
    ("caract", "agg"): {"1", "2"},
    ("caract", "atm"): {"1", "2", "3", "4", "5", "6", "7", "8", "9"},
    ("caract", "col"): {"1", "2", "3", "4", "5", "6", "7"},
    ("caract", "int"): {"1", "2", "3", "4", "5", "6", "7", "8", "9"},
}

GRAV_LABELS = {"1": "Unharmed", "2": "Killed", "3": "Hospitalised", "4": "Slight injury"}

# ----------------------------------------------------------------------------
# Processing functions (pure, testable)
# ----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_tables(data_dir: str):
    raw = {}
    for name, fname in FILES.items():
        path = os.path.join(data_dir, fname)
        raw[name] = pd.read_csv(path, sep=";", dtype=str, encoding="utf-8")
    return raw


def normalize_missing(s: pd.Series) -> pd.Series:
    s2 = s.astype("string").str.strip()
    return s2.mask(s2.isin(SENTINELS))


def missing_report(df: pd.DataFrame) -> pd.DataFrame:
    d = df.apply(normalize_missing)
    n_miss = d.isna().sum()
    pct = (n_miss / len(d) * 100).round(2)
    return (
        pd.DataFrame({"column": df.columns, "n_missing": n_miss.values, "pct_missing": pct.values})
        .sort_values("pct_missing", ascending=False)
        .reset_index(drop=True)
    )


def true_missing_ratio(df: pd.DataFrame) -> float:
    d = df.apply(normalize_missing)
    return round(d.isna().mean().mean() * 100, 2)


def geo_frame(caract: pd.DataFrame) -> pd.DataFrame:
    g = caract[["Num_Acc", "lat", "long"]].copy()
    for col in ["lat", "long"]:
        g[col] = pd.to_numeric(g[col].str.replace(",", ".", regex=False), errors="coerce")
    g["outside_mainland"] = ~(g["lat"].between(41, 51.5) & g["long"].between(-5.5, 9.8))
    g["impossible"] = (~g["lat"].between(-90, 90)) | (~g["long"].between(-180, 180))
    return g


def age_frame(usagers: pd.DataFrame) -> pd.DataFrame:
    u = usagers[["Num_Acc", "an_nais", "grav"]].copy()
    u["an_nais"] = pd.to_numeric(u["an_nais"], errors="coerce")
    u["age"] = 2024 - u["an_nais"]
    return u


def categorical_conformity(raw: dict) -> pd.DataFrame:
    rows = []
    for (tbl, col), valid in DOMAINS.items():
        s = raw[tbl][col].astype("string").str.strip()
        out = s[~s.isin(valid) & s.notna()]
        n_sentinel = int((out == "-1").sum())
        n_true = int(len(out) - n_sentinel)
        rows.append({
            "variable": f"{tbl}.{col}",
            "out_of_domain (invalid)": n_true,
            "sentinel -1 (unknown)": n_sentinel,
            "status": "compliant" if n_true == 0 else "anomaly",
        })
    return pd.DataFrame(rows)


def summary_table(raw: dict) -> pd.DataFrame:
    return pd.DataFrame([
        {
            "table": name,
            "rows": len(df),
            "columns": df.shape[1],
            "unique_accidents": df["Num_Acc"].nunique(),
            "exact_duplicates": int(df.duplicated().sum()),
            "repeated_key": int(df.duplicated(subset=["Num_Acc"]).sum()),
            "avg % missing": true_missing_ratio(df),
        }
        for name, df in raw.items()
    ])


def quality_score(raw: dict) -> int:
    """Global 0-100 score combining completeness, uniqueness and validity."""
    completeness = 100 - np.mean([true_missing_ratio(df) for df in raw.values()])
    exact_dups = sum(df.duplicated().sum() for df in raw.values())
    total_rows = sum(len(df) for df in raw.values())
    uniqueness = 100 - (exact_dups / total_rows * 100)
    conf = categorical_conformity(raw)
    validity = 100 if conf["out_of_domain (invalid)"].sum() == 0 else 80
    return int(round(0.5 * completeness + 0.25 * uniqueness + 0.25 * validity))


# ----------------------------------------------------------------------------
# UI
# ----------------------------------------------------------------------------
def main():
    st.set_page_config(page_title="Data Quality — BAAC 2024", page_icon="🚗", layout="wide")
    st.title("Data Quality Dashboard — Road-injury accidents (BAAC 2024)")
    st.caption("Data profiling & quality — TP Data Integration & Applications")

    with st.sidebar:
        st.header("Settings")
        data_dir = st.text_input("CSV folder", value=".")
        st.markdown("---")
        st.markdown(
            "**Sentinels detected as missing:**\n\n"
            "`-1`, `N/A`, empty, `null`…\n\n"
            "The BAAC format codes unknown values as `-1`: a naïve `isna()` would miss them."
        )

    # Load
    try:
        raw = load_tables(data_dir)
    except FileNotFoundError as e:
        st.error(f"File not found: {e.filename}. Check the folder in the sidebar.")
        st.stop()

    # ---- Global KPIs ----
    n_acc = raw["caract"]["Num_Acc"].nunique()
    n_users = len(raw["usagers"])
    n_veh = len(raw["vehicules"])
    killed = (raw["usagers"]["grav"].str.strip() == "2").sum()
    score = quality_score(raw)

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Accidents", f"{n_acc:,}")
    c2.metric("People", f"{n_users:,}")
    c3.metric("Vehicles", f"{n_veh:,}")
    c4.metric("Killed", f"{killed:,}")
    c5.metric("Quality score", f"{score}/100")

    st.markdown("---")

    tab_over, tab_missing, tab_valid, tab_dup, tab_map = st.tabs(
        ["📊 Overview", "🕳️ Missing", "✅ Validity", "👯 Duplicates", "🗺️ Map"]
    )

    # ================= OVERVIEW =================
    with tab_over:
        st.subheader("Summary per table")
        summ = summary_table(raw)
        st.dataframe(summ, use_container_width=True, hide_index=True)

        colA, colB = st.columns(2)
        with colA:
            fig = px.bar(summ, x="table", y="avg % missing",
                         title="Average % of missing values per table",
                         color="avg % missing", color_continuous_scale="Reds", text="avg % missing")
            st.plotly_chart(fig, use_container_width=True)
        with colB:
            grav = raw["usagers"]["grav"].str.strip().map(GRAV_LABELS)
            gcount = grav.value_counts().reset_index()
            gcount.columns = ["Severity", "Count"]
            fig2 = px.pie(gcount, names="Severity", values="Count",
                          title="People by injury severity", hole=0.4)
            st.plotly_chart(fig2, use_container_width=True)

    # ================= MISSING =================
    with tab_missing:
        st.subheader("Missing values per column (sentinels included)")
        table = st.selectbox("Table", list(FILES.keys()), key="miss_tbl")
        rep = missing_report(raw[table])
        fig = px.bar(rep, x="pct_missing", y="column", orientation="h",
                     title=f"% missing — {table}", color="pct_missing",
                     color_continuous_scale="OrRd")
        fig.update_layout(yaxis={"categoryorder": "total ascending"}, height=500)
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(rep, use_container_width=True, hide_index=True)
        st.info(
            "High rates (`lartpc`, `occutc`, pedestrian fields) are **structural** "
            "(conditional fields) and should not be imputed."
        )

    # ================= VALIDITY =================
    with tab_valid:
        st.subheader("Validity checks")

        g = geo_frame(raw["caract"])
        u = age_frame(raw["usagers"])
        v1, v2, v3, v4 = st.columns(4)
        v1.metric("Non-convertible coords", int(g["lat"].isna().sum()))
        v2.metric("Impossible coords", int(g["impossible"].sum()))
        v3.metric("Ages < 0 / > 110", int(((u["age"] < 0) | (u["age"] > 110)).sum()))
        v4.metric("Missing an_nais", int(u["an_nais"].isna().sum()))

        st.markdown("#### Compliance of categorical variables with their nomenclature")
        conf = categorical_conformity(raw)
        st.dataframe(conf, use_container_width=True, hide_index=True)

        st.markdown("#### Age distribution")
        bins = [0, 14, 17, 24, 34, 44, 54, 64, 74, 200]
        labels = ["0-14", "15-17", "18-24", "25-34", "35-44", "45-54", "55-64", "65-74", "75+"]
        u2 = u.dropna(subset=["age"]).copy()
        u2["band"] = pd.cut(u2["age"], bins=bins, labels=labels)
        agec = u2["band"].value_counts().sort_index().reset_index()
        agec.columns = ["Age band", "Count"]
        fig = px.bar(agec, x="Age band", y="Count", title="People by age band",
                     color="Count", color_continuous_scale="Blues")
        st.plotly_chart(fig, use_container_width=True)

    # ================= DUPLICATES =================
    with tab_dup:
        st.subheader("Duplicate analysis")
        d1, d2 = st.columns(2)
        with d1:
            st.markdown("**Strictly identical rows (true duplicates)**")
            exact = pd.DataFrame([
                {"table": n, "exact_duplicates": int(df.duplicated().sum())}
                for n, df in raw.items()
            ])
            st.dataframe(exact, use_container_width=True, hide_index=True)
        with d2:
            st.markdown("**Repeated `Num_Acc` key**")
            keyd = pd.DataFrame([
                {"table": n, "rows": len(df), "unique_accidents": df["Num_Acc"].nunique(),
                 "repeated_key": int(df.duplicated(subset=["Num_Acc"]).sum())}
                for n, df in raw.items()
            ])
            st.dataframe(keyd, use_container_width=True, hide_index=True)

        st.markdown("#### Nature of the repetitions in `lieux` (segments per accident)")
        dist = (raw["lieux"]["Num_Acc"].value_counts().value_counts().sort_index()
                .rename_axis("n_segments").reset_index(name="n_accidents"))
        fig = px.bar(dist, x="n_segments", y="n_accidents", text="n_accidents",
                     title="Number of road segments per accident (lieux)")
        st.plotly_chart(fig, use_container_width=True)
        st.warning(
            "⚠️ Repeated `Num_Acc` in `lieux` are **NOT** duplicates: they are "
            "multi-lane junctions (1:N relationship). Do not apply `drop_duplicates` blindly — "
            "pick one main segment per accident or model `lieux` as a 1:N dimension."
        )

    # ================= MAP =================
    with tab_map:
        st.subheader("Geographic distribution of accidents")
        g = geo_frame(raw["caract"]).dropna(subset=["lat", "long"])
        metro_only = st.checkbox("Mainland only", value=True)
        gm = g[~g["outside_mainland"]] if metro_only else g
        sample = gm.sample(min(8000, len(gm)), random_state=0)
        fig = px.scatter_map(
            sample, lat="lat", lon="long", zoom=4.2 if metro_only else 1.5,
            height=600, opacity=0.4, map_style="carto-positron",
            title=f"{len(sample):,} accidents shown (sample)",
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            f"{int(g['outside_mainland'].sum()):,} accidents outside mainland bounds "
            "(mostly overseas territories, valid)."
        )


if __name__ == "__main__":
    main()
