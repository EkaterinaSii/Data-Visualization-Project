import os
from typing import Optional, List


import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, Input, Output, State, dcc, html, dash_table, ctx

# -----------------------------
# Configuration
# -----------------------------
DATA_PATH = os.environ.get("BLOOD_PRESSURE_CSV", "Blood_Pressure.csv")

ALLOWED_COLUMNS = [
    "Patient_ID",
    "Year",
    "Country",
    "WHO_Region",
    "Income_Level",
    "ISO2_Country_Code",
    "Age",
    "Age_Group",
    "Sex",
    "BMI",
    "BMI_Category",
    "Smoking_Status",
    "Physical_Activity",
    "Diet_Salt_Intake",
    "Stress_Level",
    "Diabetes",
    "Family_Hx_Hypertension",
    "Systolic_BP_mmHg",
    "Diastolic_BP_mmHg",
    "Pulse_Pressure_mmHg",
    "Mean_Arterial_Pressure",
    "Heart_Rate_bpm",
    "BP_Category",
    "Country_HTN_Prevalence_pct",
    "Age_Category",
    "BP_Category_2",
]

MAP_METRICS = {
    "Country_HTN_Prevalence_pct": "HTN prevalence (%)",
    "Systolic_BP_mmHg": "Mean systolic BP (mmHg)",
    "Diastolic_BP_mmHg": "Mean diastolic BP (mmHg)",
    "Pulse_Pressure_mmHg": "Mean pulse pressure (mmHg)",
    "Mean_Arterial_Pressure": "Mean arterial pressure (mmHg)",
    "Heart_Rate_bpm": "Mean heart rate (bpm)",
    "BMI": "Mean BMI",
    "Age": "Mean age",
}

NUMERIC_COLUMNS = [
    "Year",
    "Age",
    "BMI",
    "Systolic_BP_mmHg",
    "Diastolic_BP_mmHg",
    "Pulse_Pressure_mmHg",
    "Mean_Arterial_Pressure",
    "Heart_Rate_bpm",
    "Country_HTN_Prevalence_pct",
]

TEAL_SCALE = [
    [0.0, "#e0f7f7"],
    [0.2, "#b8e6e4"],
    [0.4, "#7fd0cb"],
    [0.6, "#4eb7b1"],
    [0.8, "#218f8b"],
    [1.0, "#0b5f60"],
]

INFO_HELP = {
    "Avg systolic": [
        html.B("What it is: "),
        "\nSystolic blood pressure is the top number in a blood pressure reading. It reflects pressure in the arteries when the heart contracts.",
        html.Br(), html.Br(),
        html.B("Typical range: "),
        "\n90–119 mmHg is a typical adult range. \n120–129 mmHg is elevated. \n130 mmHg and above may suggest hypertension.",
        html.Br(), html.Br(),
        html.B("Why it matters: "),
        "\nHigher systolic pressure can increase the risk of heart disease, stroke, "
        "and kidney damage."
    ],

    "Avg diastolic": [
        html.B("What it is: "),
        "\nDiastolic blood pressure is the bottom number in a blood pressure reading. It reflects pressure in the arteries when the heart relaxes between beats.",
        html.Br(), html.Br(),
        html.B("Typical range: "),
        "\n60–79 mmHg is commonly considered typical. \n80 mmHg or above may indicate hypertension.",
        html.Br(), html.Br(),
        html.B("Why it matters: "),
        "\nHigh diastolic pressure can strain blood vessels and raise cardiovascular risk over time."
    ],

    "Avg MAP": [
        html.B("What it is: "),
        "\nMean arterial pressure (MAP) estimates the average pressure pushing blood through the arteries during one cardiac cycle.",
        html.Br(), html.Br(),
        html.B("Typical range: "),
        "\n70–100 mmHg is often considered adequate for organ perfusion in adults.",
        html.Br(), html.Br(),
        html.B("Why it matters: "),
        "\nToo high can stress blood vessels, while too low can reduce blood flow to vital organs."
    ],
    
    "Avg pulse pressure": [
        html.B("What it is: "),
        "\nPulse pressure is the difference between systolic and diastolic pressure.",
        html.Br(), html.Br(),
        html.B("Typical range: "),
        "\n30–50 mmHg is common in many healthy adults.",
        html.Br(), html.Br(),
        html.B("Why it matters: "),
        "\nPersistently high pulse pressure may be linked with arterial stiffness and higher cardiovascular risk."
    ],

    "Avg heart rate": [
        html.B("What it is: "),
        "\nHeart rate is the number of heartbeats per minute.",
        html.Br(), html.Br(),
        html.B("Typical range: "),
        "\nFor many adults at rest, around 60–100 bpm is a common reference range.",
        html.Br(), html.Br(),
        html.B("Why it matters: "),
        "\nPersistently high resting heart rate can be associated with stress, poor fitness, or heart problems."
    ],

    "HTN prevalence": [
        html.B("What it is: "),
        "\nHypertension prevalence is the share of people in the dataset or country estimated to have high blood pressure.",
        html.Br(), html.Br(),
        html.B("Typical range: "),
        "\nHigher percentages suggest a larger population burden of cardiovascular risk.",
        html.Br(), html.Br(),
        html.B("Why it matters: "),
        "\nHigher prevalence can indicate greater expected risk of stroke, heart disease, and kidney complications at population level."
    ],

    "BMI": [
        html.B("What it is: "),
        "\nBody mass index (BMI) relates weight to height.",
        html.Br(), html.Br(),
        html.B("Typical categories:"),
        "\nUnder 18.5: underweight\n"
        "18.5–24.9: healthy range\n"
        "25.0–29.9: overweight\n"
        "30 or more: obesity\n\n",
        html.B("Why it matters: "),
        "\nHigher BMI is often associated with greater risk of hypertension, diabetes, and cardiovascular disease."
    ],
}

# -----------------------------
# Data helpers
# -----------------------------
def load_data(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Could not find '{path}'. Put Blood_Pressure.csv in the same folder "
            "or set BLOOD_PRESSURE_CSV to the full path."
        )

    df = pd.read_csv(path)

    available = [c for c in ALLOWED_COLUMNS if c in df.columns]
    missing = [c for c in ALLOWED_COLUMNS if c not in df.columns]
    if missing:
        print("Warning: missing columns:", missing)

    df = df[available].copy()

    for col in NUMERIC_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in df.columns:
        if col not in NUMERIC_COLUMNS:
            df[col] = df[col].astype("string").fillna("Unknown")

    return df


def dropdown_options(series: pd.Series) -> List[dict]:
    values = sorted(v for v in series.dropna().astype(str).unique() if v != "Unknown")
    return [{"label": v, "value": v} for v in values]


def apply_filters(
    df: pd.DataFrame,
    year_range: Optional[List[int]],
    sex: Optional[str],
    age_group: Optional[str],
    bmi_category: Optional[str],
    smoking: Optional[str],
    physical: Optional[str],
    salt: Optional[str],
    stress: Optional[str],
    diabetes: Optional[str],
    family_hx: Optional[str],
) -> pd.DataFrame:
    dff = df.copy()

    if year_range and "Year" in dff.columns:
        dff = dff[(dff["Year"] >= year_range[0]) & (dff["Year"] <= year_range[1])]
    if sex:
        dff = dff[dff["Sex"] == sex]
    if age_group:
        dff = dff[dff["Age_Group"] == age_group]
    if bmi_category:
        dff = dff[dff["BMI_Category"] == bmi_category]
    if smoking:
        dff = dff[dff["Smoking_Status"] == smoking]
    if physical:
        dff = dff[dff["Physical_Activity"] == physical]
    if salt:
        dff = dff[dff["Diet_Salt_Intake"] == salt]
    if stress:
        dff = dff[dff["Stress_Level"] == stress]
    if diabetes:
        dff = dff[dff["Diabetes"] == diabetes]
    if family_hx:
        dff = dff[dff["Family_Hx_Hypertension"] == family_hx]

    return dff


def aggregate_for_map(dff: pd.DataFrame, metric: str) -> pd.DataFrame:
    if dff.empty:
        return pd.DataFrame(columns=["Country", metric, "Patients", "WHO_Region", "Income_Level", "ISO2_Country_Code"])

    agg = (
        dff.groupby("Country", dropna=False)
        .agg(
            **{
                metric: (metric, "mean"),
                "Patients": ("Patient_ID", "count"),
                "WHO_Region": ("WHO_Region", lambda x: x.mode().iloc[0] if not x.mode().empty else "Unknown"),
                "Income_Level": ("Income_Level", lambda x: x.mode().iloc[0] if not x.mode().empty else "Unknown"),
                "ISO2_Country_Code": (
                    "ISO2_Country_Code",
                    lambda x: x.dropna().iloc[0] if not x.dropna().empty else ""
                ),
            }
        )
        .reset_index()
    )

    agg[metric] = agg[metric].round(2)
    agg["ISO2_Country_Code"] = agg["ISO2_Country_Code"].astype(str).str.upper()

    return agg

def help_icon(text):
    return html.Span(
        [
            html.Span("?", className="help-icon-mark"),
            html.Div(text, className="help-tooltip-box"),
        ],
        className="help-tooltip-wrap",
    )

def stat_card(label: str, value: str, tooltip=None):
    label_row = html.Div(
        [
            html.Span(label, className="stat-label-text"),
            help_icon(tooltip) if tooltip else None,
        ],
        className="stat-label-row",
    )

    return html.Div(
        [
            html.Div(label_row, className="stat-label"),
            html.Div(value, className="stat-value"),
        ],
        className="stat-card",
    )


def info_cards_for_df(dff: pd.DataFrame, country: Optional[str] = None) -> List:
    if dff.empty:
        return [html.Div("No data for the selected filters.", className="info-empty")]

    title = country if country else "Dataset overview"
    years = f"{int(dff['Year'].min())}–{int(dff['Year'].max())}" if "Year" in dff.columns and dff["Year"].notna().any() else "N/A"

    cards = [
        stat_card("Scope", title),
        stat_card("Patients", f"{len(dff):,}"),
        stat_card("Countries", f"{dff['Country'].nunique():,}" if "Country" in dff.columns else "N/A"),
        stat_card("Years", years),
        stat_card("Avg systolic", f"{dff['Systolic_BP_mmHg'].mean():.1f} mmHg" if dff["Systolic_BP_mmHg"].notna().any() else "N/A", INFO_HELP["Avg systolic"]),
        stat_card("Avg diastolic", f"{dff['Diastolic_BP_mmHg'].mean():.1f} mmHg" if dff["Diastolic_BP_mmHg"].notna().any() else "N/A", INFO_HELP["Avg diastolic"]),
        stat_card("Avg MAP", f"{dff['Mean_Arterial_Pressure'].mean():.1f} mmHg" if dff["Mean_Arterial_Pressure"].notna().any() else "N/A", INFO_HELP["Avg MAP"]),
        stat_card("Avg pulse pressure", f"{dff['Pulse_Pressure_mmHg'].mean():.1f} mmHg" if dff["Pulse_Pressure_mmHg"].notna().any() else "N/A", INFO_HELP["Avg pulse pressure"]),
        stat_card("Avg heart rate", f"{dff['Heart_Rate_bpm'].mean():.1f} bpm" if dff["Heart_Rate_bpm"].notna().any() else "N/A", INFO_HELP["Avg heart rate"]),
        stat_card("HTN prevalence", f"{dff['Country_HTN_Prevalence_pct'].mean():.1f}%" if dff["Country_HTN_Prevalence_pct"].notna().any() else "N/A", INFO_HELP["HTN prevalence"]),
        stat_card("BMI", f"{dff['BMI'].mean():.1f}" if dff["BMI"].notna().any() else "N/A", INFO_HELP["BMI"]),
    ]
    return cards


def make_map(map_df: pd.DataFrame, metric: str, selected_country: Optional[str] = None, compact: bool = False):
    if map_df.empty:
        fig = go.Figure()
        fig.update_layout(
            template="plotly_white",
            annotations=[{
                "text": "No data available for the selected filters",
                "showarrow": False,
                "xref": "paper",
                "yref": "paper",
                "x": 0.5,
                "y": 0.5,
                "font": {"size": 18},
            }],
            margin=dict(l=10, r=10, t=40, b=10),
            height=280 if compact else 680,
        )
        return fig

    fig = px.choropleth(
        map_df,
        locations="Country",
        locationmode="country names",
        color=metric,
        hover_name="Country",
        custom_data=["Patients", "WHO_Region", "Income_Level"],
        color_continuous_scale=TEAL_SCALE,
        projection="natural earth",
    )

    if selected_country:
        fig.add_trace(
            go.Scattergeo(
                locations=map_df["Country"],
                locationmode="country names",
                text=map_df["Country"],
                mode="text",
                hoverinfo="none",
                hovertemplate=None,
                textfont=dict(size=14, color="black"),
                showlegend=False,
            )
        )

    fig.update_traces(
        hovertemplate=(
            "<b>%{location}</b><br>"
            + f"{MAP_METRICS.get(metric, metric)}: "
            + "%{z:.2f}<br>"
            + "Patients: %{customdata[0]:,.0f}<br>"
            + "Income level: %{customdata[2]}<extra></extra>"
        ),
        marker_line_color="white",
        marker_line_width=0.5,
    )

    geo_kwargs = dict(
        showframe=False,
        showcoastlines=False,
        bgcolor="rgba(0,0,0,0)",
    )

    if selected_country:
        geo_kwargs.update(
            fitbounds="locations",
            visible=False,
        )

    fig.update_geos(**geo_kwargs)
    fig.update_layout(
        template="plotly_white",
        margin=dict(l=10, r=10, t=50, b=10),
        height=360 if compact else 720,
        coloraxis_colorbar=dict(title=MAP_METRICS.get(metric, metric)),
        title=dict(
            text=f"Global map: {MAP_METRICS.get(metric, metric)}" if not selected_country else f"Focused map: {selected_country}",
            x=0.02,
            xanchor="left",
        ),
        clickmode="event+select",
    )
    
    return fig


def empty_figure(message: str, height: int = 320):
    fig = go.Figure()
    fig.update_layout(
        template="plotly_white",
        annotations=[{
            "text": message,
            "showarrow": False,
            "xref": "paper",
            "yref": "paper",
            "x": 0.5,
            "y": 0.5,
            "font": {"size": 16},
        }],
        margin=dict(l=10, r=10, t=40, b=10),
        height=height,
    )
    return fig


def make_country_figures(country_df: pd.DataFrame):
    if country_df.empty:
        msg = empty_figure("No country data to display.")
        return msg, msg, msg, msg

    trend = (
        country_df.groupby("Year", as_index=False)[["Systolic_BP_mmHg", "Diastolic_BP_mmHg", "Mean_Arterial_Pressure"]]
        .mean()
        .sort_values("Year")
    )
    fig_trend = px.line(
        trend,
        x="Year",
        y=["Systolic_BP_mmHg", "Diastolic_BP_mmHg", "Mean_Arterial_Pressure"],
        markers=True,
        title="Trend over years",
    )
    fig_trend.update_layout(
        template="plotly_white",
        margin=dict(l=10, r=10, t=40, b=10),
        legend_title_text="Metric",
        height=340
    )

    age_bp = (
        country_df.groupby("Age_Group", as_index=False)[["Systolic_BP_mmHg", "Diastolic_BP_mmHg"]]
        .mean()
        .sort_values("Age_Group")
    )
    fig_age = px.bar(
        age_bp,
        x="Age_Group",
        y=["Systolic_BP_mmHg", "Diastolic_BP_mmHg"],
        barmode="group",
        title="BP by age group",
    )
    fig_age.update_layout(
        template="plotly_white",
        margin=dict(l=10, r=10, t=40, b=10),
        legend_title_text="Metric",
        height=420
    )

    sex_box = px.box(
        country_df,
        x="Sex",
        y="Systolic_BP_mmHg",
        color="Sex",
        points="outliers",
        title="Systolic BP by sex",
    )
    sex_box.update_layout(
        template="plotly_white",
        margin=dict(l=10, r=10, t=40, b=10),
        showlegend=False,
        height=420
    )


    bp_order = [
        "Normal",
        "Elevated",
        "Stage 1 Hypertension",
        "Stage 2 Hypertension",
        "Severe Hypertension",
    ]


    existing_bp = [
        bp for bp in bp_order
        if bp in country_df["BP_Category_2"].dropna().astype(str).unique().tolist()
    ]

    if not existing_bp:
        msg = empty_figure("No BP category data available.", height=420)
        return fig_trend, fig_age, sex_box, msg

    total_counts = (
        country_df.groupby("BP_Category_2")
        .size()
        .reindex(existing_bp, fill_value=0)
        .reset_index(name="Count")
    )

    age_counts = (
        country_df.groupby(["BP_Category_2", "Age_Group"])
        .size()
        .reset_index(name="Count")
    )

    fig = go.Figure()

    spacing = 1.4
    x_positions = [i * spacing for i in range(len(existing_bp))]

    # Background gray bars
    fig.add_trace(
        go.Bar(
            x=x_positions,
            y=total_counts["Count"],
            name="Total",
            marker_color="lightgray",
            opacity=0.65,
            width=1.4,
            hoverinfo="none",

        )
    )

    age_order = [
        "Infant (0-1)",
        "Early Childhood (1-5)",
        "Middle Childhood (6-10)",
        "Early Adolescence (11-15)",
        "Late Adolescence (16-18)",
        "Young Adult (19-29)",
        "Adult (30-39)",
        "Middle-Aged (40-49)",
        "Middle-Aged Senior (50-59)",
        "Young Elderly (60-69)",
        "Elderly (70-79)",
        "Very Elderly (80+)",
    ]

    age_groups = [
        age for age in age_order
        if age in country_df["Age_Group"].dropna().astype(str).unique().tolist()
    ]

    n_groups = max(len(age_groups), 1)
    bar_width = min(1.5 / n_groups, 0.15)

    for i, age_group in enumerate(age_groups):
        age_series = (
            age_counts.loc[age_counts["Age_Group"] == age_group]
            .set_index("BP_Category_2")["Count"]
            .reindex(existing_bp, fill_value=0)
        )

        offset = (i - (n_groups - 1) / 2) * bar_width
        x_age = [x + offset for x in x_positions]

        fig.add_trace(
            go.Bar(
                x=x_age,
                y=age_series.values,
                name=age_group,
                width=bar_width,
                customdata=[[age_group]] * len(age_series.values),
                hovertemplate=(
                    "Age group: %{customdata[0]}<br>"
                    "Count: %{y}<extra></extra>"
                ),
            )
        )

    fig.update_layout(
        template="plotly_white",
        barmode="overlay",
        title="People count by BP category and age group",
        xaxis_title="Blood pressure category",
        yaxis_title="People count",
        margin=dict(l=10, r=10, t=50, b=60),
        height=420,
        legend_title_text="Age group",
    )

    fig.update_xaxes(
        tickmode="array",
        tickvals=x_positions,
        ticktext=existing_bp,
        range=[x_positions[0] - 0.8, x_positions[-1] + 0.8],
    )

    return fig, fig_trend, fig_age, sex_box

def overview_note():
    return html.Div(
        [
            html.H3("Why this matters", className="panel-title"),
            html.P(
                "High blood pressure is one of the most important cardiovascular risk factors worldwide. "
                "It can increase the risk of stroke, heart disease, kidney damage, and other long-term complications. "
                "Exploring differences across countries and population groups can help identify patterns, risk factors, "
                "and areas where prevention and treatment may be especially important.",
                className="overview-note-text",
            ),
        ],
        className="overview-note-wrap",
    )
    

# -----------------------------
# App boot
# -----------------------------
df = load_data(DATA_PATH)

app = Dash(__name__)
server = app.server

year_min = int(df["Year"].min()) if "Year" in df.columns and df["Year"].notna().any() else 2000
year_max = int(df["Year"].max()) if "Year" in df.columns and df["Year"].notna().any() else 2024

app.title = "Blood Pressure Dashboard"

app.layout = html.Div(
    [
        dcc.Store(id="selected-country"),
        dcc.Store(id="selected-sex"),
        dcc.Store(id="selected-smoking"),
        html.Div(
            [
                html.H1("Blood Pressure Dashboard", className="page-title"),
                html.P("Data visualization dashboard for exploring blood pressure metrics across countries and demographics.\nClick on the map to focus on a specific country, or use the filters to explore different subsets of the data.\nMade by Ekaterina Siikavirta",
                    className="page-subtitle",
                ),
            ],
            className="header",
        ),
        html.Div(
            [
                html.Div(
                    [
                        html.H3("Filters", className="panel-title"),
                        html.Label("Map metric"),
                        dcc.Dropdown(
                            id="metric-dropdown",
                            options=[{"label": v, "value": k} for k, v in MAP_METRICS.items()],
                            value="Country_HTN_Prevalence_pct",
                            clearable=False,
                        ),
                        html.Label("Year range"),
                        dcc.RangeSlider(
                            id="year-range",
                            min=year_min,
                            max=year_max,
                            step=1,
                            value=[year_min, year_max],
                            marks={year_min: str(year_min), year_max: str(year_max)},
                            tooltip={"placement": "bottom", "always_visible": False},
                        ),
                        
                        html.Label("Sex"),
                        html.Div(
                            [
                                html.Button(
                                    html.Img(src="/assets/male-icon.png", className="sex-icon"),
                                    id="sex-male-btn",
                                    n_clicks=0,
                                    className="sex-btn",
                                ),
                                html.Button(
                                    html.Img(src="/assets/female-icon.png", className="sex-icon"),
                                    id="sex-female-btn",
                                    n_clicks=0,
                                    className="sex-btn",
                                ),
                            ],
                            className="sex-toggle-row",
                        ),

                        html.Label("Age group"),
                        dcc.Dropdown(id="age-group-dropdown", options=dropdown_options(df["Age_Group"]), placeholder="All"),
                        html.Label("BMI category"),
                        dcc.Dropdown(id="bmi-category-dropdown", options=dropdown_options(df["BMI_Category"]), placeholder="All"),
                        
                        html.Label("Smoking status"),
                        html.Div(
                            [
                                html.Button(
                                    [
                                        html.Img(src="/assets/no-smoking-icon.png", className="smoking-icon"),
                                        html.Span("Non-Smoker", className="smoking-btn-label"),
                                    ],
                                    id="smoking-non-btn",
                                    n_clicks=0,
                                    className="smoking-btn",
                                ),
                                html.Button(
                                    [
                                        html.Img(src="/assets/smoking-status.png", className="smoking-icon"),
                                        html.Span("Current Smoker", className="smoking-btn-label"),
                                    ],
                                    id="smoking-current-btn",
                                    n_clicks=0,
                                    className="smoking-btn",
                                ),
                                html.Button(
                                    [
                                        html.Img(src="/assets/ex-smoker.png", className="smoking-icon"),
                                        html.Span("Ex-Smoker", className="smoking-btn-label"),
                                    ],
                                    id="smoking-ex-btn",
                                    n_clicks=0,
                                    className="smoking-btn",
                                ),
                            ],
                            className="smoking-toggle-row",
                        ),


                        html.Label("Physical activity"),
                        dcc.Dropdown(id="physical-dropdown", options=dropdown_options(df["Physical_Activity"]), placeholder="All"),
                        html.Label("Diet salt intake"),
                        dcc.Dropdown(id="salt-dropdown", options=dropdown_options(df["Diet_Salt_Intake"]), placeholder="All"),
                        html.Label("Stress level"),
                        dcc.Dropdown(id="stress-dropdown", options=dropdown_options(df["Stress_Level"]), placeholder="All"),
                        html.Label("Diabetes"),
                        dcc.Dropdown(id="diabetes-dropdown", options=dropdown_options(df["Diabetes"]), placeholder="All"),
                        html.Label("Family history of hypertension"),
                        dcc.Dropdown(id="family-dropdown", options=dropdown_options(df["Family_Hx_Hypertension"]), placeholder="All"),
                        html.Button("Reset country selection", id="reset-country-btn", n_clicks=0, className="reset-btn"),
                    ],
                    className="filters-panel",
                ),
                html.Div(
                    [html.Div(id="main-content")],
                    className="content-panel",
                ),
            ],
            className="dashboard-shell",
        ),
    ],
    className="page",
)

@app.callback(
    Output("selected-smoking", "data"),
    Input("smoking-non-btn", "n_clicks"),
    Input("smoking-current-btn", "n_clicks"),
    Input("smoking-ex-btn", "n_clicks"),
    State("selected-smoking", "data"),
    prevent_initial_call=True,
)
def update_selected_smoking(non_clicks, current_clicks, ex_clicks, current_smoking):
    trigger = ctx.triggered_id

    if trigger == "smoking-non-btn":
        return None if current_smoking == "Non-Smoker" else "Non-Smoker"

    if trigger == "smoking-current-btn":
        return None if current_smoking == "Current Smoker" else "Current Smoker"

    if trigger == "smoking-ex-btn":
        return None if current_smoking == "Ex-Smoker" else "Ex-Smoker"

    return current_smoking

@app.callback(
    Output("smoking-non-btn", "className"),
    Output("smoking-current-btn", "className"),
    Output("smoking-ex-btn", "className"),
    Input("selected-smoking", "data"),
)
def style_smoking_buttons(selected_smoking):
    non_class = "smoking-btn active" if selected_smoking == "Non-Smoker" else "smoking-btn"
    current_class = "smoking-btn active" if selected_smoking == "Current Smoker" else "smoking-btn"
    ex_class = "smoking-btn active" if selected_smoking == "Ex-Smoker" else "smoking-btn"
    return non_class, current_class, ex_class


@app.callback(
    Output("sex-male-btn", "className"),
    Output("sex-female-btn", "className"),
    Input("selected-sex", "data"),
)
def style_sex_buttons(selected_sex):
    male_class = "sex-btn active" if selected_sex == "Male" else "sex-btn"
    female_class = "sex-btn active" if selected_sex == "Female" else "sex-btn"
    return male_class, female_class

@app.callback(
    Output("selected-sex", "data"),
    Input("sex-male-btn", "n_clicks"),
    Input("sex-female-btn", "n_clicks"),
    State("selected-sex", "data"),
    prevent_initial_call=True,
)

def update_selected_sex(male_clicks, female_clicks, current_sex):
    trigger = ctx.triggered_id

    if trigger == "sex-male-btn":
        return None if current_sex == "Male" else "Male"

    if trigger == "sex-female-btn":
        return None if current_sex == "Female" else "Female"

    return current_sex


@app.callback(
    Output("selected-country", "data"),
    Input("reset-country-btn", "n_clicks"),
    Input("metric-dropdown", "value"),
    Input("year-range", "value"),
    Input("selected-sex", "data"),
    Input("age-group-dropdown", "value"),
    Input("bmi-category-dropdown", "value"),
    Input("selected-smoking", "data"),
    Input("physical-dropdown", "value"),
    Input("salt-dropdown", "value"),
    Input("stress-dropdown", "value"),
    Input("diabetes-dropdown", "value"),
    Input("family-dropdown", "value"),
    Input({"type": "dynamic-map", "index": "main"}, "clickData"),
    State("selected-country", "data"),
    prevent_initial_call=True,
)


def update_selected_country(
    reset_clicks,
    metric,
    year_range,
    sex,
    age_group,
    bmi_category,
    smoking,
    physical,
    salt,
    stress,
    diabetes,
    family_hx,
    click_data,
    current_country,
):
    trigger = ctx.triggered_id

    if trigger == "reset-country-btn":
        return None

    if isinstance(trigger, dict) and trigger.get("type") == "dynamic-map":
        if click_data and click_data.get("points"):
            return click_data["points"][0].get("location")

    return current_country


@app.callback(
    Output("main-content", "children"),
    Input("selected-country", "data"),
    Input("metric-dropdown", "value"),
    Input("year-range", "value"),
    Input("selected-sex", "data"),
    Input("age-group-dropdown", "value"),
    Input("bmi-category-dropdown", "value"),
    Input("selected-smoking", "data"),
    Input("physical-dropdown", "value"),
    Input("salt-dropdown", "value"),
    Input("stress-dropdown", "value"),
    Input("diabetes-dropdown", "value"),
    Input("family-dropdown", "value"),
)
def render_content(
    selected_country,
    metric,
    year_range,
    sex,
    age_group,
    bmi_category,
    smoking,
    physical,
    salt,
    stress,
    diabetes,
    family_hx,
):
    dff = apply_filters(
        df,
        year_range,
        sex,
        age_group,
        bmi_category,
        smoking,
        physical,
        salt,
        stress,
        diabetes,
        family_hx,
    )

    map_df = aggregate_for_map(dff, metric)

    if not selected_country or selected_country not in map_df["Country"].astype(str).tolist():
        map_fig = make_map(map_df, metric, compact=False)
        return html.Div(
            [
                html.Div(
                    dcc.Graph(
                        id={"type": "dynamic-map", "index": "main"},
                        figure=map_fig,
                        className="graph-fill",
                        style={"height": "100%", "width": "100%"},
                        config={
                            "displayModeBar": False,
                            "responsive": True,
                            "scrollZoom": True,
                            "staticPlot": False,
                        },
                    ),
                    className="overview-map-wrap",
                ),
                html.Div(
                    [
                        html.Div(
                            [
                                html.H3("Overall dataset information", className="panel-title"),
                                html.Div(info_cards_for_df(dff), className="stats-grid"),
                                html.Div(
                                    [
                                        html.H4("Top 5 countries by selected metric"),
                                        dash_table.DataTable(
                                            data=map_df.sort_values(metric, ascending=False).head(5).to_dict("records"),
                                            columns=[
                                                {"name": "Country", "id": "Country"},
                                                {"name": MAP_METRICS.get(metric, metric), "id": metric},
                                            ],
                                            style_table={"overflowX": "auto"},
                                            style_cell={"padding": "8px", "fontFamily": "Arial", "fontSize": 13},
                                            style_header={"fontWeight": "bold"},
                                            page_size=10,
                                        ),
                                    ],
                                    className="datatable-wrap",
                                ),
                            ],
                            className="overview-info-wrap",
                        ),
                        overview_note(),
                    ],
                    className="overview-right-column",
                ),
            ],
            className="overview-layout",
        )

    country_df = dff[dff["Country"] == selected_country].copy()
    compact_map_df = map_df[map_df["Country"] == selected_country].copy()
    map_fig = make_map(compact_map_df, metric, selected_country=selected_country, compact=True)
    fig_trend, fig_age, fig_sex, fig_bp_cat = make_country_figures(country_df)

    return html.Div(
        [
            html.Div(
                [
                    html.Div(
                        dcc.Graph(
                            id={"type": "dynamic-map", "index": "main"},
                            figure=map_fig,
                            config={"displayModeBar": False, "responsive": True},
                            className="graph-fill",
                        ),
                        className="country-map-wrap",
                    ),
                    html.Div(
                        [
                            html.H3(f"{selected_country} overview", className="panel-title"),
                            html.Div(info_cards_for_df(country_df, selected_country), className="stats-grid country-stats"),
                        ],
                        className="country-info-wrap",
                    ),
                ],
                className="country-left-column",
            ),
            html.Div(
                [
                    html.H3("Specific plots", className="panel-title"),
                    html.Div(
                        [
                            dcc.Graph(figure=fig_trend, config={"displayModeBar": False, "responsive": True}, className="plot-card"),
                            dcc.Graph(figure=fig_age, config={"displayModeBar": False, "responsive": True}, className="plot-card"),
                            dcc.Graph(figure=fig_sex, config={"displayModeBar": False, "responsive": True}, className="plot-card"),
                            dcc.Graph(figure=fig_bp_cat, config={"displayModeBar": False, "responsive": True}, className="plot-card"),
                        ],
                        className="plots-grid",
                    ),
                ],
                className="country-plots-wrap",
            ),
        ],
        className="country-layout",
    )


app.index_string = """
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
            html, body {
                margin: 0;
                min-height: 100%;
                font-family: Arial, Helvetica, sans-serif;
                background: #2F4F4F;
                color: #142426;
            }
            .page {
                padding: 16px;
                min-height: 100vh;
                box-sizing: border-box;
            }
            .header {
                background: white;
                border: 1px solid #d9eceb;
                border-radius: 16px;
                padding: 18px 22px;
                margin-bottom: 16px;
                box-shadow: 0 4px 14px rgba(8, 62, 61, 0.06);
            }
            .page-title {
                margin: 0;
                color: #0b5f60;
            }
            .page-subtitle {
                margin: 8px 0 0;
                color: #476466;
            }
            .dashboard-shell {
                display: grid;
                grid-template-columns: 270px minmax(0, 1fr);
                gap: 16px;
                align-items: stretch;
                min-height: calc(100vh - 190px);
            }
            .filters-panel,
            .overview-map-wrap,
            .overview-info-wrap,
            .country-map-wrap,
            .country-info-wrap,
            .country-plots-wrap {
                background: white;
                border: 1px solid #d9eceb;
                border-radius: 16px;
                box-shadow: 0 4px 14px rgba(8, 62, 61, 0.06);
            }
            .filters-panel {
                padding: 20px;
                height: 100%;
                min-height: 100%;
                overflow-y: auto;
                box-sizing: border-box;
            }

            .filters-panel label {
                display: block;
                font-size: 20px;
                font-weight: 700;
                margin: 14px 0 8px;
                color: #234244;
            }

            .panel-title {
                margin: 0 0 16px;
                color: #0b5f60;
            }
            .reset-btn {
                width: 100%;
                margin-top: 16px;
                background: #0b5f60;
                color: white;
                border: 0;
                border-radius: 10px;
                padding: 12px 14px;
                cursor: pointer;
                font-weight: 700;
            }
            .content-panel {
                min-width: 0;
                width: 100%;
            }
            .overview-layout {
                display: grid;
                grid-template-columns: minmax(0, 2.5fr) minmax(380px, 1fr);
                gap: 16px;
                align-items: start;
            }

            .country-map-wrap,
            .country-info-wrap,
            .country-plots-wrap,
            .overview-info-wrap,
            .overview-note-wrap {
                padding: 18px;
                min-width: 0;
                box-sizing: border-box;
            }

            .overview-map-wrap {
                padding: 6px 10px;
                min-width: 0;
                box-sizing: border-box;
                min-height: calc(100vh - 190px);
                display: flex;
                align-items: center;
                justify-content: center;
            }

            .graph-fill {
                width: 100%;
                height: 100%;
            }
            .stats-grid {
                display: grid;
                grid-template-columns: repeat(2, minmax(0, 1fr));
                gap: 10px;
            }

            .sex-toggle-row {
                display: flex;
                gap: 12px;
                align-items: center;
                margin-top: 6px;
            }

            .sex-btn {
                flex: 1;
                background: #f8fdfd;
                border: 1px solid #cfe5e4;
                border-radius: 12px;
                padding: 12px;
                cursor: pointer;
                display: flex;
                align-items: center;
                justify-content: center;
                min-height: 64px;
                transition: 0.2s ease;
            }

            .sex-btn:hover {
                background: #eef8f8;
                border-color: #9ed0cd;
            }

            .sex-icon {
                width: 48px;
                height: 48px;
                object-fit: contain;
            }


            .sex-btn.active {
                background: #dff2f1;
                border: 2px solid #0b5f60;
            }


            .smoking-toggle-row {
                display: grid;
                grid-template-columns: 1fr;
                gap: 10px;
                margin-top: 4px;
            }

            .smoking-btn {
                width: 100%;
                background: #f8fdfd;
                border: 1px solid #cfe5e4;
                border-radius: 12px;
                padding: 12px 10px;
                cursor: pointer;
                display: flex;
                align-items: center;
                justify-content: flex-start;
                gap: 12px;
                transition: 0.2s ease;
                text-align: left;
                min-height: 56px;
            }

            .smoking-btn:hover {
                background: #eef8f8;
                border-color: #9ed0cd;
            }

            .smoking-btn.active {
                background: #dff2f1;
                border: 2px solid #0b5f60;
            }

            .smoking-icon {
                width: 38px;
                height: 38px;
                object-fit: contain;
                flex-shrink: 0;
            }

            .smoking-btn-label {
                font-size: 15px;
                font-weight: 700;
                color: #234244;
                display: flex;
                align-items: center;
                height: 100%;
            }

            .country-stats {
                grid-template-columns: repeat(2, minmax(0, 1fr));
            }
            .stat-card {
                border: 1px solid #d9eceb;
                border-radius: 12px;
                padding: 12px;
                background: #f8fdfd;
            }
            .stat-label {
                font-size: 12px;
                color: #527173;
                margin-bottom: 5px;
            }
            .stat-label-row {
                display: inline-flex;
                align-items: center;
                gap: 6px;
                flex-wrap: wrap;
            }

            .help-tooltip-wrap {
                position: relative;
                display: inline-block;
                vertical-align: middle;
            }

            .help-icon-mark {
                display: inline-flex;
                align-items: center;
                justify-content: center;
                width: 16px;
                height: 16px;
                border-radius: 50%;
                background: #dff2f1;
                color: #0b5f60;
                font-size: 11px;
                font-weight: 700;
                cursor: help;
                border: 1px solid #9ed0cd;
                line-height: 1;
            }

            .help-tooltip-box {
                display: none;
                position: absolute;
                top: 24px;
                right: 0;
                width: 460px;
                max-width: min(460px, 90vw);
                background: #163234;
                color: white;
                padding: 18px 20px;
                border-radius: 14px;
                font-size: 16px;
                line-height: 1.4;
                box-shadow: 0 10px 28px rgba(0, 0, 0, 0.24);
                z-index: 9999;
                text-align: left;
                word-break: break-word;
            }

            .help-tooltip-wrap:hover .help-tooltip-box {
                display: block;
            }
            .stat-value {
                font-size: 18px;
                font-weight: 700;
                color: #163234;
            }
            .datatable-wrap {
                margin-top: 16px;
            }
            .country-layout {
                display: grid;
                grid-template-columns: minmax(430px, 0.95fr) minmax(760px, 1.85fr);
                gap: 16px;
                align-items: stretch;
                min-height: calc(100vh - 190px);
            }
            .country-left-column {
                display: grid;
                grid-template-rows: minmax(360px, 0.9fr) minmax(0, 1.1fr);
                gap: 16px;
                min-height: calc(100vh - 190px);
            }
            .country-map-wrap {
                min-height: 360px;
                display: flex;
                align-items: center;
                justify-content: center;
            }
            .country-info-wrap {
                min-height: 0;
            }
            .country-plots-wrap {
                display: flex;
                flex-direction: column;
                min-height: calc(100vh - 190px);
            }
            .plots-grid {
                display: grid;
                grid-template-columns: repeat(2, minmax(0, 1fr));
                gap: 14px;
                flex: 1;
                align-content: stretch;
            }
            .plot-card {
                min-height: 400px;
            }
            .info-empty {
                padding: 14px;
                border: 1px dashed #b8d7d5;
                border-radius: 12px;
                color: #476466;
                background: #fbffff;
            }

            .overview-right-column {
                display: grid;
                grid-template-rows: auto auto;
                gap: 16px;
                min-width: 0;
            }

            .overview-note-wrap {
                background: white;
                border: 1px solid #d9eceb;
                border-radius: 16px;
                box-shadow: 0 4px 14px rgba(8, 62, 61, 0.06);
                padding: 18px;
                box-sizing: border-box;
            }

            .overview-note-text {
                margin: 0;
                color: #335456;
                font-size: 15px;
                line-height: 1.7;
            }

            

            @media (max-width: 1400px) {
                .overview-layout {
                    grid-template-columns: minmax(0, 1.9fr) minmax(340px, 1fr);
                }
            }
            @media (max-width: 1200px) {
                .overview-layout,
                .country-layout,
                .dashboard-shell {
                    grid-template-columns: 1fr;
                }
                .country-left-column {
                    grid-template-rows: auto auto;
                    min-height: auto;
                }
                .country-plots-wrap,
                .country-layout {
                    min-height: auto;
                }
                .filters-panel {
                    position: static;
                    max-height: none;
                }
                .overview-map-wrap {
                    min-height: auto;
                }
            }
            @media (max-width: 900px) {
                .plots-grid,
                .stats-grid {
                    grid-template-columns: 1fr;
                }
                .page {
                    padding: 10px;
                }
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
"""

if __name__ == "__main__":
    app.run(debug=False)
