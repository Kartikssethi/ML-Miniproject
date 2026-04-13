import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pydeck as pdk
import re
import json
from urllib.error import URLError
from urllib.parse import urlparse, unquote, urlencode
from urllib.request import Request, urlopen
from pathlib import Path

# -------------------------
# LOAD MODEL
# -------------------------
model = joblib.load("model.pkl")

st.set_page_config(page_title="F1 Race Simulator", layout="wide")


def apply_insane_ui_theme():
    """Apply an F1-inspired dark theme with red accent styling."""
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Titillium+Web:wght@400;600;700&family=Rajdhani:wght@500;700&display=swap');

        .stApp {
            background:
                linear-gradient(180deg, #07090f 0%, #090b11 40%, #07090d 100%);
            color: #f5f7fa;
            font-family: 'Titillium Web', sans-serif;
        }

        h1, h2, h3, .section-title {
            font-family: 'Rajdhani', sans-serif;
            letter-spacing: 0.3px;
        }

        .hero-shell {
            border: 1px solid #232833;
            border-top: 3px solid #e10600;
            background: linear-gradient(135deg, #0d1119 0%, #0a0d14 100%);
            border-radius: 10px;
            padding: 16px 20px;
            margin: 4px 0 14px 0;
            box-shadow: 0 10px 28px rgba(0, 0, 0, 0.38);
            position: relative;
            overflow: hidden;
        }

        .hero-content {
            position: relative;
            z-index: 2;
            max-width: 62%;
        }

        .hero-pattern {
            position: absolute;
            right: 12px;
            top: 16px;
            width: 38%;
            height: calc(100% - 32px);
            border: 1px solid #24314a;
            border-radius: 10px;
            background-image: repeating-linear-gradient(
                -45deg,
                rgba(255, 255, 255, 0.08) 0,
                rgba(255, 255, 255, 0.08) 16px,
                rgba(255, 255, 255, 0.01) 16px,
                rgba(255, 255, 255, 0.01) 34px
            );
            opacity: 0.72;
            z-index: 1;
        }

        .hero-kicker {
            font-size: 0.78rem;
            color: #ff4c47;
            text-transform: uppercase;
            letter-spacing: 1.1px;
            margin-bottom: 6px;
            font-weight: 700;
        }

        .hero-title {
            font-size: 3.2rem;
            margin: 0;
            line-height: 0.95;
            color: #ffffff;
            text-transform: uppercase;
            letter-spacing: 0.8px;
        }

        .hero-sub {
            color: #c0c6d1;
            margin-top: 7px;
            font-size: 0.95rem;
        }

        .section-shell {
            border: 1px solid #1d2330;
            border-radius: 10px;
            padding: 10px 12px;
            background: linear-gradient(180deg, #0c111a 0%, #0a0f17 100%);
            margin-bottom: 10px;
        }

        .section-title {
            margin: 2px 0 2px 0;
            font-size: 1.05rem;
            color: #f8fbff;
        }

        .section-subtitle {
            margin: 0 0 6px 0;
            font-size: 0.85rem;
            color: #99a7bc;
        }

        div[data-testid="stMetric"] {
            border: 1px solid #212a38;
            border-radius: 10px;
            background: linear-gradient(180deg, #0d121b, #0b0f17);
            padding: 10px 12px;
        }

        [data-testid="stMetricLabel"] {
            color: #a9b5c7;
            font-weight: 600;
        }

        [data-testid="stMetricValue"] {
            color: #ffffff;
        }

        div.stButton > button {
            border: 1px solid #b90f0f;
            background: linear-gradient(180deg, #f01919, #c30000);
            color: white;
            font-weight: 700;
            border-radius: 999px;
            padding: 0.55rem 1rem;
            transition: filter 120ms ease, transform 120ms ease;
            box-shadow: 0 6px 16px rgba(166, 0, 0, 0.45);
        }

        div.stButton > button:hover {
            transform: translateY(-1px);
            filter: brightness(1.06);
            box-shadow: 0 8px 20px rgba(166, 0, 0, 0.52);
        }

        .stDataFrame, div[data-testid="stDataEditor"] {
            border: 1px solid #1c2432;
            border-radius: 10px;
        }

        .watch-shell {
            margin-top: 10px;
            border: 1px solid #1c2534;
            border-radius: 10px;
            background: linear-gradient(180deg, #0d1020, #0b0e19);
            padding: 14px;
        }

        .watch-title {
            font-family: 'Rajdhani', sans-serif;
            font-size: 2.1rem;
            font-weight: 700;
            color: #f7f9ff;
            letter-spacing: 0.5px;
            margin-bottom: 10px;
        }

        .watch-row {
            display: grid;
            grid-template-columns: 1fr 1fr 1.5fr;
            gap: 12px;
            align-items: stretch;
        }

        .watch-card {
            background: #000000;
            border: 1px solid #232323;
            border-radius: 10px;
            min-height: 120px;
            display: flex;
            align-items: center;
            justify-content: center;
            text-align: center;
            color: #ffffff;
            font-family: 'Rajdhani', sans-serif;
            font-size: 2rem;
            font-weight: 700;
            padding: 10px;
        }

        .watch-pattern {
            border: 1px solid #1f2736;
            border-radius: 10px;
            min-height: 120px;
            background-image: repeating-linear-gradient(
                -45deg,
                rgba(255, 255, 255, 0.06) 0,
                rgba(255, 255, 255, 0.06) 16px,
                rgba(255, 255, 255, 0.01) 16px,
                rgba(255, 255, 255, 0.01) 32px
            );
            position: relative;
        }

        .watch-caption {
            margin-top: 10px;
            text-align: center;
            color: #eef3ff;
            font-size: 1.65rem;
            font-family: 'Rajdhani', sans-serif;
            font-weight: 700;
        }

        .watch-caption span {
            border-bottom: 4px solid #f2f4f8;
            padding-bottom: 1px;
        }

        .double-red-lines {
            margin: 14px 0 12px 0;
            display: grid;
            gap: 4px;
        }

        .double-red-lines div {
            height: 10px;
            background: #e10600;
            border-radius: 2px;
        }

        .circuit-banner-title {
            font-family: 'Rajdhani', sans-serif;
            font-size: 3rem;
            font-weight: 700;
            color: #f8f9ff;
            letter-spacing: 0.6px;
            margin: 4px 0 8px 0;
        }

        @media (max-width: 900px) {
            .watch-row {
                grid-template-columns: 1fr;
            }
            .watch-title {
                font-size: 1.6rem;
            }
            .hero-title {
                font-size: 2.15rem;
            }
            .hero-content {
                max-width: 100%;
            }
            .hero-pattern {
                position: relative;
                width: 100%;
                height: 86px;
                right: auto;
                top: auto;
                margin-top: 12px;
            }
            .watch-card {
                font-size: 1.5rem;
            }
            .watch-caption {
                font-size: 1.2rem;
            }
            .circuit-banner-title {
                font-size: 2rem;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def section_header(title: str, subtitle: str = ""):
    st.markdown(
        (
            "<div class='section-shell'>"
            f"<div class='section-title'>{title}</div>"
            f"<div class='section-subtitle'>{subtitle}</div>"
            "</div>"
        ),
        unsafe_allow_html=True,
    )


def render_watch_section(country: str):
    """Render a stylized where-to-watch band similar to official F1 pages."""
    regional_watch_map = {
        "india": ("FanCode", "TATA Play\nFanCode Sports"),
        "united kingdom": ("Sky Sports F1", "NOW Sports"),
        "united states": ("ESPN", "F1 TV Pro"),
        "italy": ("Sky Sport F1", "NOW TV"),
    }

    ckey = str(country).strip().lower()
    left, right = regional_watch_map.get(ckey, ("F1 TV", "Local Broadcast"))

    st.markdown(
        (
            "<div class='watch-shell'>"
            "<div class='watch-title'>WHERE TO WATCH</div>"
            "<div class='watch-row'>"
            f"<div class='watch-card'>{left}</div>"
            f"<div class='watch-card'>{right.replace(chr(10), '<br/>')}</div>"
            "<div class='watch-pattern'></div>"
            "</div>"
            "<div class='watch-caption'><span>Broadcast Information</span></div>"
            "</div>"
        ),
        unsafe_allow_html=True,
    )


def render_circuit_banner():
    st.markdown(
        """
        <div class='double-red-lines'>
            <div></div>
            <div></div>
        </div>
        <div class='circuit-banner-title'>CIRCUIT</div>
        """,
        unsafe_allow_html=True,
    )


apply_insane_ui_theme()
st.markdown(
    """
    <div class='hero-shell'>
    <div class='hero-content'>
            <div class='hero-kicker'>Official Style Race Hub</div>
    <h1 class='hero-title'>RACE WEEKEND SIMULATOR</h1>
            <div class='hero-sub'>Configure the weekend, inspect the circuit, and launch race simulations with a broadcast-style interface.</div>
    </div>
    <div class='hero-pattern'></div>
    </div>
    """,
    unsafe_allow_html=True,
)

if "grid_state" not in st.session_state:
    st.session_state["grid_state"] = {}

if "quali_state" not in st.session_state:
    st.session_state["quali_state"] = {}


def render_starting_grid_visual(setup_df: pd.DataFrame):
    """Render a visual, lane-style starting grid from selected grid positions."""
    st.markdown(
        """
        <style>
        .grid-track {
            border-radius: 14px;
            padding: 12px;
            background: linear-gradient(180deg, #0b1018 0%, #090d14 100%);
            border: 1px solid #232d3b;
        }
        .grid-row {
            display: flex;
            gap: 10px;
            margin-bottom: 10px;
        }
        .grid-slot {
            flex: 1;
            min-height: 68px;
            border-radius: 10px;
            border: 1px solid #2a3444;
            background: rgba(12, 16, 24, 0.9);
            padding: 8px 10px;
            display: flex;
            align-items: center;
            justify-content: space-between;
        }
        .grid-slot.empty {
            opacity: 0.35;
            border-style: dashed;
        }
        .grid-pos {
            font-weight: 700;
            color: #ff3f3f;
            margin-right: 8px;
            white-space: nowrap;
        }
        .grid-driver {
            color: #f2f6fc;
            font-size: 0.92rem;
            line-height: 1.2;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
            margin-right: 8px;
        }
        .grid-car {
            font-size: 1.1rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    position_to_driver = dict(zip(setup_df["Grid"], setup_df["Driver"]))
    max_pos = int(max(position_to_driver.keys())) if position_to_driver else 0
    max_pos = max(2, max_pos)

    html_rows = ["<div class='grid-track'>"]

    for pos in range(1, max_pos + 1, 2):
        left_driver = position_to_driver.get(pos)
        right_driver = position_to_driver.get(pos + 1)

        left_html = (
            f"<div class='grid-slot'><span class='grid-pos'>P{pos}</span>"
            f"<span class='grid-driver'>{left_driver}</span><span class='grid-car'>🏎️</span></div>"
            if left_driver
            else f"<div class='grid-slot empty'><span class='grid-pos'>P{pos}</span><span class='grid-driver'>Empty</span><span class='grid-car'>▫️</span></div>"
        )
        right_html = (
            f"<div class='grid-slot'><span class='grid-pos'>P{pos + 1}</span>"
            f"<span class='grid-driver'>{right_driver}</span><span class='grid-car'>🏎️</span></div>"
            if right_driver
            else f"<div class='grid-slot empty'><span class='grid-pos'>P{pos + 1}</span><span class='grid-driver'>Empty</span><span class='grid-car'>▫️</span></div>"
        )

        html_rows.append(f"<div class='grid-row'>{left_html}{right_html}</div>")

    html_rows.append("</div>")
    st.markdown("".join(html_rows), unsafe_allow_html=True)


def render_circuit_experience(circuit_row: pd.Series):
    """Render a stylish circuit info panel with an interactive map."""

    @st.cache_data(show_spinner=False)
    def extract_circuit_layout_image(page_url: str):
        """Fetch a high-quality circuit image from Wikimedia APIs."""
        if not isinstance(page_url, str) or not page_url.startswith("http"):
            return None

        parsed = urlparse(page_url)
        title = unquote(parsed.path.split("/")[-1]).replace("_", " ").strip()
        if not title:
            return None

        api_base = "https://en.wikipedia.org/w/api.php?"
        headers = {"User-Agent": "F1-Race-Simulator/1.0 (Streamlit)"}

        def fetch_json(params: dict):
            query = urlencode(params)
            req = Request(f"{api_base}{query}", headers=headers)
            with urlopen(req, timeout=10) as response:
                return json.loads(response.read().decode("utf-8", errors="ignore"))

        try:
            # 1) Get all page images and prioritize layout/track/map files.
            images_data = fetch_json(
                {
                    "action": "query",
                    "format": "json",
                    "titles": title,
                    "prop": "images",
                    "imlimit": "max",
                }
            )

            pages = images_data.get("query", {}).get("pages", {})
            first_page = next(iter(pages.values()), {})
            image_entries = first_page.get("images", [])

            preferred_titles = []
            for item in image_entries:
                image_title = item.get("title", "")
                lower = image_title.lower()

                # Keep only candidate track images and avoid logos/flags/icons.
                include = (
                    ("circuit" in lower or "track" in lower or "layout" in lower or "map" in lower)
                    and (lower.endswith(".svg") or lower.endswith(".png") or lower.endswith(".jpg") or lower.endswith(".jpeg"))
                )
                exclude = any(x in lower for x in ["flag", "logo", "icon", "location map", "country"])

                if include and not exclude:
                    preferred_titles.append(image_title)

            # Prefer SVG layout files first, then PNG/JPG.
            preferred_titles.sort(key=lambda t: (0 if t.lower().endswith(".svg") else 1, len(t)))

            if preferred_titles:
                imageinfo_data = fetch_json(
                    {
                        "action": "query",
                        "format": "json",
                        "titles": preferred_titles[0],
                        "prop": "imageinfo",
                        "iiprop": "url",
                    }
                )
                image_pages = imageinfo_data.get("query", {}).get("pages", {})
                image_page = next(iter(image_pages.values()), {})
                imageinfo = image_page.get("imageinfo", [])
                if imageinfo and imageinfo[0].get("url"):
                    return imageinfo[0]["url"]

            # 2) Fallback: use high-resolution page image thumbnail/original.
            pageimage_data = fetch_json(
                {
                    "action": "query",
                    "format": "json",
                    "titles": title,
                    "prop": "pageimages",
                    "piprop": "thumbnail|original",
                    "pithumbsize": "1600",
                }
            )
            fallback_pages = pageimage_data.get("query", {}).get("pages", {})
            fallback_page = next(iter(fallback_pages.values()), {})

            thumb = fallback_page.get("thumbnail", {}).get("source")
            original = fallback_page.get("original", {}).get("source")
            return original or thumb
        except (URLError, TimeoutError, ValueError, KeyError, json.JSONDecodeError):
            return None
    st.markdown(
        """
        <style>
        .circuit-card {
            border-radius: 14px;
            padding: 14px;
            background: linear-gradient(135deg, #0c121b 0%, #0b1018 100%);
            border: 1px solid #283143;
            border-top: 2px solid #e10600;
            margin-bottom: 10px;
        }
        .circuit-title {
            color: #ffffff;
            font-size: 1.08rem;
            font-weight: 700;
            margin-bottom: 4px;
        }
        .circuit-subtitle {
            color: #a8b6c8;
            font-size: 0.92rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    def build_layout_points(seed: int, n_points: int = 360) -> pd.DataFrame:
        """Create a deterministic circuit-like closed curve for visual layout preview."""
        rng = np.random.default_rng(seed)
        t = np.linspace(0, 2 * np.pi, n_points)

        a1, a2, a3 = rng.uniform(0, 2 * np.pi, 3)
        s1, s2, s3 = rng.uniform(0.08, 0.28, 3)

        radius = (
            1.0
            + s1 * np.sin(2 * t + a1)
            + s2 * np.cos(3 * t + a2)
            + s3 * np.sin(5 * t + a3)
        )

        x = radius * np.cos(t)
        y = 0.75 * radius * np.sin(t)

        # Apply deterministic rotation so each circuit feels different.
        theta = rng.uniform(0, 2 * np.pi)
        x_rot = x * np.cos(theta) - y * np.sin(theta)
        y_rot = x * np.sin(theta) + y * np.cos(theta)

        x_smooth = pd.Series(x_rot).rolling(7, center=True, min_periods=1).mean()
        y_smooth = pd.Series(y_rot).rolling(7, center=True, min_periods=1).mean()

        return pd.DataFrame({"x": x_smooth, "y": y_smooth, "idx": np.arange(n_points)})

    st.markdown(
        (
            "<div class='circuit-card'>"
            f"<div class='circuit-title'>{circuit_row['name']}</div>"
            f"<div class='circuit-subtitle'>{circuit_row['location']}, {circuit_row['country']}</div>"
            "</div>"
        ),
        unsafe_allow_html=True,
    )

    map_df = pd.DataFrame(
        {
            "lat": [float(circuit_row["lat"])],
            "lng": [float(circuit_row["lng"])],
            "label": [circuit_row["name"]],
        }
    )

    layout_df = build_layout_points(int(circuit_row["circuitId"]))
    layout_image_url = extract_circuit_layout_image(circuit_row.get("url", ""))

    st.caption("Circuit Explorer")
    c1, c2 = st.columns([1.2, 1])

    with c1:
        st.markdown("Track Location")

        deck = pdk.Deck(
            map_style="dark",
            initial_view_state=pdk.ViewState(
                latitude=float(circuit_row["lat"]),
                longitude=float(circuit_row["lng"]),
                zoom=12,
                pitch=45,
                bearing=25,
            ),
            layers=[
                pdk.Layer(
                    "ScatterplotLayer",
                    data=map_df,
                    get_position="[lng, lat]",
                    get_radius=650,
                    get_fill_color="[255, 70, 70, 190]",
                    pickable=True,
                ),
                pdk.Layer(
                    "TextLayer",
                    data=map_df,
                    get_position="[lng, lat]",
                    get_text="label",
                    get_color="[230, 236, 245]",
                    get_size=16,
                    get_alignment_baseline="bottom",
                    get_pixel_offset="[0, -18]",
                ),
            ],
            tooltip={"text": "{label}"},
        )

        st.pydeck_chart(deck)

    with c2:
        st.markdown("Circuit Layout")
        if layout_image_url:
            st.image(layout_image_url, caption="Circuit layout", use_container_width=True)
        else:
            st.vega_lite_chart(
                layout_df,
                {
                    "width": "container",
                    "height": 360,
                    "background": "#0d131b",
                    "layer": [
                        {
                            "mark": {"type": "line", "strokeWidth": 7, "color": "#f84464"},
                            "encoding": {
                                "x": {"field": "x", "type": "quantitative", "axis": None},
                                "y": {"field": "y", "type": "quantitative", "axis": None},
                                "order": {"field": "idx", "type": "quantitative"},
                            },
                        },
                        {
                            "transform": [{"filter": "datum.idx == 0"}],
                            "mark": {"type": "point", "filled": True, "size": 180, "color": "#ffffff"},
                            "encoding": {
                                "x": {"field": "x", "type": "quantitative"},
                                "y": {"field": "y", "type": "quantitative"},
                            },
                        },
                    ],
                },
                width="stretch",
            )
            st.caption("Layout image unavailable for this track, showing generated outline")

    m1, m2, m3 = st.columns(3)
    with m1:
        st.metric("Country", str(circuit_row["country"]))
    with m2:
        altitude = circuit_row.get("alt", np.nan)
        alt_text = "N/A" if pd.isna(altitude) else f"{int(float(altitude))} m"
        st.metric("Altitude", alt_text)
    with m3:
        st.metric("Circuit ID", str(int(circuit_row["circuitId"])))

# -------------------------
# LOAD DATA
# -------------------------
@st.cache_data
def load_data():
    base = Path("archive (2)")
    results = pd.read_csv(base / "results.csv", na_values=["\\N"])
    races = pd.read_csv(base / "races.csv", na_values=["\\N"])
    qualifying = pd.read_csv(base / "qualifying.csv", na_values=["\\N"])
    drivers = pd.read_csv(base / "drivers.csv", na_values=["\\N"])
    circuits = pd.read_csv(base / "circuits.csv", na_values=["\\N"])
    return results, races, qualifying, drivers, circuits


# -------------------------
# LOAD CIRCUITS
# -------------------------
@st.cache_data
def load_circuits():
    _, _, _, _, circuits = load_data()

    circuits["display_name"] = (
        circuits["name"] + " (" + circuits["location"] + ", " + circuits["country"] + ")"
    )

    circuits["lat"] = pd.to_numeric(circuits["lat"], errors="coerce")
    circuits["lng"] = pd.to_numeric(circuits["lng"], errors="coerce")
    circuits["alt"] = pd.to_numeric(circuits["alt"], errors="coerce")

    circuits = circuits.dropna(subset=["lat", "lng"]).copy()

    circuit_map = dict(zip(circuits["display_name"], circuits["circuitId"].astype(str)))

    return circuit_map, circuits


# -------------------------
# BUILD DRIVER STATS
# -------------------------
@st.cache_data
def build_driver_stats():
    results, races, qualifying, drivers, _ = load_data()

    df = results.merge(races, on="raceId")
    df = df.merge(
        qualifying[["raceId", "driverId", "position"]],
        on=["raceId", "driverId"],
        how="left"
    )
    df = df.merge(drivers, on="driverId")

    df.rename(columns={"position": "quali_position"}, inplace=True)

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["dob"] = pd.to_datetime(df["dob"], errors="coerce")

    df = df.sort_values("date")

    df["win"] = (df["positionOrder"] == 1).astype(int)

    driver_group = df.groupby("driverId")
    constructor_group = df.groupby("constructorId")

    df["driver_prior_races"] = driver_group.cumcount()
    df["driver_prev_wins"] = driver_group["win"].cumsum() - df["win"]
    df["driver_prev_points"] = driver_group["points"].cumsum() - df["points"]

    df["driver_prev_avg_finish"] = (
        driver_group["positionOrder"].cumsum() - df["positionOrder"]
    ) / df["driver_prior_races"].replace(0, np.nan)

    df["constructor_prior_races"] = constructor_group.cumcount()
    df["constructor_prev_wins"] = constructor_group["win"].cumsum() - df["win"]
    df["constructor_prev_points"] = constructor_group["points"].cumsum() - df["points"]

    df["constructor_prev_avg_finish"] = (
        constructor_group["positionOrder"].cumsum() - df["positionOrder"]
    ) / df["constructor_prior_races"].replace(0, np.nan)

    df["driver_age"] = (df["date"] - df["dob"]).dt.days / 365.25

    latest = df.sort_values("date").groupby("driverId").tail(1)
    latest = latest.fillna(0)

    return latest


# -------------------------
# DRIVER SELECTION
# -------------------------
stats_df = build_driver_stats()

driver_names = stats_df["forename"] + " " + stats_df["surname"]
driver_map = dict(zip(driver_names, stats_df["driverId"]))

selected_drivers = st.multiselect(
    "Select Drivers",
    options=list(driver_map.keys()),
    default=list(driver_map.keys())[:10]
)

k1, k2 = st.columns(2)
with k1:
    st.metric("Selected Drivers", len(selected_drivers))
with k2:
    st.metric("Grid Slots", "20")

# -------------------------
# GRID ASSIGNMENT (UNIQUE)
# -------------------------
section_header("Grid & Qualifying Setup", "Edit values directly in the race table for rapid scenario testing")

grid_inputs = {}
quali_inputs = {}
has_invalid_grid = False

if selected_drivers:
    setup_rows = []

    for idx, driver in enumerate(selected_drivers, start=1):
        default_position = min(idx, 20)
        saved_grid = st.session_state["grid_state"].get(driver, default_position)
        saved_quali = st.session_state["quali_state"].get(driver, saved_grid)

        setup_rows.append({
            "Driver": driver,
            "Grid": int(min(max(saved_grid, 1), 20)),
            "Quali": int(min(max(saved_quali, 1), 20)),
        })

    setup_df = pd.DataFrame(setup_rows)

    edited_setup = st.data_editor(
        setup_df,
        width="stretch",
        hide_index=True,
        key="race_setup_editor",
        column_config={
            "Driver": st.column_config.TextColumn("Driver", disabled=True),
            "Grid": st.column_config.NumberColumn("Grid", min_value=1, max_value=20, step=1),
            "Quali": st.column_config.NumberColumn("Quali", min_value=1, max_value=20, step=1),
        },
    )

    edited_setup["Grid"] = edited_setup["Grid"].astype(int)
    edited_setup["Quali"] = edited_setup["Quali"].astype(int)

    st.session_state["grid_state"].update(
        dict(zip(edited_setup["Driver"], edited_setup["Grid"]))
    )
    st.session_state["quali_state"].update(
        dict(zip(edited_setup["Driver"], edited_setup["Quali"]))
    )

    grid_inputs = dict(zip(edited_setup["Driver"], edited_setup["Grid"]))
    quali_inputs = dict(zip(edited_setup["Driver"], edited_setup["Quali"]))

    grid_counts = edited_setup["Grid"].value_counts()
    duplicate_grids = grid_counts[grid_counts > 1].index.tolist()
    has_invalid_grid = len(duplicate_grids) > 0

    if has_invalid_grid:
        duplicates_text = ", ".join(str(v) for v in sorted(duplicate_grids))
        st.error(f"Grid positions must be unique. Duplicate slots: {duplicates_text}")

    st.caption("Starting Grid Preview")
    st.dataframe(
        edited_setup.sort_values("Grid").reset_index(drop=True),
        width="stretch",
    )

    st.caption("Starting Grid Visual")
    render_starting_grid_visual(edited_setup)
else:
    st.info("Select at least one driver to configure grid and qualifying positions.")
    has_invalid_grid = True

# -------------------------
# CIRCUIT SELECTION
# -------------------------
section_header("Circuit Command Center", "Explore track geography and layout before simulation")

circuit_map, circuits_df = load_circuits()

selected_circuit = st.selectbox(
    "Choose Track",
    options=list(circuit_map.keys())
)

selected_circuit_row = circuits_df[circuits_df["display_name"] == selected_circuit].iloc[0]
circuit_id = str(int(selected_circuit_row["circuitId"]))

render_watch_section(str(selected_circuit_row["country"]))
render_circuit_banner()

render_circuit_experience(selected_circuit_row)

# -------------------------
# PREDICTION
# -------------------------
if st.button("Simulate Race 🏁", disabled=has_invalid_grid):

    rows = []

    for driver in selected_drivers:
        d_id = driver_map[driver]
        row = stats_df[stats_df["driverId"] == d_id].iloc[0]

        rows.append({
            "year": 2024,
            "round": 1,
            "grid": grid_inputs[driver],
            "quali_position": quali_inputs[driver],
            "driver_age": row["driver_age"],
            "driver_prior_races": row["driver_prior_races"],
            "driver_prev_wins": row["driver_prev_wins"],
            "driver_prev_points": row["driver_prev_points"],
            "driver_prev_avg_finish": row["driver_prev_avg_finish"],
            "constructor_prior_races": row["constructor_prior_races"],
            "constructor_prev_wins": row["constructor_prev_wins"],
            "constructor_prev_points": row["constructor_prev_points"],
            "constructor_prev_avg_finish": row["constructor_prev_avg_finish"],
            "driverId": str(int(row["driverId"])),
            "constructorId": str(int(row["constructorId"])),
            "circuitId": str(circuit_id)
        })

    input_df = pd.DataFrame(rows)

    # Raw probabilities
    probs = model.predict_proba(input_df)[:, 1]

    # Normalize for UI
    probs = probs / probs.sum()

    input_df["Driver"] = selected_drivers
    input_df["Win Probability (%)"] = (probs * 100).round(2)

    result = input_df.sort_values("Win Probability (%)", ascending=False).reset_index(drop=True)

    section_header("Race Prediction Leaderboard", "Projected finish strength based on current race setup")

    # Table
    st.dataframe(
        result[["Driver", "grid", "Win Probability (%)"]],
        width="stretch"
    )

    # Leaderboard
    section_header("Podium View", "Top ranked outcomes from the current simulation")

    for i, row in result.iterrows():
        st.metric(
            label=f"#{i+1} {row['Driver']}",
            value=f"{row['Win Probability (%)']}%"
        )

    winner = result.iloc[0]

    st.success(
        f"🏁 Predicted Winner: {winner['Driver']} ({winner['Win Probability (%)']}%)"
    )