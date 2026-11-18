"""PathSnap evaluator UI (runs on laptop 2).

The UI only talks to the API over HTTP (docs/API_CONTRACT.md). It never loads the model,
the dataset or any ground truth: it shows the trips the API offers, sends the chosen
trip_id to the API, and draws what comes back.

Layout: sidebar = API connection + the trips + Run button; main page = scores on top, map below.

Run (laptop 2):
    streamlit run ui/app.py
Point it at laptop 1 with the sidebar box, or start it with
    PATHSNAP_API_URL=http://<laptop-1-ip>:8000 streamlit run ui/app.py
"""
from __future__ import annotations

import json
import os
import time

import folium
import requests
import streamlit as st

DEFAULT_API = os.environ.get("PATHSNAP_API_URL", "http://localhost:8000")
HEALTH_TIMEOUT_S = 3
PREDICT_TIMEOUT_S = 300          # big trips + HMM can take a while
MAX_GPS_POINTS = 3000            # GPS dots drawn on the map (the full track is drawn as a thin line)
MAP_HEIGHT = 600

GPS_COLOR = "#2563eb"
ROUTE_COLOR = "#dc2626"

st.set_page_config(page_title="PathSnap", page_icon="🗺️", layout="wide")

st.markdown("""
<style>
  .block-container {padding-top: 1.4rem; padding-bottom: 1rem; max-width: 1400px;}
  section[data-testid="stSidebar"] {background: #f3f7f4; border-right: 1px solid #e3ebe5;}
  section[data-testid="stSidebar"] .block-container {padding-top: 1.2rem;}
  .stButton > button {border-radius: 999px; font-weight: 600; padding: .55rem 1.2rem;
                      transition: transform .05s ease, box-shadow .15s ease;}
  .stButton > button[kind="primary"] {background: linear-gradient(135deg, #22c55e, #15803d); border: 0;
                                      box-shadow: 0 4px 14px rgba(22, 163, 74, .30);}
  .stButton > button[kind="primary"]:hover {box-shadow: 0 6px 18px rgba(22, 163, 74, .40);}
  .stButton > button:active {transform: translateY(1px);}
  .stTextInput input {border-radius: 999px;}
  div[data-testid="stMetric"] {background: #ffffff; border: 1px solid #e6ece8; border-radius: 18px;
                               padding: 16px 18px; box-shadow: 0 2px 10px rgba(15, 23, 42, .05);}
  div[data-testid="stMetricLabel"] p {font-size: .85rem; color: #64748b; font-weight: 600;}
  div[data-testid="stMetricValue"] {font-size: 2.1rem; font-weight: 700; color: #0f172a;}
  .ps-brand {font-size: 1.5rem; font-weight: 800; color: #14532d; margin-bottom: .1rem;}
  .ps-sub {color: #64748b; font-size: .85rem; margin-bottom: .8rem;}
  .ps-pill {display: inline-flex; align-items: center; gap: 8px; padding: 6px 14px; border-radius: 999px;
            font-weight: 600; font-size: .9rem;}
  .ps-pill.ok {background: #dcfce7; color: #166534;}
  .ps-pill.bad {background: #fee2e2; color: #991b1b;}
  .ps-dot {width: 9px; height: 9px; border-radius: 50%; display: inline-block;}
  .ps-model {color: #475569; font-size: .82rem; margin: 8px 0 10px;}
  .ps-head {display: flex; align-items: baseline; gap: 14px; flex-wrap: wrap; margin-bottom: 6px;}
  .ps-head h2 {margin: 0; font-size: 1.6rem; color: #0f172a;}
  .ps-tag {padding: 3px 12px; border-radius: 999px; background: #f0fdf4; border: 1px solid #bbf7d0;
           color: #166534; font-size: .8rem; font-weight: 600;}
  .ps-empty {border: 2px dashed #d1e7d8; border-radius: 22px; padding: 48px 24px; text-align: center;
             color: #475569; background: #fbfdfb;}
  .ps-empty h2 {color: #14532d; margin-bottom: .3rem;}
  iframe {border-radius: 18px; box-shadow: 0 2px 14px rgba(15, 23, 42, .08);}
</style>
""", unsafe_allow_html=True)


# ----------------------------------------------------------------------------
# API calls (every request goes to laptop 1; nothing is computed here).
# Health and trip list are cached briefly so clicking around does not re-ask the API.
# ----------------------------------------------------------------------------
@st.cache_resource
def http() -> requests.Session:
    return requests.Session()          # keeps the connection to laptop 1 open between calls


@st.cache_data(ttl=15, show_spinner=False)
def get_health(base: str):
    try:
        r = http().get(f"{base}/health", timeout=HEALTH_TIMEOUT_S)
        r.raise_for_status()
        return r.json()
    except requests.RequestException:
        return None


@st.cache_data(ttl=300, show_spinner=False)
def get_trips(base: str) -> list[dict]:
    r = http().get(f"{base}/trips", timeout=30)
    r.raise_for_status()
    return r.json()["trips"]


def api_predict(base: str, trip_id: str) -> dict:
    r = http().post(f"{base}/predict", json={"trip_id": trip_id}, timeout=PREDICT_TIMEOUT_S)
    if r.status_code != 200:
        try:
            detail = r.json().get("detail", r.text)
        except ValueError:
            detail = r.text
        raise RuntimeError(f"API error {r.status_code}: {detail}")
    return r.json()


def fmt_pct(v) -> str:
    return "–" if v is None else f"{v:.1%}"


def fmt_duration(s) -> str:
    if s is None:
        return "–"
    m = int(round(s / 60))
    return f"{m} min" if m < 120 else f"{m / 60:.1f} h"


# ----------------------------------------------------------------------------
# Map: built once per result and kept as HTML (fast to redraw on every rerun)
# ----------------------------------------------------------------------------
LEGEND_HTML = f"""
<div style="position: fixed; bottom: 22px; left: 22px; z-index: 9999; background: rgba(255,255,255,.95);
            padding: 10px 14px; border-radius: 14px; box-shadow: 0 2px 10px rgba(0,0,0,.15);
            font: 13px/1.7 -apple-system, Segoe UI, sans-serif; color: #1f2937;">
  <b>Legend</b><br>
  <span style="display:inline-block;width:10px;height:10px;border-radius:50%;
               background:{GPS_COLOR};margin-right:6px;"></span>Raw GPS points<br>
  <span style="display:inline-block;width:18px;height:4px;border-radius:2px;background:{ROUTE_COLOR};
               margin:0 6px 3px 0;vertical-align:middle;"></span>Predicted route<br>
  <span style="margin-right:4px;">🟢</span>Start &nbsp; <span style="margin-right:4px;">🏁</span>End
</div>
"""


def build_map_html(res: dict) -> str:
    gps = res["gps"]                                   # [lon, lat] (GeoJSON order)
    m = folium.Map(tiles=None, control_scale=True)
    folium.TileLayer("OpenStreetMap", name="Street map").add_to(m)

    step = max(1, len(gps) // MAX_GPS_POINTS)
    g_gps = folium.FeatureGroup(name="Raw GPS points", show=True)
    folium.PolyLine([(lat, lon) for lon, lat in gps], color=GPS_COLOR, weight=1, opacity=0.35).add_to(g_gps)
    # all dots in ONE GeoJSON layer (much faster than one marker object per point)
    folium.GeoJson(
        {"type": "Feature", "properties": {}, "geometry": {"type": "MultiPoint", "coordinates": gps[::step]}},
        marker=folium.CircleMarker(radius=2.5, color=GPS_COLOR, fill=True, fill_color=GPS_COLOR,
                                   fill_opacity=0.85, weight=0),
    ).add_to(g_gps)
    g_gps.add_to(m)

    lines = [ln for ln in res["predicted_route"] if len(ln) >= 2]
    if lines:
        folium.GeoJson(
            {"type": "Feature", "properties": {}, "geometry": {"type": "MultiLineString", "coordinates": lines}},
            name="Predicted route",
            style_function=lambda _: {"color": ROUTE_COLOR, "weight": 4.5, "opacity": 0.85},
        ).add_to(m)

    g_ends = folium.FeatureGroup(name="Start / end", show=True)
    folium.Marker((gps[0][1], gps[0][0]), tooltip="Start", icon=folium.Icon(color="green", icon="play")).add_to(g_ends)
    folium.Marker((gps[-1][1], gps[-1][0]), tooltip="End", icon=folium.Icon(color="black", icon="flag")).add_to(g_ends)
    g_ends.add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)
    m.get_root().html.add_child(folium.Element(LEGEND_HTML))
    lons, lats = [p[0] for p in gps], [p[1] for p in gps]
    m.fit_bounds([[min(lats), min(lons)], [max(lats), max(lons)]], padding=(20, 20))
    m.get_root().width, m.get_root().height = "100%", f"{MAP_HEIGHT}px"
    return m.get_root().render()


# ----------------------------------------------------------------------------
# Sidebar: API connection + trips + run
# ----------------------------------------------------------------------------
with st.sidebar:
    st.markdown('<div class="ps-brand">🗺️ PathSnap</div>'
                '<div class="ps-sub">Noisy GPS in, the real road route out.</div>', unsafe_allow_html=True)

    st.markdown("**API connection**")
    api = st.text_input("API address (laptop 1)", value=DEFAULT_API, label_visibility="collapsed",
                        help="e.g. http://192.168.137.1:8000").rstrip("/")
    health = get_health(api)
    if health:
        st.markdown('<span class="ps-pill ok"><span class="ps-dot" style="background:#16a34a"></span>Connected</span>'
                    f'<div class="ps-model">{health.get("classifier", "–")} + {health.get("decoder", "–")}</div>',
                    unsafe_allow_html=True)
    else:
        st.markdown('<span class="ps-pill bad"><span class="ps-dot" style="background:#dc2626"></span>'
                    'Not connected</span><div class="ps-model">Is the API running on laptop 1, on the same '
                    'network, with port 8000 open?</div>', unsafe_allow_html=True)
    if st.button("↻ Reconnect", width="stretch"):
        get_health.clear()
        get_trips.clear()
        st.rerun()

    trips, chosen, run = [], None, False
    if health:
        try:
            trips = get_trips(api)
        except requests.RequestException as e:
            st.error(f"Could not load the trips: {e}")
    if trips:
        st.markdown(f"**Trips ({len(trips)})**")
        labels = {t["trip_id"]: f"{t['trip_id']}  ·  {t.get('n_points', 0):,} pts  ·  {fmt_duration(t.get('duration_s'))}"
                  for t in trips}
        chosen = st.radio("Trip", list(labels), format_func=labels.get, label_visibility="collapsed",
                          key="trip_choice")
        run = st.button("▶  Run prediction", type="primary", width="stretch")

# ----------------------------------------------------------------------------
# Run the prediction (API on laptop 1)
# ----------------------------------------------------------------------------
if run and chosen:
    with st.spinner(f"Sending trip {chosen} to the API…"):
        t0 = time.perf_counter()
        try:
            res = api_predict(api, chosen)
            res["_roundtrip_ms"] = (time.perf_counter() - t0) * 1000
            st.session_state["result"] = res
            st.session_state["map_html"] = build_map_html(res)
            st.session_state.pop("error", None)
        except (requests.RequestException, RuntimeError) as e:
            st.session_state["error"] = str(e)

# ----------------------------------------------------------------------------
# Main page: results on top, map below
# ----------------------------------------------------------------------------
if health and health.get("mock"):
    st.warning("**MOCK API**: test data from `ui/mock_api.py` (nearest-road baseline, no model). "
               "Point the UI at the real API for the evaluation.", icon="⚠️")
if "error" in st.session_state:
    st.error(st.session_state["error"])

res = st.session_state.get("result")
if res is None:
    st.markdown('<div class="ps-empty"><h2>Pick a trip to begin</h2>'
                '<p>Choose one of the trips in the sidebar and press <b>Run prediction</b>.<br>'
                'The API on laptop 1 reconstructs the route and scores it; the result appears here.</p></div>',
                unsafe_allow_html=True)
    st.stop()

if "map_html" not in st.session_state:          # result set from outside (e.g. tests): build the map once
    st.session_state["map_html"] = build_map_html(res)

met = res.get("metrics", {})
top1, base = met.get("top1_accuracy"), met.get("top1_accuracy_classifier_only")
st.markdown(f'<div class="ps-head"><h2>Trip {res["trip_id"]}</h2>'
            f'<span class="ps-tag">{res.get("classifier", "–")} + {res.get("decoder", "–")}</span></div>',
            unsafe_allow_html=True)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Top-1 accuracy", fmt_pct(top1),
          delta=None if top1 is None or base is None else f"{(top1 - base) * 100:+.1f} pp vs classifier",
          help="Share of scored GPS points matched to the correct road segment (computed by the API).")
c2.metric("Classifier only", fmt_pct(base), help="Top-1 accuracy before HMM/Viterbi decoding.")
c3.metric("Route continuity", fmt_pct(met.get("route_continuity")),
          help="Share of consecutive points whose predicted segments are the same or connected.")
c4.metric("GPS points", f"{res['n_points']:,}",
          help=f"{met.get('n_scored_points', '–')} of them have a reference label and are scored.")
st.caption(f"API time {res.get('timing_ms', {}).get('total', 0):,.0f} ms · "
           f"round trip {res.get('_roundtrip_ms', 0):,.0f} ms · {len(res['predicted_route'])} route piece(s)")

st.iframe(st.session_state["map_html"], height=MAP_HEIGHT + 10)   # HTML built by our own code (numbers from our API)

with st.expander("Response details"):
    summary = {k: v for k, v in res.items() if k not in ("gps", "predicted_route")}
    summary["predicted_route"] = f"{len(res['predicted_route'])} polyline(s)"
    st.json(summary)
    st.download_button("Download full response (JSON)", json.dumps(res),
                       file_name=f"pathsnap_{res['trip_id']}.json", mime="application/json")
