"""PathSnap evaluator UI (runs on laptop 2).

The UI only talks to the API over HTTP (docs/API_CONTRACT.md). It never loads the model,
the dataset or any ground truth: it shows the trips the API offers, sends the chosen
trip_id to the API, and draws what comes back.

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
import pandas as pd
import requests
import streamlit as st
from streamlit_folium import st_folium

DEFAULT_API = os.environ.get("PATHSNAP_API_URL", "http://localhost:8000")
HEALTH_TIMEOUT_S = 3
PREDICT_TIMEOUT_S = 300          # big trips + HMM can take a while
MAX_GPS_MARKERS = 2000           # draw at most this many GPS dots (the full track is also drawn as a line)

GPS_COLOR = "#2a6fdb"
ROUTE_COLOR = "#d62728"

st.set_page_config(page_title="PathSnap", page_icon="🗺️", layout="wide")


# ----------------------------------------------------------------------------
# API calls (every request goes to laptop 1; nothing is computed here)
# ----------------------------------------------------------------------------
def api_get(base: str, path: str, timeout: float):
    r = requests.get(f"{base}{path}", timeout=timeout)
    r.raise_for_status()
    return r.json()


def api_predict(base: str, trip_id: str) -> dict:
    r = requests.post(f"{base}/predict", json={"trip_id": trip_id}, timeout=PREDICT_TIMEOUT_S)
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
# Map
# ----------------------------------------------------------------------------
LEGEND_HTML = f"""
<div style="position: fixed; bottom: 24px; left: 24px; z-index: 9999; background: white;
            padding: 8px 12px; border-radius: 6px; box-shadow: 0 1px 4px rgba(0,0,0,.3);
            font: 13px/1.6 sans-serif; color: #222;">
  <b>Legend</b><br>
  <span style="display:inline-block;width:10px;height:10px;border-radius:50%;
               background:{GPS_COLOR};margin-right:6px;"></span>Raw GPS points<br>
  <span style="display:inline-block;width:18px;height:4px;background:{ROUTE_COLOR};
               margin:0 6px 3px 0;vertical-align:middle;"></span>Predicted route<br>
  <span style="margin-right:4px;">🟢</span>Start &nbsp; <span style="margin-right:4px;">🏁</span>End
</div>
"""


def build_map(res: dict) -> folium.Map:
    gps = [(lat, lon) for lon, lat in res["gps"]]               # API sends [lon, lat]; folium wants (lat, lon)
    m = folium.Map(tiles=None, control_scale=True)
    folium.TileLayer("OpenStreetMap", name="Street map").add_to(m)

    g_gps = folium.FeatureGroup(name="Raw GPS points", show=True)
    folium.PolyLine(gps, color=GPS_COLOR, weight=1, opacity=0.4).add_to(g_gps)
    step = max(1, len(gps) // MAX_GPS_MARKERS)
    for i in range(0, len(gps), step):
        folium.CircleMarker(gps[i], radius=2, color=GPS_COLOR, fill=True, fill_opacity=0.8,
                            weight=0, tooltip=f"GPS point {i}").add_to(g_gps)
    g_gps.add_to(m)

    g_route = folium.FeatureGroup(name="Predicted route", show=True)
    for line in res["predicted_route"]:
        if len(line) >= 2:
            folium.PolyLine([(lat, lon) for lon, lat in line], color=ROUTE_COLOR, weight=4,
                            opacity=0.85).add_to(g_route)
    g_route.add_to(m)

    g_ends = folium.FeatureGroup(name="Start / end", show=True)
    folium.Marker(gps[0], tooltip="Start", icon=folium.Icon(color="green", icon="play")).add_to(g_ends)
    folium.Marker(gps[-1], tooltip="End", icon=folium.Icon(color="black", icon="flag")).add_to(g_ends)
    g_ends.add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)
    m.get_root().html.add_child(folium.Element(LEGEND_HTML))
    lats, lons = [p[0] for p in gps], [p[1] for p in gps]
    m.fit_bounds([[min(lats), min(lons)], [max(lats), max(lons)]], padding=(20, 20))
    return m


# ----------------------------------------------------------------------------
# Sidebar: connection to laptop 1
# ----------------------------------------------------------------------------
with st.sidebar:
    st.header("API connection")
    api = st.text_input("API address (laptop 1)", value=DEFAULT_API,
                        help="e.g. http://192.168.1.20:8000").rstrip("/")
    try:
        health = api_get(api, "/health", HEALTH_TIMEOUT_S)
        st.success("Connected", icon="🟢")
        st.caption(f"Classifier: **{health.get('classifier', '–')}**  \nDecoder: **{health.get('decoder', '–')}**")
    except requests.RequestException as e:
        health = None
        st.error("Not connected", icon="🔴")
        st.caption(f"`{type(e).__name__}`: check that the API is running on laptop 1, "
                   "both laptops are on the same network, and port 8000 is open.")
    if st.button("Reconnect / refresh trips", width="stretch"):
        st.session_state.pop("trips", None)
        st.rerun()

# ----------------------------------------------------------------------------
# Main page
# ----------------------------------------------------------------------------
st.title("🗺️ PathSnap: GPS map-matching")
st.caption("Pick a trip, send it to the API, and see the route the model reconstructs from noisy GPS.")

if health is None:
    st.info("Waiting for the API. Start it on laptop 1, then press **Reconnect** in the sidebar.")
    st.stop()
if health.get("mock"):
    st.warning("**MOCK API**: this is test data from `ui/mock_api.py` (nearest-road baseline, "
               "no model). Point the UI at the real API for the evaluation.", icon="⚠️")

if st.session_state.get("trips_api") != api:          # address changed -> reload the list
    st.session_state.pop("trips", None)
if "trips" not in st.session_state:
    try:
        st.session_state["trips"] = api_get(api, "/trips", 30)["trips"]
        st.session_state["trips_api"] = api
    except requests.RequestException as e:
        st.error(f"Could not load the trip list: {e}")
        st.stop()
trips = st.session_state["trips"]

left, right = st.columns([1, 2.6], gap="large")

with left:
    st.subheader(f"Trips ({len(trips)})")
    table = pd.DataFrame({
        "Trip": [t["trip_id"] for t in trips],
        "GPS points": [t.get("n_points") for t in trips],
        "Duration": [fmt_duration(t.get("duration_s")) for t in trips],
    })
    event = st.dataframe(table, hide_index=True, on_select="rerun", selection_mode="single-row",
                         key="trip_table", width="stretch")
    rows = event.selection.rows
    chosen = trips[rows[0]]["trip_id"] if rows else None
    st.markdown(f"Selected: **{chosen}**" if chosen else "Click a row to select a trip.")
    run = st.button("▶ Run prediction", type="primary", disabled=chosen is None, width="stretch")

    if run and chosen:
        with st.spinner(f"Sending trip {chosen} to the API…"):
            t0 = time.perf_counter()
            try:
                res = api_predict(api, chosen)
                res["_roundtrip_ms"] = (time.perf_counter() - t0) * 1000
                st.session_state["result"] = res
                st.session_state.pop("error", None)
            except (requests.RequestException, RuntimeError) as e:
                st.session_state["error"] = str(e)
                st.session_state.pop("result", None)

with right:
    if "error" in st.session_state:
        st.error(st.session_state["error"])
    res = st.session_state.get("result")
    if res is None:
        st.info("Select a trip on the left and press **Run prediction**.")
    else:
        met = res.get("metrics", {})
        st.subheader(f"Trip {res['trip_id']}")
        c1, c2, c3, c4 = st.columns(4)
        top1, base = met.get("top1_accuracy"), met.get("top1_accuracy_classifier_only")
        c1.metric("Top-1 accuracy", fmt_pct(top1),
                  delta=None if top1 is None or base is None else f"{(top1 - base) * 100:+.1f} pp vs classifier only",
                  help="Share of scored GPS points matched to the correct road segment (computed by the API).")
        c2.metric("Classifier only", fmt_pct(base), help="Top-1 accuracy before HMM/Viterbi decoding.")
        c3.metric("Route continuity", fmt_pct(met.get("route_continuity")),
                  help="Share of consecutive points whose predicted segments are the same or connected.")
        c4.metric("GPS points", f"{res['n_points']:,}",
                  help=f"{met.get('n_scored_points', '–')} of them have a reference label and are scored.")
        st.caption(f"Model: **{res.get('classifier', '–')}** + **{res.get('decoder', '–')}** · "
                   f"API time {res.get('timing_ms', {}).get('total', 0):,.0f} ms · "
                   f"round trip {res['_roundtrip_ms']:,.0f} ms")

        st_folium(build_map(res), key=f"map_{res['trip_id']}", height=560,
                  use_container_width=True, returned_objects=[])

        with st.expander("Response details"):
            summary = {k: v for k, v in res.items() if k not in ("gps", "predicted_route")}
            summary["predicted_route"] = f"{len(res['predicted_route'])} polyline(s)"
            st.json(summary)
            st.download_button("Download full response (JSON)", json.dumps(res),
                               file_name=f"pathsnap_{res['trip_id']}.json", mime="application/json")
