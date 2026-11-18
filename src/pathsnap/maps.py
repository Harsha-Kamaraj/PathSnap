"""Folium maps of one trip: raw GPS, road network, reference route, classifier-only and HMM routes."""
from __future__ import annotations

import folium
import numpy as np
import pandas as pd
import shapely

LAYERS = {  # name: (colour, line weight)
    "Road network (arcs near the track)": ("#9a9a9a", 1.5),
    "Reference route": ("#1b9e77", 7),
    "Classifier only (arc of each fix)": ("#d95f02", 4),
    "HMM / Viterbi route": ("#3355cc", 3),
}
GPS_COLOUR = "#111111"


def _arc_lines(arcs: list[int], nodes: pd.DataFrame, arcs_df: pd.DataFrame) -> list[list[tuple]]:
    fr, to = arcs_df["from_node"].to_numpy(), arcs_df["to_node"].to_numpy()
    lat, lon = nodes["lat"].to_numpy(), nodes["lon"].to_numpy()
    return [[(lat[fr[a]], lon[fr[a]]), (lat[to[a]], lon[to[a]])] for a in arcs]


def _legend(title: str) -> folium.Element:
    items = "".join(f'<div><span style="display:inline-block;width:22px;height:{max(3, w)}px;background:{c};'
                    f'vertical-align:middle;margin-right:6px"></span>{n}</div>' for n, (c, w) in LAYERS.items())
    items += (f'<div><span style="display:inline-block;width:8px;height:8px;border-radius:50%;background:{GPS_COLOUR};'
              f'margin:0 13px 0 7px"></span>Raw GPS fixes</div>')
    html = (f'<div style="position:fixed;bottom:24px;left:24px;z-index:9999;background:white;padding:8px 10px;'
            f'border:1px solid #888;border-radius:4px;font:12px sans-serif"><b>{title}</b>{items}</div>')
    return folium.Element(html)


def trip_map(title: str, track: pd.DataFrame, nodes: pd.DataFrame, arcs_df: pd.DataFrame, ref_arcs: list[int],
             clf_arcs: list[int], hmm_pieces: list[list[int]], network_radius_m: float = 100.0,
             max_gps_points: int = 1500) -> folium.Map:
    """`track` needs lon, lat, x, y; network = arcs within `network_radius_m` of any GPS fix."""
    m = folium.Map(tiles="OpenStreetMap", control_scale=True)
    nxy = nodes[["x", "y"]].to_numpy()
    fr, to = arcs_df["from_node"].to_numpy(), arcs_df["to_node"].to_numpy()
    canon = arcs_df["canonical_arc_id"].to_numpy()
    tree = shapely.STRtree(shapely.linestrings(np.stack([nxy[fr], nxy[to]], axis=1)))
    near = np.unique(tree.query(shapely.points(track[["x", "y"]].to_numpy()), predicate="dwithin",
                                distance=network_radius_m)[1])
    near = np.unique(canon[near])

    def add_lines(name, arcs):
        c, w = LAYERS[name]
        g = folium.FeatureGroup(name=name)
        for ln in _arc_lines(list(arcs), nodes, arcs_df):
            folium.PolyLine(ln, color=c, weight=w, opacity=0.85).add_to(g)
        g.add_to(m)

    add_lines("Road network (arcs near the track)", near)
    add_lines("Reference route", sorted(set(ref_arcs)))
    add_lines("Classifier only (arc of each fix)", sorted(set(clf_arcs)))
    c, w = LAYERS["HMM / Viterbi route"]
    g = folium.FeatureGroup(name="HMM / Viterbi route")
    lat, lon = nodes["lat"].to_numpy(), nodes["lon"].to_numpy()
    for pc in hmm_pieces:
        pts = [(lat[fr[pc[0]]], lon[fr[pc[0]]])] + [(lat[to[a]], lon[to[a]]) for a in pc]
        folium.PolyLine(pts, color=c, weight=w, opacity=0.95).add_to(g)
    g.add_to(m)

    g = folium.FeatureGroup(name="Raw GPS fixes")
    step = max(1, len(track) // max_gps_points)
    for r in track.iloc[::step].itertuples():
        folium.CircleMarker((r.lat, r.lon), radius=2, color=GPS_COLOUR, fill=True, fill_opacity=0.8,
                            weight=0).add_to(g)
    g.add_to(m)
    m.fit_bounds([[track.lat.min(), track.lon.min()], [track.lat.max(), track.lon.max()]])
    folium.LayerControl(collapsed=False).add_to(m)
    m.get_root().html.add_child(_legend(title))
    return m
