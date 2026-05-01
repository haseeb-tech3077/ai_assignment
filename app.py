import math
import heapq
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
import streamlit as st
from collections import deque


#  MODULE1: City road network data

# Node: { "name": string, "x": float, "y": float }
# x, y = normalized 2D coordinates (0–10 scale) which are used for heuristic calculation
NODES = {
    0:  {"name": "Saddar",               "x": 5.0, "y": 5.0},
    1:  {"name": "Committee Chowk",      "x": 4.2, "y": 5.8},
    2:  {"name": "Faizabad",             "x": 6.5, "y": 6.2},
    3:  {"name": "Murree Road",          "x": 5.8, "y": 4.0},
    4:  {"name": "Liaquat Bagh",         "x": 4.5, "y": 4.5},
    5:  {"name": "Peshawar Morr",        "x": 7.0, "y": 7.5},
    6:  {"name": "F-10 Markaz",          "x": 8.0, "y": 6.5},
    7:  {"name": "G-9 Karachi Company",  "x": 7.5, "y": 5.0},
    8:  {"name": "Blue Area",            "x": 8.8, "y": 5.8},
    9:  {"name": "Aabpara",              "x": 9.0, "y": 6.5},
    10: {"name": "Sector F-6",           "x": 9.5, "y": 7.0},
    11: {"name": "Raja Bazar",           "x": 3.8, "y": 5.2},
    12: {"name": "Dhoke Kala Khan",      "x": 3.0, "y": 4.0},
    13: {"name": "Pirwadhai",            "x": 3.5, "y": 6.8},
    14: {"name": "Motorway Interchange", "x": 2.5, "y": 7.5},
    15: {"name": "Bahria Town Gate",     "x": 1.5, "y": 6.5},
    16: {"name": "Westridge",            "x": 4.0, "y": 3.0},
    17: {"name": "Chaklala",             "x": 6.0, "y": 2.5},
    18: {"name": "Airport",              "x": 7.0, "y": 2.0},
    19: {"name": "Rawat",                "x": 7.5, "y": 1.0},
    20: {"name": "Koral Chowk",          "x": 8.5, "y": 3.0},
    21: {"name": "I-8 Markaz",           "x": 8.0, "y": 4.0},
    22: {"name": "H-8",                  "x": 8.5, "y": 4.5},
    23: {"name": "Tarnol",               "x": 2.0, "y": 8.5},
    24: {"name": "Gulzar-e-Quaid",       "x": 5.5, "y": 3.0},
}

# Edges: (first node i.e node_a, second node: node_b, distance: base_distance_km)
EDGES = [
    (0,  1,  1.2),
    (0,  3,  1.5),
    (0,  4,  0.8),
    (0,  11, 0.9),
    (1,  4,  0.7),
    (1,  11, 0.5),
    (1,  13, 1.8),
    (2,  5,  1.5),
    (2,  3,  2.0),
    (2,  7,  1.8),
    (3,  4,  1.0),
    (3,  17, 2.5),
    (3,  24, 1.2),
    (4,  12, 1.3),
    (4,  16, 1.5),
    (5,  6,  1.2),
    (5,  13, 2.2),
    (6,  9,  1.5),
    (6,  7,  1.8),
    (7,  8,  1.3),
    (7,  21, 1.0),
    (7,  22, 0.8),
    (8,  9,  0.9),
    (9,  10, 0.7),
    (11, 12, 1.4),
    (12, 16, 1.5),
    (13, 14, 2.0),
    (14, 15, 2.5),
    (14, 23, 1.8),
    (16, 17, 2.8),
    (17, 18, 1.5),
    (17, 24, 1.0),
    (18, 19, 2.0),
    (18, 20, 1.8),
    (19, 20, 1.5),
    (20, 21, 1.3),
    (21, 22, 0.6),
    (22, 8,  1.5),
    (24, 7,  2.0),
    (2,  0,  2.5),
]

# Traffic multipliers: increase edge weights to simulate congestion

TRAFFIC_MULTIPLIERS = {
    "Off-Peak (Night / Afternoon)": 1.0,
    "Morning Rush (7–10 AM)":       1.6,
    "Evening Rush (5–8 PM)":        1.8,
}

#  MODULE2: Graph building

def build_adjacency_list(traffic_mode: str) -> dict:    # Build a weighted adjacency list from EDGES, applying the traffic multiplier
    # Here traffic_mode is Key from TRAFFIC_MULTIPLIERS dict.
    multiplier = TRAFFIC_MULTIPLIERS.get(traffic_mode, 1.0)
    adj = {node_id: [] for node_id in NODES}

    for (a, b, dist) in EDGES:
        weighted = round(dist * multiplier, 3)
        adj[a].append((b, weighted))
        adj[b].append((a, weighted))   # undirected graph

    return adj

def get_node_names_sorted() -> list:        # Returns list of sorted tuples by name
    return sorted(NODES.items(), key=lambda x: x[1]["name"])

#  Heuristic function

def heuristic(node_id: int, goal_id: int) -> float:         # Calculates as h(n) = sqrt((x_n - x_goal)^2 + (y_n - y_goal)^2)
    n = NODES[node_id]
    g = NODES[goal_id]
    return math.sqrt((n["x"] - g["x"]) ** 2 + (n["y"] - g["y"]) ** 2)       # Euclidean distance


def astar_search(adj: dict, source: int, goal: int) -> dict:

    # Edge case: source equals goal
    if source == goal:
        return {"path": [source], "cost": 0.0, "nodes_explored": 1, "found": True}

    # Initialise open list with source node
    h_start = heuristic(source, goal)
    open_list = [(h_start, 0.0, source)]   # (f_score, g_score, node_id)

    came_from = {}                          # {node: parent_node}
    g_score   = {source: 0.0}              # best cost to reach each node
    closed    = set()                       # fully expanded nodes
    nodes_explored = 0

    # Main loop 
    while open_list:
        f, g, current = heapq.heappop(open_list)

        # Skip node already expanded with lower cost
        if current in closed:
            continue

        closed.add(current)
        nodes_explored += 1

        #  Goal reached → reconstruct path
        if current == goal:
            path = []
            node = goal
            while node != source:
                path.append(node)
                node = came_from[node]
            path.append(source)
            path.reverse()
            return {
                "path": path,
                "cost": round(g_score[goal], 3),
                "nodes_explored": nodes_explored,
                "found": True
            }

        # Expand neighbors
        for neighbor, edge_cost in adj.get(current, []):
            if neighbor in closed:
                continue

            tentative_g = g_score[current] + edge_cost

            # Only push if this path is better than any previously found path
            if tentative_g < g_score.get(neighbor, math.inf):
                came_from[neighbor] = current
                g_score[neighbor]   = tentative_g
                f_new = tentative_g + heuristic(neighbor, goal)
                heapq.heappush(open_list, (f_new, tentative_g, neighbor))

    # No path found 
    return {"path": [], "cost": float("inf"), "nodes_explored": nodes_explored, "found": False}

#  MODULE3: Route visualization

def draw_route(path: list, title: str = "City Road Network") -> plt.Figure:

    # Build NetworkX graph 
    G = nx.Graph()
    for node_id, info in NODES.items():
        G.add_node(node_id, label=info["name"])
    for (a, b, dist) in EDGES:
        G.add_edge(a, b, weight=dist)

    # Node positions from our coordinate map
    pos = {nid: (info["x"], info["y"]) for nid, info in NODES.items()}

    # Figure setup
    fig, ax = plt.subplots(figsize=(14, 9))
    fig.patch.set_facecolor("#1E1E2E")
    ax.set_facecolor("#1E1E2E")

    # Draw all edges (dim grey)
    nx.draw_networkx_edges(G, pos, edge_color="#444466", width=1.2, alpha=0.6, ax=ax)

    # Draw all nodes (light blue)
    nx.draw_networkx_nodes(G, pos, node_color="#5E81AC", node_size=280, alpha=0.9, ax=ax)

    # Highlight optimal path 
    if len(path) > 1:
        path_edges = [(path[i], path[i + 1]) for i in range(len(path) - 1)]
        nx.draw_networkx_edges(G, pos, edgelist=path_edges,
                               edge_color="#A3BE8C", width=4.5, alpha=1.0, ax=ax)

        # Intermediate route nodes (yellow)
        mid_nodes = path[1:-1]
        if mid_nodes:
            nx.draw_networkx_nodes(G, pos, nodelist=mid_nodes,
                                   node_color="#EBCB8B", node_size=350, ax=ax)

        # Source (red) and destination (green)
        nx.draw_networkx_nodes(G, pos, nodelist=[path[0]],
                               node_color="#BF616A", node_size=520, ax=ax) # Source
        nx.draw_networkx_nodes(G, pos, nodelist=[path[-1]],
                               node_color="#A3BE8C", node_size=520, ax=ax) # Destination

    # Node labels (wrap long names)
    labels = {nid: info["name"].replace(" ", "\n") for nid, info in NODES.items()}
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=6,
                            font_color="#D8DEE9", ax=ax)

    # Legend
    legend_items = [
        mpatches.Patch(color="#BF616A", label="Source"),
        mpatches.Patch(color="#A3BE8C", label="Destination"),
        mpatches.Patch(color="#EBCB8B", label="Route Stop"),
        mpatches.Patch(color="#5E81AC", label="Other Intersection"),
    ]
    ax.legend(handles=legend_items, loc="upper left",
              facecolor="#2E3440", labelcolor="#D8DEE9", fontsize=9)

    ax.set_title(title, color="#ECEFF4", fontsize=13, fontweight="bold", pad=12)
    ax.axis("off")
    plt.tight_layout()
    return fig


#  MODULE4: Streamlit GUI

st.set_page_config(
    page_title="Smart Rickshaw Route Planner",
    layout="wide"
)

st.markdown("""
<style>
    .main-title { font-size:2.2rem; font-weight:800; color:#2E75B6; text-align:center; }
    .subtitle   { font-size:1.0rem; color:#888; text-align:center; margin-bottom:1rem; }
    .metric-card {
        background: linear-gradient(135deg,#1a3a5c,#2e5f8a);
        border-radius:10px; padding:1rem; color:white; text-align:center;
    }
    .metric-value { font-size:1.8rem; font-weight:bold; color:#A3BE8C; }
    .metric-label { font-size:0.82rem; color:#ccc; }
    .route-step  {
        padding:6px 14px; margin:4px 0;
        border-left:4px solid #2E75B6;
        background: brown; border-radius:4px; font-size:0.93rem;
    }
    .route-step.start { border-left-color:#BF616A; background:red; }
    .route-step.end   { border-left-color:#A3BE8C; background:darkgreen; }
</style>
""", unsafe_allow_html=True)

#  Header 
st.markdown('<div class="main-title">Smart Rickshaw Route Planner</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Rawalpindi / Islamabad City Network — A* Informed Search Algorithm</div>', unsafe_allow_html=True)
st.markdown("---")

# Sidebar
with st.sidebar:
    st.markdown("## Route Settings")
    st.markdown("---")

    sorted_nodes = get_node_names_sorted()     # [(id, {name, x, y}), ...]
    node_labels  = [info["name"] for _, info in sorted_nodes]
    node_ids     = [nid          for nid, _  in sorted_nodes]

    source_name = st.selectbox("**Starting Point**",  node_labels, index=0)
    dest_name   = st.selectbox("**Destination**",     node_labels, index=5)

    st.markdown("---")
    traffic_mode = st.radio("**Traffic Condition**", list(TRAFFIC_MULTIPLIERS.keys()))

    st.markdown("---")
    find_btn = st.button("Find Optimal Route", use_container_width=True, type="primary")

# Main Panel
if find_btn:
    source_id = node_ids[node_labels.index(source_name)]
    dest_id   = node_ids[node_labels.index(dest_name)]

    if source_id == dest_id:
        st.warning("Source and destination are the same. Please select different locations.")
        st.stop()

    # Build graph with traffic weights
    adj = build_adjacency_list(traffic_mode)

    # Run A*
    result = astar_search(adj, source_id, dest_id)

    if not result["found"]:
        st.error("No route found.")
        st.stop()

    # Metrics row
    est_time_min = (result["cost"] / 25) * 60   # assume ~25 km/h avg speed
    c1, c2, c3, c4 = st.columns(4)
    for col, val, label in [
        (c1, f"{result['cost']:.2f} km",      "Total Distance"),
        (c2, f"{est_time_min:.0f} min",        "Est. Travel Time"),
        (c3, str(len(result["path"]) - 1),     "Road Segments"),
        (c4, str(result["nodes_explored"]),    "Nodes Explored (A*)"),
    ]:
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{val}</div>
                <div class="metric-label">{label}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Map + Route steps side-by-side 
    map_col, info_col = st.columns([2, 1])

    with map_col:
        st.markdown("### City Road Network")
        fig = draw_route(
            result["path"],
            title=f"A* Route: {source_name} → {dest_name}  |  {traffic_mode}"
        )
        st.pyplot(fig)

    with info_col:
        st.markdown("### Turn-by-Turn")
        st.markdown(f"**Algorithm:** A* (Informed Search)")
        st.markdown(f"**Traffic:** `{traffic_mode}`")
        st.markdown("---")

        for i, nid in enumerate(result["path"]):
            name = NODES[nid]["name"]
            if i == 0:
                st.markdown(f'<div class="route-step start"><b>START:</b> {name}</div>', unsafe_allow_html=True)
            elif i == len(result["path"]) - 1:
                st.markdown(f'<div class="route-step end"><b>END:</b> {name}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="route-step"> Step {i}: {name}</div>', unsafe_allow_html=True)

else:
    # ── Welcome / preview screen ──────────────────────────────────────────────
    st.markdown("""
    ### Select routes and then press find route button
    """)
    fig = draw_route([], title="Rawalpindi / Islamabad Road Network — Select a route to begin")
    st.pyplot(fig)

# Footer
st.markdown("---")
st.markdown(
    "<center><small>Abdul Haseeb | Saad Ahmad | Ahmed Hassan</small></center>",
    unsafe_allow_html=True
)
st.markdown(
    "<center><small>i233077 | i233076 | i233057</small></center>",
    unsafe_allow_html=True
)
