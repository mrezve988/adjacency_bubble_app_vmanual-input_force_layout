# ✅ Full app combining working layout with scoring + suggestions
import streamlit as st
import pandas as pd
from pyvis.network import Network
import streamlit.components.v1 as components
import tempfile
import os
import itertools
import plotly.graph_objects as go
import networkx as nx

st.set_page_config(page_title="Adjacency-Based Room Planner", layout="wide")
st.title("🏠 Adjacency-Based Room Planner")

room_data = {
    "Living Room": {"length": 15, "width": 12, "privacy": "Public"},
    "Kitchen": {"length": 10, "width": 10, "privacy": "Service"},
    "Dining": {"length": 12, "width": 10, "privacy": "Public"},
    "Bedroom 1": {"length": 12, "width": 12, "privacy": "Private"},
    "Bedroom 2": {"length": 10, "width": 10, "privacy": "Private"},
    "Bedroom 3": {"length": 10, "width": 10, "privacy": "Private"},
    "Toilet 1": {"length": 6, "width": 5, "privacy": "Service"},
    "Bath": {"length": 8, "width": 6, "privacy": "Private"},
    "Store": {"length": 8, "width": 6, "privacy": "Service"}
}
standard_adjacencies = [
    ("Living Room"↔ "Dining"), ("Dining", "Kitchen"), ("Kitchen", "Store"),
    ("Dining", "Bedroom 1"), ("Bedroom 1", "Bath"), ("Dining", "Bedroom 2"),
    ("Dining", "Bedroom 3"), ("Living Room", "Toilet 1")
]
room_list = list(room_data.keys())

col1, col2 = st.columns(2)
with col1:
    st.markdown("### 🧭 Standard Room Info + Adjacency")
    adj_map = {room: [] for room in room_list}
    for a, b in standard_adjacencies:
        adj_map[a].append(b)
        adj_map[b].append(a)
    standard_df = pd.DataFrame([
        {"Room": room,
         "Area (L x W = A)": f"{d['length']}' x {d['width']}' = {d['length']*d['width']} sqft",
         "Privacy": d["privacy"],
         "Adjacent To": ", ".join(adj_map[room])} for room, d in room_data.items()
    ])
    st.dataframe(standard_df, use_container_width=True)

with col2:
    st.markdown("### ✍️ User Input")
    with st.expander("📐 Room Sizes and Privacy"):
        user_inputs = []
        for room in room_list:
            c1, c2, c3 = st.columns([1, 1, 2])
            length = c1.number_input(f"{room} Length", 1, 50, room_data[room]["length"], key=f"{room}_l")
            width = c2.number_input(f"{room} Width", 1, 50, room_data[room]["width"], key=f"{room}_w")
            privacy = c3.selectbox(f"{room} Privacy", ["Public", "Private", "Service"],
                                   index=["Public", "Private", "Service"].index(room_data[room]["privacy"]), key=f"{room}_p")
            user_inputs.append({
                "Room": room,
                "Area (L x W = A)": f"{length}' x {width}' = {length * width} sqft",
                "Privacy": privacy
            })
        user_df = pd.DataFrame(user_inputs)

    with st.expander("✅ Define Adjacencies"):
        user_adjacencies = []
        for a, b in itertools.combinations(room_list, 2):
            if st.checkbox(f"{a} adjacent to {b}", key=f"{a}_{b}"):
                user_adjacencies.append((a, b))

privacy_colors = {"Public": "green", "Private": "blue", "Service": "orange"}

def draw_static(df, edges, title):
    G = nx.Graph()
    sizes, colors = [], []
    for _, r in df.iterrows():
        name = r["Room"]
        dims = r["Area (L x W = A)"].split("=")[0].split("x")
        area = int(dims[0].strip().replace("'", "")) * int(dims[1].strip().replace("'", ""))
        sizes.append(area)
        colors.append(privacy_colors.get(r["Privacy"], "gray"))
        G.add_node(name)
    for a, b in edges:
        if a in G.nodes and b in G.nodes:
            G.add_edge(a, b)
    pos = nx.spring_layout(G, seed=42)
    x, y = [], []
    for edge in G.edges():
        x += [pos[edge[0]][0], pos[edge[1]][0], None]
        y += [pos[edge[0]][1], pos[edge[1]][1], None]
    node_x, node_y = zip(*[pos[n] for n in G.nodes()])
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y, mode='lines', line=dict(width=1)))
    fig.add_trace(go.Scatter(x=node_x, y=node_y, mode='markers+text', marker=dict(size=[a/3 for a in sizes], color=colors),
                             text=list(G.nodes()), textposition="bottom center"))
    fig.update_layout(title=title, showlegend=False, height=500, width=500)
    st.plotly_chart(fig)

def draw_interactive(df, edges):
    net = Network(height="500px", width="100%", bgcolor="#fff")
    for _, r in df.iterrows():
        room = r["Room"]
        dims = r["Area (L x W = A)"].split("=")[0].split("x")
        area = int(dims[0].strip().replace("'", "")) * int(dims[1].strip().replace("'", ""))
        net.add_node(room, label=room, size=area / 2, color=privacy_colors.get(r["Privacy"], "gray"))
    for a, b in edges:
        if a in df["Room"].values and b in df["Room"].values:
            net.add_edge(a, b)
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".html")
    net.save_graph(tmp.name)
    components.html(open(tmp.name, 'r').read(), height=500)
    os.unlink(tmp.name)

# Visual comparison
st.markdown("### 📐 Plan Diagrams")
c1, c2 = st.columns(2)
with c1:
    st.markdown("#### Standard")
    draw_static(standard_df, standard_adjacencies, "Standard Bubble")
with c2:
    st.markdown("#### User")
    draw_static(user_df, user_adjacencies, "User Bubble")

c3, c4 = st.columns(2)
with c3:
    st.markdown("#### Standard Interactive")
    draw_interactive(standard_df, standard_adjacencies)
with c4:
    st.markdown("#### User Interactive")
    draw_interactive(user_df, user_adjacencies)

# ---------- Scoring ----------
st.markdown("### 🧮 Scoring System")
std_set = set(tuple(sorted(p)) for p in standard_adjacencies)
usr_set = set(tuple(sorted(p)) for p in user_adjacencies)
intersection = std_set & usr_set
union = std_set | usr_set
jaccard_score = len(intersection) / len(union) if union else 1

size_devs = []
for room in room_list:
    std_area = room_data[room]['length'] * room_data[room]['width']
    user_dims = user_df[user_df["Room"] == room]["Area (L x W = A)"].values[0].split("=")[1].strip().split()[0]
    user_area = int(user_dims)
    size_devs.append(abs(std_area - user_area) / std_area)
size_score = 1 - (sum(size_devs) / len(size_devs))

privacy_mismatches = sum([
    room_data[room]['privacy'] != user_df[user_df["Room"] == room]["Privacy"].values[0]
    for room in room_list
])
privacy_score = 1 - (privacy_mismatches / len(room_list))
final_score = round((0.4 * jaccard_score + 0.4 * size_score + 0.2 * privacy_score) * 100, 2)

st.metric("📊 Adjacency Match", f"{jaccard_score*100:.1f}%")
st.metric("📐 Size Accuracy", f"{size_score*100:.1f}%")
st.metric("🔒 Privacy Match", f"{privacy_score*100:.1f}%")
st.metric("✅ Final Score", f"{final_score}%")

# ---------- Suggestions ----------
st.markdown("### 💡 Suggestions")
improvements = []

missing = std_set - usr_set
for a, b in sorted(missing):
    improvements.append(f"Connect **{a}** to **{b}** (missing adjacency).")

for room in room_list:
    std_priv = room_data[room]['privacy']
    usr_priv = user_df[user_df["Room"] == room]["Privacy"].values[0]
    if std_priv != usr_priv:
        improvements.append(f"Change **{room}** privacy from **{usr_priv}** to **{std_priv}**.")

for room in room_list:
    std_area = room_data[room]['length'] * room_data[room]['width']
    usr_area = int(user_df[user_df["Room"] == room]["Area (L x W = A)"].values[0].split("=")[1].split()[0])
    deviation = abs(std_area - usr_area) / std_area * 100
    if deviation > 20:
        improvements.append(f"Adjust **{room}** size (deviation is {deviation:.1f}%).")

if improvements:
    for item in improvements:
        st.markdown(f"- {item}")
else:
    st.success("✅ Your layout is well-matched!")

# ---------- Circulation Distance Estimation ----------
st.markdown("### 📏 Estimated Circulation Distance")

import numpy as np
from sklearn.linear_model import LinearRegression

def estimate_circulation(df, edges):
    # Assign arbitrary layout positions (grid layout for simplification)
    positions = {}
    for i, room in enumerate(df["Room"]):
        x = i % 3
        y = i // 3
        positions[room] = (x * 10, y * 10)  # grid with 10ft spacing

    distances = []
    for a, b in edges:
        if a in positions and b in positions:
            xa, ya = positions[a]
            xb, yb = positions[b]
            dist = np.sqrt((xa - xb) ** 2 + (ya - yb) ** 2)
            distances.append(dist)

    total_dist = sum(distances)
    return round(total_dist, 2)

std_circ = estimate_circulation(standard_df, standard_adjacencies)
usr_circ = estimate_circulation(user_df, user_adjacencies)

st.metric("📐 Standard Circulation Estimate", f"{std_circ} ft")
st.metric("🧑‍🎨 User Circulation Estimate", f"{usr_circ} ft")
if usr_circ > std_circ:
    st.warning("⚠️ Consider optimizing layout to reduce circulation distance.")
else:
    st.success("✅ Circulation distance is optimized.")

# ---------- Circulation Path Visualization ----------
st.markdown("### 🗺 Circulation Path Visualization")

def draw_force_circulation(df, edges, title):
    import networkx as nx
    import plotly.graph_objects as go
    import numpy as np

    G = nx.Graph()
    color_map = []
    size_map = []

    for _, row in df.iterrows():
        room = row["Room"]
        dims = row["Area (L x W = A)"].split("=")[0].split("x")
        length = int(dims[0].strip().replace("'", ""))
        width = int(dims[1].strip().replace("'", ""))
        area = length * width
        G.add_node(room)
        color_map.append(privacy_colors.get(row["Privacy"], "gray"))
        size_map.append(area)

    for a, b in edges:
        if a in G.nodes and b in G.nodes:
            G.add_edge(a, b)

    pos = nx.spring_layout(G, seed=42)  # logical spring layout
    edge_x, edge_y = [], []
    edge_labels = []
    for a, b in G.edges():
        x0, y0 = pos[a]
        x1, y1 = pos[b]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]
        dist = round(np.sqrt((x0 - x1)**2 + (y0 - y1)**2) * 10, 1)
        edge_labels.append(((x0 + x1) / 2, (y0 + y1) / 2, dist))

    node_x, node_y = zip(*[pos[n] for n in G.nodes()])

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=edge_x, y=edge_y, mode='lines', line=dict(width=1, color='gray')))
    for x, y, label in edge_labels:
        fig.add_trace(go.Scatter(x=[x], y=[y], text=[f"{label} ft"], mode='text', textfont=dict(size=10)))
    fig.add_trace(go.Scatter(
        x=node_x, y=node_y, mode='markers+text',
        marker=dict(size=[a / 3 for a in size_map], color=color_map, line=dict(width=2, color='DarkSlateGrey')),
        text=list(G.nodes()), textposition="bottom center"
    ))
    fig.update_layout(title=title, showlegend=False, height=550, width=550, margin=dict(l=10, r=10, t=30, b=10))
    st.plotly_chart(fig)

col_fd1, col_fd2 = st.columns(2)
with col_fd1:
    st.markdown("#### 🧭 Standard Force-Directed")
    draw_force_circulation(standard_df, standard_adjacencies, "Standard Circulation")

with col_fd2:
    st.markdown("#### 🧑‍🎨 User Force-Directed")
    draw_force_circulation(user_df, user_adjacencies, "User Circulation")
