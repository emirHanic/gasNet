"""
GasNET - Simple Compressible Gas Pipeline Network Calculator
==============================================================

WHAT THIS IS
------------
A small, functional Streamlit app for computing pressure distribution and
flow in a BRANCHING TREE (node-to-node, splits allowed) network of gas
pipeline segments, using steady, isothermal, compressible-flow equations.

Flow is specified per NODE, not per pipe: mark each node Inlet (gas enters
here), Outlet (gas leaves here) or Passing (junction only, no external
flow), and give the flow at Inlet/Outlet nodes. Segment flow is then
DERIVED from these node demands via mass balance (see "NETWORK MODEL"
below) - you never enter a pipe's own flow rate directly.

Pressure works the same way: pick exactly one node anywhere in the network
- inlet, outlet, or a node in the middle - and give its pressure. Every
other node's pressure, including the inlet's if you picked a different
reference point, is then SOLVED from that one anchor (forward toward
downstream nodes, backward - algebraically inverted, always solvable - to
work out what an upstream/inlet pressure would need to be). This lets you
ask "what inlet pressure do I need to guarantee X bar at this delivery
point?" directly, without first guessing an inlet pressure and iterating.

SUPPORTED FLOW EQUATIONS
-------------------------
1. General Fundamental Isothermal Flow Equation
        P_up^2 - e^s * P_down^2 = (16 f Le Z R Tf) / (pi^2 D^5 M) * mdot^2
   with the Darcy friction factor f obtained from either:
        - Colebrook-White (iterated), or
        - Swamee-Jain (explicit approximation)
   This form is exact for steady isothermal compressible pipe flow
   (kinetic-energy term neglected, standard pipeline-engineering practice)
   and is unit-system consistent (SI throughout). "up"/"down" here mean the
   PHYSICAL flow direction through the pipe, which can be opposite to how
   you drew it (node_from -> node_to) if a mid-network inlet forces gas
   back toward the source - see NETWORK MODEL.

2. Weymouth, Panhandle A, Panhandle B
   These classic empirical equations are implemented as DIFFERENT
   TRANSMISSION-FACTOR (i.e. different Darcy friction-factor) correlations
   plugged into the SAME general fundamental equation above:
        F = 2 / sqrt(f)          (transmission factor <-> Darcy f)
        Weymouth:     F = 11.18 * D_inches^(1/6)                (Re-independent)
        Panhandle A:  F = 7.211 * Re^0.07305 * E                (E = pipeline efficiency)
        Panhandle B:  F = 16.7  * Re^0.01961 * E
   NOTE: These historical correlations were originally fit in imperial units
   (diameter in inches). The published constants above are widely cited
   (e.g. Menon, "Gas Pipeline Hydraulics"), but different references quote
   slightly different constants. Cross-check against your governing
   standard before using for final engineering design. The General equation
   (Colebrook-White/Swamee-Jain) is the most rigorous, unit-independent
   option and is recommended as the default.

NETWORK MODEL
-------------
- Topology: a BRANCHING TREE. A node may feed multiple downstream segments
  (a split); a node fed by MORE THAN ONE segment (a merge) or a cycle is
  rejected with a clear error - resolving flow splits at a real merge/loop
  needs an iterative network solver (Hardy Cross / Newton-Raphson), out of
  scope here.
- Node flow: every node is Passing (0 external flow), Inlet (+injection,
  Sm3/h) or Outlet (-withdrawal, Sm3/h) - an input, except for a
  topological ROOT node (no parent pipe), whose flow is always DERIVED
  (= the net of everything downstream of it), since a root has no other
  pipe to carry a mismatch. Every other segment's flow is the net demand
  of its entire downstream subtree (its own node's demand plus all of its
  children's), computed by summing from the leaves inward. If a subtree is
  a net INJECTOR (its own mid-network inlet exceeds its own demand), the
  segment feeding it carries flow in REVERSE (down to up) - handled
  automatically and flagged in the results.
- Node flow uses the GLOBAL default gas gravity/base conditions to convert
  Sm3/h to kg/s (not a segment's individual gas-property override), since a
  node's demand isn't tied to any one adjacent pipe.
- Pressure: exactly one node per connected tree must have a known pressure
  (the "anchor"). Every other node's pressure is solved outward from it
  along the tree, using the segment equation forward (anchor is physically
  upstream) or its algebraic inverse (anchor is physically downstream) as
  needed. The inverse direction has no infeasibility case - a required
  upstream pressure always exists - which is what supports "how much inlet
  pressure would I need" queries.

ASSUMPTIONS
-----------
- Steady-state, ISOTHERMAL, compressible single-phase gas flow.
- Ideal-gas-like behaviour corrected by a constant compressibility factor Z
  per segment (no full equation of state).
- Gas gravity G (relative to air), Z, viscosity mu are constant over each
  segment (may differ segment to segment) for the pressure-drop physics;
  node-flow unit conversion always uses the global default gas gravity.
- Kinetic-energy (velocity head) change term is neglected, as is standard
  for the "fundamental" pipeline flow equation.
- Elevation effect included through the classical exponential correction
  factor s (static-head term), applied per segment using the PHYSICAL
  upstream/downstream elevations.
- All pressures are ABSOLUTE.

HOW TO RUN
----------
    pip install streamlit pandas
    streamlit run app.py
"""

import json
import math
from collections import defaultdict

import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------
R_UNIVERSAL = 8.314462618   # J/(mol.K)
M_AIR = 0.0289647           # kg/mol
G_EARTH = 9.80665           # m/s^2

# ---------------------------------------------------------------------------
# Unit conversion helpers (UI uses practical units; calcs use SI internally)
# ---------------------------------------------------------------------------
def c_to_k(t_c):
    return t_c + 273.15


def bar_to_pa(p_bar):
    return p_bar * 1.0e5


def pa_to_bar(p_pa):
    return p_pa / 1.0e5


def mm_to_m(x_mm):
    return x_mm / 1000.0


def cp_to_pas(mu_cp):
    return mu_cp * 1.0e-3


def m_to_in(x_m):
    return x_m / 0.0254


# ---------------------------------------------------------------------------
# Friction factor correlations (Darcy-Weisbach f)
# ---------------------------------------------------------------------------
def colebrook_white(re, rel_rough, tol=1.0e-8, maxiter=100):
    """Iterated Colebrook-White implicit equation for Darcy friction factor."""
    if re <= 0:
        return float("nan")
    if re < 2300.0:
        return 64.0 / re  # laminar flow
    f = 0.02  # initial guess
    for _ in range(maxiter):
        rhs = -2.0 * math.log10(rel_rough / 3.7 + 2.51 / (re * math.sqrt(f)))
        f_new = 1.0 / (rhs ** 2)
        if abs(f_new - f) < tol:
            return f_new
        f = f_new
    return f


def swamee_jain(re, rel_rough):
    """Explicit Swamee-Jain approximation to Colebrook-White."""
    if re <= 0:
        return float("nan")
    if re < 2300.0:
        return 64.0 / re  # laminar flow
    return 0.25 / (math.log10(rel_rough / 3.7 + 5.74 / re ** 0.9)) ** 2


def transmission_factor_to_darcy_f(big_f):
    """Convert a transmission factor F (F = 2/sqrt(f)) to Darcy friction factor f."""
    return 4.0 / big_f ** 2


# ---------------------------------------------------------------------------
# Segment-level compressible flow physics
# ---------------------------------------------------------------------------
def solve_segment_coeffs(seg, mdot, h_up, h_down, equation, ff_method, k_ratio):
    """
    Compute the friction factor, elevation correction and lumped K
    coefficient for one pipe segment, given the flow MAGNITUDE (kg/s) and
    the PHYSICAL upstream/downstream elevations (h_up may be the segment's
    topological downstream end if flow is reversed - the caller decides).

    Returns a dict of coefficients, or a string error message.
    """
    length_m = seg["L_m"]
    dia_m = seg["D_m"]
    rough_m = seg["rough_m"]
    gas_g = seg["G"]
    tf_k = seg["Tf_K"]
    z_factor = seg["Z"]
    mu_pas = seg["mu_pas"]
    eff = seg["E"]

    if length_m <= 0 or dia_m <= 0 or tf_k <= 0 or z_factor <= 0 or mu_pas <= 0 or mdot < 0:
        return "Invalid input (length/diameter/temperature/Z/viscosity must be positive)."

    molar_mass = gas_g * M_AIR
    area = math.pi / 4.0 * dia_m ** 2
    re = 4.0 * mdot / (math.pi * dia_m * mu_pas) if mdot > 0 else 0.0
    rel_rough = rough_m / dia_m

    if equation == "General (Colebrook-White / Swamee-Jain)":
        if re <= 0:
            f = float("nan")
        elif ff_method == "Colebrook-White":
            f = colebrook_white(re, rel_rough)
        else:
            f = swamee_jain(re, rel_rough)
    elif equation == "Weymouth":
        dia_in = m_to_in(dia_m)
        big_f = 11.18 * dia_in ** (1.0 / 6.0)
        f = transmission_factor_to_darcy_f(big_f)
    elif equation == "Panhandle A":
        big_f = 7.211 * (re ** 0.07305) * eff if re > 0 else float("nan")
        f = transmission_factor_to_darcy_f(big_f)
    elif equation == "Panhandle B":
        big_f = 16.7 * (re ** 0.01961) * eff if re > 0 else float("nan")
        f = transmission_factor_to_darcy_f(big_f)
    else:
        return f"Unknown equation: {equation}"

    if f is None or math.isnan(f) or f <= 0:
        return "Could not compute a valid friction factor (check flow rate / diameter / viscosity)."

    # Elevation correction (classical exponential static-head term), using
    # the PHYSICAL flow direction's elevations.
    s = 2.0 * G_EARTH * molar_mass * (h_down - h_up) / (z_factor * R_UNIVERSAL * tf_k)
    le = length_m if abs(s) < 1.0e-8 else length_m * (math.exp(s) - 1.0) / s

    k_coeff = 16.0 * f * z_factor * R_UNIVERSAL * tf_k / (math.pi ** 2 * dia_m ** 5 * molar_mass)

    return {
        "f": f, "re": re, "s": s, "le": le, "k_coeff": k_coeff,
        "molar_mass": molar_mass, "area": area, "z": z_factor, "tf_k": tf_k,
    }


def solve_forward_pressure(p_up_pa, mdot, coeffs):
    """Given the physically-upstream pressure, solve for the downstream one."""
    rhs = p_up_pa ** 2 - coeffs["k_coeff"] * coeffs["le"] * mdot ** 2
    if rhs <= 0:
        return None, (
            "Not feasible: pressure drop exceeds available upstream pressure "
            "(flow too high / diameter too small / segment too long)."
        )
    return math.sqrt(rhs / math.exp(coeffs["s"])), None


def solve_backward_pressure(p_down_pa, mdot, coeffs):
    """
    Given the physically-downstream pressure, solve for the upstream one -
    the algebraic inverse of solve_forward_pressure. Always has a solution
    (sum of non-negative terms under the square root): there is always a
    well-defined upstream pressure that delivers a given downstream
    pressure at a given flow, which is exactly what supports "how much
    inlet pressure do I need" queries.
    """
    p_up_sq = coeffs["k_coeff"] * coeffs["le"] * mdot ** 2 + math.exp(coeffs["s"]) * p_down_pa ** 2
    return math.sqrt(p_up_sq), None


# ---------------------------------------------------------------------------
# Topology (structure only - which nodes connect to which)
# ---------------------------------------------------------------------------
def build_topology(segments):
    """
    Build a branching-tree topology from segment node_from/node_to fields and
    validate it. Only BRANCHING (fan-out from a source) is supported - a node
    fed by more than one segment (a merge) or a cycle is rejected with a
    clear error, since resolving flow splits at a real merge/loop needs an
    iterative network solver, out of scope for this tool.

    Returns:
        children: dict[node_name] -> list of segment dicts leaving that node
        roots: sorted list of source node names (no incoming segment)
        errors: list of human-readable topology problems (empty if valid)
        bad_nodes: set of node names implicated in a topology error
        bad_seg_ids: set of segment ids implicated in a topology error
    """
    children = defaultdict(list)
    incoming = defaultdict(int)
    incoming_segs = defaultdict(list)
    all_nodes = set()
    errors = []
    bad_nodes = set()
    bad_seg_ids = set()

    for seg in segments:
        nf, nt = seg["node_from"], seg["node_to"]
        all_nodes.add(nf)
        all_nodes.add(nt)
        if nf == nt:
            errors.append(
                f"Segment '{seg['name']}' has the same upstream and downstream node "
                f"('{nf}') - a pipe cannot loop back on itself."
            )
            bad_seg_ids.add(seg["id"])
            bad_nodes.add(nf)
            continue
        children[nf].append(seg)
        incoming[nt] += 1
        incoming_segs[nt].append(seg)

    for node, count in incoming.items():
        if count > 1:
            names = ", ".join(s["name"] for s in incoming_segs[node])
            errors.append(
                f"Node '{node}' is fed by {count} segments ({names}) - merging flows "
                "is not supported (branching tree only)."
            )
            bad_nodes.add(node)
            bad_seg_ids.update(s["id"] for s in incoming_segs[node])

    roots = sorted(n for n in all_nodes if incoming.get(n, 0) == 0)

    # Any node reachable a second time during this walk necessarily has
    # incoming > 1 and was already reported as a merge above (a cycle
    # attached to a root always creates a second incoming edge into some
    # node on the loop); we only need this walk to find nodes NOT reachable
    # from any root at all, i.e. a cycle fully disconnected from a source.
    visited = set(roots)
    queue = list(roots)
    while queue:
        node = queue.pop(0)
        for seg in children.get(node, []):
            nxt = seg["node_to"]
            if nxt in visited:
                continue
            visited.add(nxt)
            queue.append(nxt)

    unreached = all_nodes - visited
    if unreached:
        errors.append(
            f"Node(s) {', '.join(sorted(unreached))} are not reachable from any source "
            "node (likely part of a cycle with no source)."
        )
        bad_nodes.update(unreached)

    return children, roots, errors, bad_nodes, bad_seg_ids


def compute_components(children, roots):
    """Map each node to the root of the tree/component it belongs to."""
    comp_of = {}
    for r in roots:
        seen = {r}
        queue = [r]
        while queue:
            node = queue.pop(0)
            comp_of[node] = r
            for seg in children.get(node, []):
                nxt = seg["node_to"]
                if nxt not in seen:
                    seen.add(nxt)
                    queue.append(nxt)
    return comp_of


# ---------------------------------------------------------------------------
# Mass balance: node demands -> per-segment flow (magnitude + direction)
# ---------------------------------------------------------------------------
def compute_node_local_demand(node_names, node_settings, roots, global_G, tb_k, pb_pa):
    """
    Convert each node's Sm3/h input into a signed local net demand in kg/s
    (withdrawal positive, injection negative). Root nodes always contribute
    0 here - their flow is derived, not user input (see module docstring).
    Uses the GLOBAL default gas gravity/base conditions, not any segment's
    per-segment override, since a node's demand isn't tied to one pipe.
    """
    molar_mass = global_G * M_AIR
    rho_base = pb_pa * molar_mass / (R_UNIVERSAL * tb_k)
    local_demand = {}
    for n in node_names:
        if n in roots:
            local_demand[n] = 0.0
            continue
        cfg = node_settings[n]
        if cfg["role"].startswith("Passing"):
            local_demand[n] = 0.0
        else:
            mdot = (cfg["flow_sm3h"] / 3600.0) * rho_base
            local_demand[n] = mdot if cfg["role"].startswith("Outlet") else -mdot
    return local_demand, rho_base


def compute_segment_flows(children, roots, local_demand):
    """
    Post-order sum: the flow through the segment feeding a node equals that
    node's own local demand plus the (already-computed) flow of every
    segment leaving it. Returns signed kg/s per segment id (positive =
    node_from -> node_to, negative = physically reversed) and the same
    subtree total per node (used to display each root's derived flow).
    """
    seg_flow = {}
    subtree_demand = {}

    def visit(node):
        total = local_demand.get(node, 0.0)
        for seg in children.get(node, []):
            child_total = visit(seg["node_to"])
            seg_flow[seg["id"]] = child_total
            total += child_total
        subtree_demand[node] = total
        return total

    for r in roots:
        visit(r)
    return seg_flow, subtree_demand


# ---------------------------------------------------------------------------
# Pressure solve: BFS outward from the known-pressure anchor node(s)
# ---------------------------------------------------------------------------
def build_adjacency(segments):
    """node -> list of (segment, is_forward); is_forward True means this
    node is the segment's topological node_from."""
    adj = defaultdict(list)
    for seg in segments:
        adj[seg["node_from"]].append((seg, True))
        adj[seg["node_to"]].append((seg, False))
    return adj


def solve_network_pressures(segments, node_names, node_settings, seg_flow, equation, ff_method, k_ratio):
    """
    BFS outward from every node marked as a pressure anchor, solving each
    adjacent segment forward (anchor is physically upstream) or backward
    (anchor is physically downstream) as needed. Returns:
        node_pressure_pa: dict node -> pressure in Pa, or None if unsolved
        seg_results: dict seg_id -> result dict (f, re, mdot, P_up, P_down,
                     v_up, v_down, mach_down, phys_up, phys_down, error)
    """
    adjacency = build_adjacency(segments)
    node_pressure = {}
    seg_results = {}

    anchors = [n for n in node_names if node_settings[n]["pressure_known"]]
    visited = set(anchors)
    for a in anchors:
        node_pressure[a] = bar_to_pa(node_settings[a]["pressure_bar"])
    queue = list(anchors)

    while queue:
        node = queue.pop(0)
        p_known = node_pressure[node]
        for seg, is_forward in adjacency[node]:
            other = seg["node_to"] if is_forward else seg["node_from"]
            if other in visited:
                continue
            visited.add(other)
            queue.append(other)

            mdot_signed = seg_flow.get(seg["id"], 0.0)
            mdot = abs(mdot_signed)
            if mdot_signed >= 0:
                phys_up, phys_down = seg["node_from"], seg["node_to"]
                h_up, h_down = seg["elev_up_m"], seg["elev_dn_m"]
            else:
                phys_up, phys_down = seg["node_to"], seg["node_from"]
                h_up, h_down = seg["elev_dn_m"], seg["elev_up_m"]

            result = {
                "mdot": mdot, "phys_up": phys_up, "phys_down": phys_down, "error": None,
                "f": None, "re": None, "P_up": None, "P_down": None,
                "v_up": None, "v_down": None, "mach_down": None,
            }

            coeffs = solve_segment_coeffs(seg, mdot, h_up, h_down, equation, ff_method, k_ratio)
            if isinstance(coeffs, str):
                result["error"] = coeffs
                seg_results[seg["id"]] = result
                node_pressure[other] = None
                continue

            result["f"], result["re"] = coeffs["f"], coeffs["re"]

            if node == phys_up:
                p_up_val = p_known
                p_down_val, err = solve_forward_pressure(p_known, mdot, coeffs)
            else:
                p_down_val = p_known
                p_up_val, err = solve_backward_pressure(p_known, mdot, coeffs)

            if err:
                result["error"] = err
                node_pressure[other] = None
            else:
                result["P_up"], result["P_down"] = p_up_val, p_down_val
                rho_up = p_up_val * coeffs["molar_mass"] / (coeffs["z"] * R_UNIVERSAL * coeffs["tf_k"])
                rho_down = p_down_val * coeffs["molar_mass"] / (coeffs["z"] * R_UNIVERSAL * coeffs["tf_k"])
                result["v_up"] = mdot / (rho_up * coeffs["area"]) if rho_up > 0 else float("nan")
                result["v_down"] = mdot / (rho_down * coeffs["area"]) if rho_down > 0 else float("nan")
                sonic_v = math.sqrt(k_ratio * coeffs["z"] * R_UNIVERSAL * coeffs["tf_k"] / coeffs["molar_mass"])
                result["mach_down"] = result["v_down"] / sonic_v if sonic_v > 0 else float("nan")
                node_pressure[other] = p_down_val if other == phys_down else p_up_val

            seg_results[seg["id"]] = result

    # Segments never reached (e.g. beyond a failed segment, or an
    # unreachable/unsolved component) still need a row in the results table.
    for seg in segments:
        if seg["id"] not in seg_results:
            seg_results[seg["id"]] = {
                "mdot": abs(seg_flow.get(seg["id"], 0.0)), "phys_up": None, "phys_down": None,
                "error": "Skipped: not reachable from the pressure anchor (check topology/anchor placement).",
                "f": None, "re": None, "P_up": None, "P_down": None,
                "v_up": None, "v_down": None, "mach_down": None,
            }

    return node_pressure, seg_results


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------
def _dot_escape(text):
    return str(text).replace('"', "'")


def build_network_dot(segments, roots, node_settings, node_pressure_bar, seg_flow, seg_results, bad_nodes, bad_seg_ids):
    """Render the network topology (and, once solved, roles/pressures/dP) as
    a Graphviz DOT string. st.graphviz_chart renders DOT client-side in the
    browser, so no extra Python package or system Graphviz install is needed."""
    all_nodes = set(roots)
    for seg in segments:
        all_nodes.add(seg["node_from"])
        all_nodes.add(seg["node_to"])

    lines = [
        "digraph GasNet {",
        "rankdir=LR;",
        'node [shape=box, style="rounded,filled", fontname="Helvetica", '
        'fillcolor="#eef2f7", color="#4a6fa5"];',
        'edge [fontname="Helvetica", fontsize=10, color="#4a6fa5"];',
    ]
    for node in sorted(all_nodes):
        label = _dot_escape(node)
        if node in node_pressure_bar:
            label += f"\\n{node_pressure_bar[node]:.2f} bar a"
        fill, border = "#eef2f7", "#4a6fa5"
        role = node_settings.get(node, {}).get("role", "") if node_settings else ""
        if node in bad_nodes:
            fill, border = "#f8d7da", "#c0392b"
        elif node in roots:
            fill, border = "#d4edda", "#2e7d32"
        elif role.startswith("Inlet"):
            fill, border = "#cfe8ff", "#1565c0"
        elif role.startswith("Outlet"):
            fill, border = "#ffe6cc", "#d17a00"
        lines.append(f'"{_dot_escape(node)}" [label="{label}", fillcolor="{fill}", color="{border}"];')

    for seg in segments:
        res = seg_results.get(seg["id"]) if seg_results else None
        signed = seg_flow.get(seg["id"], 0.0) if seg_flow else 0.0
        label = _dot_escape(seg["name"])
        color, style = "#4a6fa5", "solid"
        if seg["id"] in bad_seg_ids:
            color = "#c0392b"
        elif res is not None and res.get("error"):
            color = "#c0392b"
            label += "\\nERROR"
        else:
            if res is not None and res.get("P_up") is not None and res.get("P_down") is not None:
                label += f"\\ndP={pa_to_bar(res['P_up'] - res['P_down']):.3f} bar"
            if signed < 0:
                style = "dashed"
                label += "\\n(reversed flow)"
        lines.append(
            f'"{_dot_escape(seg["node_from"])}" -> "{_dot_escape(seg["node_to"])}" '
            f'[label="{label}", color="{color}", fontcolor="{color}", style="{style}"];'
        )
    lines.append("}")
    return "\n".join(lines)


def build_text_tree(children, roots):
    """Plain-text indented preview of the branching tree, as a fallback/
    supplement to the graphviz diagram."""
    lines = []

    def walk(node, prefix):
        for seg in children.get(node, []):
            lines.append(f"{prefix}{node} --[{seg['name']}]--> {seg['node_to']}")
            walk(seg["node_to"], prefix + "    ")

    for r in roots:
        walk(r, "")
    return "\n".join(lines) if lines else "(no segments)"


# ---------------------------------------------------------------------------
# Streamlit app
# ---------------------------------------------------------------------------
st.set_page_config(page_title="GasNET", layout="wide")

st.title("GasNET — Compressible Gas Pipeline Network Calculator")
st.caption(
    "Steady, isothermal, compressible gas flow in a branching-tree pipeline network "
    "(splits allowed, no merges/loops). Flow and pressure are specified per NODE. "
    "All pressures are absolute. See the module docstring in app.py for assumptions."
)

# Placeholder positioned at the top of the sidebar; filled in at the end of
# the script (see bottom) once every other widget has run at least once.
save_load_container = st.sidebar.container()

# --- session-state segment bookkeeping -------------------------------------
if "seg_counter" not in st.session_state:
    st.session_state.seg_counter = 0
if "seg_ids" not in st.session_state:
    st.session_state.seg_ids = []


def add_segment():
    st.session_state.seg_counter += 1
    sid = st.session_state.seg_counter
    n = len(st.session_state.seg_ids) + 1
    st.session_state.seg_ids.append(sid)
    st.session_state[f"name_{sid}"] = f"Seg-{n}"
    st.session_state[f"from_{sid}"] = f"N{n}"
    st.session_state[f"to_{sid}"] = f"N{n + 1}"
    st.session_state[f"L_{sid}"] = 1000.0
    st.session_state[f"D_{sid}"] = 150.0
    st.session_state[f"rough_{sid}"] = 0.046
    st.session_state[f"elevup_{sid}"] = 0.0
    st.session_state[f"elevdn_{sid}"] = 0.0
    st.session_state[f"override_{sid}"] = False
    st.session_state[f"G_{sid}"] = 0.6
    st.session_state[f"Tf_{sid}"] = 15.0
    st.session_state[f"Z_{sid}"] = 0.95
    st.session_state[f"mu_{sid}"] = 0.0110


def remove_segment(sid):
    if sid in st.session_state.seg_ids:
        st.session_state.seg_ids.remove(sid)


# --- save / load the whole network as a JSON file ----------------------------
# Lets you persist a big network without needing the server to stay running:
# download the current input data to a file now, and re-upload it later (in
# this session, a fresh one, or after a restart) to restore it exactly.
SEGMENT_FIELD_KEYS = {
    "name": "name_", "from": "from_", "to": "to_", "L": "L_", "D": "D_",
    "rough": "rough_", "elevup": "elevup_", "elevdn": "elevdn_", "override": "override_",
    "G": "G_", "Tf": "Tf_", "Z": "Z_", "mu": "mu_",
}
GLOBAL_FIELD_KEYS = {
    "equation": "eq_select", "ff_method": "ff_method_radio", "eff_factor": "eff_factor_input",
    "k_ratio": "k_ratio_input", "global_G": "global_G_input", "global_Tf_C": "global_Tf_input",
    "global_Z": "global_Z_input", "global_mu_cP": "global_mu_input",
    "tb_c": "tb_c_input", "pb_bar": "pb_bar_input",
}


def get_current_node_names():
    names = set()
    for sid in st.session_state.get("seg_ids", []):
        f, t = st.session_state.get(f"from_{sid}"), st.session_state.get(f"to_{sid}")
        if f:
            names.add(f)
        if t:
            names.add(t)
    return sorted(names)


def build_export_payload():
    payload = {"gasnet_version": 1, "global": {}, "segments": [], "nodes": {}}
    for field, key in GLOBAL_FIELD_KEYS.items():
        if key in st.session_state:
            payload["global"][field] = st.session_state[key]
    for sid in st.session_state.get("seg_ids", []):
        seg_data = {"sid": sid}
        for field, prefix in SEGMENT_FIELD_KEYS.items():
            seg_data[field] = st.session_state.get(f"{prefix}{sid}")
        payload["segments"].append(seg_data)
    for n in get_current_node_names():
        payload["nodes"][n] = {
            "role": st.session_state.get(f"noderole_{n}"),
            "flow": st.session_state.get(f"nodeflow_{n}"),
            "pknown": st.session_state.get(f"nodepknown_{n}"),
            "pressure": st.session_state.get(f"nodep_{n}"),
        }
    return payload


def apply_uploaded_network():
    f = st.session_state.get("upload_json_file")
    if f is None:
        return
    try:
        payload = json.loads(f.getvalue().decode("utf-8"))
    except Exception as exc:
        st.session_state["_load_error"] = f"Could not parse file: {exc}"
        st.session_state["_load_success"] = False
        return

    # Clear any leftover per-segment keys from the CURRENT network first, so
    # nothing stale survives if the loaded file has fewer segments.
    for sid in list(st.session_state.get("seg_ids", [])):
        for prefix in SEGMENT_FIELD_KEYS.values():
            st.session_state.pop(f"{prefix}{sid}", None)

    g = payload.get("global", {})
    for field, key in GLOBAL_FIELD_KEYS.items():
        if field in g and g[field] is not None:
            st.session_state[key] = g[field]

    new_ids = []
    for seg_data in payload.get("segments", []):
        sid = seg_data["sid"]
        new_ids.append(sid)
        st.session_state[f"name_{sid}"] = seg_data.get("name") or f"Seg-{sid}"
        st.session_state[f"from_{sid}"] = seg_data.get("from") or ""
        st.session_state[f"to_{sid}"] = seg_data.get("to") or ""
        st.session_state[f"L_{sid}"] = seg_data.get("L") if seg_data.get("L") is not None else 1000.0
        st.session_state[f"D_{sid}"] = seg_data.get("D") if seg_data.get("D") is not None else 150.0
        st.session_state[f"rough_{sid}"] = seg_data.get("rough") if seg_data.get("rough") is not None else 0.046
        st.session_state[f"elevup_{sid}"] = seg_data.get("elevup") if seg_data.get("elevup") is not None else 0.0
        st.session_state[f"elevdn_{sid}"] = seg_data.get("elevdn") if seg_data.get("elevdn") is not None else 0.0
        st.session_state[f"override_{sid}"] = bool(seg_data.get("override"))
        st.session_state[f"G_{sid}"] = seg_data.get("G") if seg_data.get("G") is not None else 0.6
        st.session_state[f"Tf_{sid}"] = seg_data.get("Tf") if seg_data.get("Tf") is not None else 15.0
        st.session_state[f"Z_{sid}"] = seg_data.get("Z") if seg_data.get("Z") is not None else 0.95
        st.session_state[f"mu_{sid}"] = seg_data.get("mu") if seg_data.get("mu") is not None else 0.0110

    st.session_state.seg_ids = new_ids
    st.session_state.seg_counter = max(new_ids) if new_ids else 0

    for n, cfg in payload.get("nodes", {}).items():
        st.session_state[f"noderole_{n}"] = cfg.get("role") or "Passing (no external flow)"
        st.session_state[f"nodeflow_{n}"] = cfg.get("flow") if cfg.get("flow") is not None else 1000.0
        st.session_state[f"nodepknown_{n}"] = bool(cfg.get("pknown"))
        st.session_state[f"nodep_{n}"] = cfg.get("pressure") if cfg.get("pressure") is not None else 50.0

    st.session_state["_load_success"] = True
    st.session_state["_load_error"] = None


_save_load_rendered = False


def render_save_load_section():
    """
    Fill in `save_load_container` (a placeholder positioned at the very top
    of the sidebar). Called right before every st.stop() as well as at the
    true end of the script, so the Save/load controls stay available even
    when the current network has a validation error - guarded so it only
    renders once per run even though several call sites can reach it.
    """
    global _save_load_rendered
    if _save_load_rendered:
        return
    _save_load_rendered = True
    with save_load_container:
        st.header("Save / load network")
        st.caption(
            "Download the current inputs (segments, node settings, global settings) as a "
            "JSON file any time - keeps your work even if this server session ends. "
            "Upload that file later (here, a fresh session, or after a restart) to "
            "restore the network exactly."
        )
        st.download_button(
            "⬇ Download network (JSON)",
            data=json.dumps(build_export_payload(), indent=2),
            file_name="gasnet_network.json",
            mime="application/json",
        )
        st.file_uploader("Load network (.json)", type=["json"], key="upload_json_file")
        st.button(
            "⬆ Load this file into the app",
            on_click=apply_uploaded_network,
            disabled=st.session_state.get("upload_json_file") is None,
        )
        if st.session_state.get("_load_error"):
            st.error(st.session_state["_load_error"])
        if st.session_state.get("_load_success"):
            st.success("Network loaded.")
            st.session_state["_load_success"] = False

# --- sidebar: global settings ------------------------------------------------
st.sidebar.header("Flow equation")
equation = st.sidebar.selectbox(
    "Equation",
    ["General (Colebrook-White / Swamee-Jain)", "Weymouth", "Panhandle A", "Panhandle B"],
    key="eq_select",
)

ff_method = "Colebrook-White"
eff_factor = 1.0
if equation == "General (Colebrook-White / Swamee-Jain)":
    ff_method = st.sidebar.radio(
        "Friction factor method", ["Colebrook-White", "Swamee-Jain"], key="ff_method_radio"
    )
elif equation in ("Panhandle A", "Panhandle B"):
    eff_factor = st.sidebar.number_input(
        "Pipeline efficiency factor E", min_value=0.5, max_value=1.0, value=0.92, step=0.01,
        key="eff_factor_input",
    )
    st.sidebar.caption("Weymouth/Panhandle constants are widely-cited textbook values "
                        "(diameter converted internally to inches). Verify against your "
                        "governing standard for final design.")

k_ratio = st.sidebar.number_input(
    "Specific heat ratio k (for Mach number estimate)", min_value=1.0, max_value=1.7, value=1.30, step=0.01,
    key="k_ratio_input",
)

st.sidebar.header("Default gas properties")
st.sidebar.caption("Used unless a segment overrides them individually. Node flow inputs "
                    "always use these global values (see docstring).")
global_G = st.sidebar.number_input(
    "Specific gravity G (air = 1)", min_value=0.01, value=0.60, step=0.01, key="global_G_input"
)
global_Tf_C = st.sidebar.number_input("Flowing temperature Tf (°C)", value=15.0, step=1.0, key="global_Tf_input")
global_Z = st.sidebar.number_input(
    "Compressibility factor Z", min_value=0.01, max_value=1.5, value=0.95, step=0.01, key="global_Z_input"
)
global_mu_cP = st.sidebar.number_input(
    "Viscosity μ (cP)", min_value=0.0001, value=0.0110, step=0.0005, format="%.4f", key="global_mu_input"
)

st.sidebar.header("Base / standard conditions")
tb_c = st.sidebar.number_input("Base temperature Tb (°C)", value=15.0, step=1.0, key="tb_c_input")
pb_bar = st.sidebar.number_input(
    "Base pressure Pb (bar a)", value=1.01325, step=0.001, format="%.5f", key="pb_bar_input"
)

st.sidebar.header("Segments (pipes)")
st.sidebar.caption("Flow is no longer set here - it's derived from node demands below.")
st.sidebar.button("+ Add segment", on_click=add_segment)

for idx, sid in enumerate(list(st.session_state.seg_ids)):
    label = st.session_state.get(f"name_{sid}", f"Segment {sid}")
    with st.sidebar.expander(f"{idx + 1}. {label} (id {sid})", expanded=False):
        st.text_input("Segment name", key=f"name_{sid}")
        c1, c2 = st.columns(2)
        with c1:
            st.text_input("Upstream node", key=f"from_{sid}")
        with c2:
            st.text_input("Downstream node", key=f"to_{sid}")
        st.number_input("Length L (m)", min_value=0.0, key=f"L_{sid}", step=10.0)
        st.number_input("Internal diameter D (mm)", min_value=0.1, key=f"D_{sid}", step=1.0)
        st.number_input("Absolute roughness ε (mm)", min_value=0.0, key=f"rough_{sid}", step=0.001, format="%.4f")
        c3, c4 = st.columns(2)
        with c3:
            st.number_input("Elevation upstream (m)", key=f"elevup_{sid}", step=1.0)
        with c4:
            st.number_input("Elevation downstream (m)", key=f"elevdn_{sid}", step=1.0)
        st.checkbox("Override global gas properties", key=f"override_{sid}")
        if st.session_state[f"override_{sid}"]:
            st.number_input("Specific gravity G (air = 1)", min_value=0.01, key=f"G_{sid}", step=0.01)
            st.number_input("Flowing temperature Tf (°C)", key=f"Tf_{sid}", step=1.0)
            st.number_input("Compressibility factor Z", min_value=0.01, max_value=1.5, key=f"Z_{sid}", step=0.01)
            st.number_input("Viscosity μ (cP)", min_value=0.0001, key=f"mu_{sid}", step=0.0005, format="%.4f")
        st.button("Remove this segment", key=f"rm_{sid}", on_click=remove_segment, args=(sid,))

# --- assemble segment definitions for calculation ---------------------------
segments = []
for sid in st.session_state.seg_ids:
    override = st.session_state[f"override_{sid}"]
    seg = {
        "id": sid,
        "name": st.session_state[f"name_{sid}"],
        "node_from": st.session_state[f"from_{sid}"],
        "node_to": st.session_state[f"to_{sid}"],
        "L_m": st.session_state[f"L_{sid}"],
        "D_m": mm_to_m(st.session_state[f"D_{sid}"]),
        "rough_m": mm_to_m(st.session_state[f"rough_{sid}"]),
        "elev_up_m": st.session_state[f"elevup_{sid}"],
        "elev_dn_m": st.session_state[f"elevdn_{sid}"],
        "G": st.session_state[f"G_{sid}"] if override else global_G,
        "Tf_K": c_to_k(st.session_state[f"Tf_{sid}"] if override else global_Tf_C),
        "Z": st.session_state[f"Z_{sid}"] if override else global_Z,
        "mu_pas": cp_to_pas(st.session_state[f"mu_{sid}"] if override else global_mu_cP),
        "E": eff_factor,
    }
    segments.append(seg)

# --- main page ----------------------------------------------------------------
if not segments:
    st.info("Add at least one pipe segment from the sidebar (\"+ Add segment\") to begin.")
    render_save_load_section()
    st.stop()

children, roots, topo_errors, bad_nodes, bad_seg_ids = build_topology(segments)

st.subheader("Network topology")

if topo_errors:
    st.error(
        "Network topology problem(s) — fix these before results can be computed "
        "(nodes/segments involved are highlighted red below):\n"
        + "\n".join(f"- {e}" for e in topo_errors)
    )
    st.graphviz_chart(build_network_dot(segments, roots, {}, {}, {}, {}, bad_nodes, bad_seg_ids))
    render_save_load_section()
    st.stop()

st.code(build_text_tree(children, roots), language=None)

# --- node settings (role / flow / known pressure) ---------------------------
st.subheader("Node settings")
st.caption(
    "Every node referenced by a segment above appears here. Mark it Inlet/Outlet and "
    "give the flow entering/leaving there, or leave it Passing (junction, no external "
    "flow). A root node's flow (marked *root*) is always derived from downstream "
    "demand, not entered. Exactly one node in each connected network must have a "
    "known pressure - it does not have to be the inlet or a root; every other node "
    "is solved from it, including working backward to size an inlet pressure."
)

node_names = sorted({s["node_from"] for s in segments} | {s["node_to"] for s in segments})

for n in node_names:
    st.session_state.setdefault(
        f"noderole_{n}", "Inlet (supply)" if n in roots else "Passing (no external flow)"
    )
    st.session_state.setdefault(f"nodeflow_{n}", 1000.0)
    st.session_state.setdefault(f"nodepknown_{n}", False)
    st.session_state.setdefault(f"nodep_{n}", 50.0)

hdr = st.columns([1.4, 1.8, 1.3, 1.0, 1.3])
hdr[0].markdown("**Node**")
hdr[1].markdown("**Role**")
hdr[2].markdown("**Flow (Sm3/h)**")
hdr[3].markdown("**P known?**")
hdr[4].markdown("**Pressure (bar a)**")

node_settings = {}
for n in node_names:
    is_root = n in roots
    cols = st.columns([1.4, 1.8, 1.3, 1.0, 1.3])
    cols[0].markdown(f"**{n}**" + (" *(root)*" if is_root else ""))
    role = cols[1].selectbox(
        "Role", ["Passing (no external flow)", "Inlet (supply)", "Outlet (delivery)"],
        key=f"noderole_{n}", label_visibility="collapsed",
    )
    if is_root:
        cols[2].caption("derived below")
        flow_val = None
    elif role.startswith("Passing"):
        cols[2].caption("—")
        flow_val = 0.0
    else:
        flow_val = cols[2].number_input(
            "Flow (Sm3/h)", min_value=0.0, key=f"nodeflow_{n}", label_visibility="collapsed"
        )
    p_known = cols[3].checkbox("P known", key=f"nodepknown_{n}", label_visibility="collapsed")
    if p_known:
        p_val = cols[4].number_input(
            "Pressure (bar a)", min_value=0.001, key=f"nodep_{n}", label_visibility="collapsed"
        )
    else:
        cols[4].caption("solved")
        p_val = None
    node_settings[n] = {"role": role, "flow_sm3h": flow_val, "pressure_known": p_known, "pressure_bar": p_val}

# --- mass balance: node demands -> per-segment flow --------------------------
local_demand, rho_base = compute_node_local_demand(
    node_names, node_settings, roots, global_G, c_to_k(tb_c), bar_to_pa(pb_bar)
)
seg_flow, subtree_demand = compute_segment_flows(children, roots, local_demand)

# --- validate exactly one pressure anchor per connected network -------------
comp_of = compute_components(children, roots)
anchors_per_root = defaultdict(list)
for n in node_names:
    if node_settings[n]["pressure_known"]:
        anchors_per_root[comp_of.get(n, n)].append(n)

pressure_errors = []
for r in roots:
    count = len(anchors_per_root.get(r, []))
    if count == 0:
        pressure_errors.append(f"Network rooted at '{r}' has no node with a known pressure - pick one.")
    elif count > 1:
        pressure_errors.append(
            f"Network rooted at '{r}' has {count} nodes with a known pressure "
            f"({', '.join(anchors_per_root[r])}) - only one is allowed per network."
        )

if pressure_errors:
    st.error("Pressure boundary condition problem(s):\n" + "\n".join(f"- {e}" for e in pressure_errors))
    st.graphviz_chart(build_network_dot(segments, roots, node_settings, {}, seg_flow, {}, set(), set()))
    render_save_load_section()
    st.stop()

# --- solve pressures ----------------------------------------------------------
node_pressure_pa, seg_results = solve_network_pressures(
    segments, node_names, node_settings, seg_flow, equation, ff_method, k_ratio
)
node_pressure_bar = {n: pa_to_bar(p) for n, p in node_pressure_pa.items() if p is not None}

st.graphviz_chart(build_network_dot(segments, roots, node_settings, node_pressure_bar, seg_flow, seg_results, set(), set()))

# --- segment results table ----------------------------------------------------
rows = []
for seg in segments:
    res = seg_results[seg["id"]]
    signed_mdot = seg_flow.get(seg["id"], 0.0)
    qb_sm3h = (abs(signed_mdot) / rho_base) * 3600.0 if rho_base else 0.0
    p_up = res["P_up"]
    p_down = res["P_down"]
    rows.append(
        {
            "Segment": seg["name"],
            "From": seg["node_from"],
            "To": seg["node_to"],
            "Flow dir": "reversed (to→from)" if signed_mdot < 0 else "from→to",
            "Qb (Sm3/h)": qb_sm3h,
            "mdot (kg/s)": res["mdot"],
            "Re (-)": res["re"],
            "f Darcy (-)": res["f"],
            "P_upstream (bar a)": pa_to_bar(p_up) if p_up is not None else None,
            "P_downstream (bar a)": pa_to_bar(p_down) if p_down is not None else None,
            "dP (bar)": pa_to_bar(p_up - p_down) if (p_up is not None and p_down is not None) else None,
            "v_upstream (m/s)": res["v_up"],
            "v_downstream (m/s)": res["v_down"],
            "Mach @ downstream (-)": res["mach_down"],
            "Status": "OK" if res["error"] is None else res["error"],
        }
    )

df = pd.DataFrame(rows)

decimals = {
    "Qb (Sm3/h)": 1,
    "mdot (kg/s)": 4,
    "Re (-)": 0,
    "f Darcy (-)": 5,
    "P_upstream (bar a)": 3,
    "P_downstream (bar a)": 3,
    "dP (bar)": 3,
    "v_upstream (m/s)": 2,
    "v_downstream (m/s)": 2,
    "Mach @ downstream (-)": 3,
}
df_display = df.copy()
for col, nd in decimals.items():
    # coerce to numeric first: unsolved segments store None in these columns,
    # and pandas' Series.round() raises on a plain None (no __round__)
    df_display[col] = pd.to_numeric(df_display[col], errors="coerce").round(nd)

st.subheader("Segment results")
st.dataframe(df_display, width="stretch")

seg_errors = [(seg["name"], seg_results[seg["id"]]["error"]) for seg in segments if seg_results[seg["id"]]["error"]]
if seg_errors:
    st.error(
        "One or more segments could not be solved:\n"
        + "\n".join(f"- {name}: {err}" for name, err in seg_errors)
    )

# --- node results table -------------------------------------------------------
node_rows = []
for n in node_names:
    cfg = node_settings[n]
    p_pa = node_pressure_pa.get(n)
    if n in roots:
        flow_sm3h = (subtree_demand.get(n, 0.0) / rho_base) * 3600.0 if rho_base else 0.0
        flow_label = f"{flow_sm3h:+.1f} (derived)"
    elif cfg["role"].startswith("Passing"):
        flow_label = "0 (passing)"
    else:
        sign = "-" if cfg["role"].startswith("Outlet") else "+"
        flow_label = f"{sign}{cfg['flow_sm3h']:.1f} (input)"
    if p_pa is not None:
        pressure_label = f"{pa_to_bar(p_pa):.3f}" + (" (anchor)" if cfg["pressure_known"] else "")
    else:
        pressure_label = "unsolved"
    node_rows.append(
        {
            "Node": n,
            "Role": cfg["role"].split(" (")[0] + (" (root)" if n in roots else ""),
            "Flow (Sm3/h)": flow_label,
            "Pressure (bar a)": pressure_label,
        }
    )

st.subheader("Node results")
st.caption(
    "'+' = net injection (supply) into the network at that node, '-' = net "
    "withdrawal (delivery). A root's flow is always derived from downstream demand."
)
st.dataframe(pd.DataFrame(node_rows), width="stretch")

with st.expander("Equation & assumptions reference"):
    st.markdown(
        """
- **General equation**: `P_up^2 - e^s*P_down^2 = (16 f Le Z R Tf) / (pi^2 D^5 M) * mdot^2`,
  friction factor `f` from Colebrook-White (iterated) or Swamee-Jain (explicit). "up"/"down"
  are the PHYSICAL flow direction, which can be the reverse of a pipe's drawn
  from→to direction if a mid-network inlet forces gas back toward the source.
- **Weymouth / Panhandle A / Panhandle B**: same general equation, with `f` derived from a
  transmission-factor correlation (`F = 2/sqrt(f)`) specific to each method.
- **Elevation correction**: `s = 2 g M (H_down - H_up) / (Z R Tf)`, effective length
  `Le = L` if `s ≈ 0`, else `Le = L (e^s - 1) / s`.
- **Reynolds number**: `Re = 4 mdot / (pi D mu)`.
- **Mach number**: estimated at the (higher-velocity) physically-downstream end using
  an ideal-gas sonic velocity `c = sqrt(k Z R Tf / M)`.
- **Node-based flow**: each segment's flow is the net demand of everything downstream
  of it (summed from the leaves inward), not a direct input - so a mid-network inlet
  can make a segment's flow run in reverse, which is detected and flagged.
- **Node-based pressure**: pick exactly one node per network to hold a known pressure;
  every other node - including the inlet - is solved from it, forward toward the
  known node's downstream side and backward (always solvable) toward its upstream side.
- All flow is assumed **steady, isothermal**, with **constant Z and mu per segment**.
- **Branching tree networks** are supported: a node may feed multiple downstream
  segments (a split). A node fed by more than one segment (a merge) or a cycle is
  rejected — that needs an iterative flow-balancing solver, out of scope here.
        """
    )

render_save_load_section()
