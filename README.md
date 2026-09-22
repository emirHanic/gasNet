# GasNET – Compressible Gas Pipeline Network Calculator

A Streamlit web application for computing steady-state pressure distribution and flow in branching-tree gas pipeline networks using isothermal, compressible-flow equations.

## Features

### Flow Calculations
- **Node-based flow specification** – flow is defined per node (Inlet/Outlet/Passing), not per pipe
- **Automatic segment flow derivation** – segment flows are derived from node demands via mass balance
- **Support for flow reversal** – mid-network inlets can force gas to flow backward through segments; this is detected and flagged automatically

### Pressure Solving
- **Single pressure anchor** – specify pressure at exactly one node anywhere in the network
- **Forward and backward solving** – pressures propagate outward from the anchor:
  - Forward: toward downstream nodes
  - Backward: toward upstream nodes (always solvable, no iteration required)
- **Inverse pressure queries** – ask "what inlet pressure do I need to guarantee X bar at this delivery point?" directly

### Flow Equations
1. **General Isothermal Equation** – most rigorous option
   - Darcy friction factor from Colebrook-White (iterated) or Swamee-Jain (explicit)
   - Exact for steady, isothermal, single-phase compressible gas flow
   - Unit-consistent in SI throughout

2. **Empirical Equations** (transmission factor correlations)
   - Weymouth
   - Panhandle A
   - Panhandle B
   - Note: Originally fit in imperial units; cross-check against your governing standard

### Network Topology
- **Branching tree only** – a node may feed multiple downstream segments (splits are OK)
- **No merges or cycles** – a node fed by more than one segment or cycles are rejected with clear error messages
- Topology validation and visualization

### Physical Effects
- **Elevation correction** – classical exponential static-head term applied per segment
- **Compressibility effects** – constant Z factor per segment
- **Viscosity effects** – variable friction factor based on Reynolds number
- **Mach number estimation** – at the physically-downstream end using ideal-gas sonic velocity

### Advanced Features
- **Per-segment gas property overrides** – use global defaults or override G, Z, Tf, μ per segment
- **Save/load networks** – download current inputs as JSON, restore later
- **Network visualization** – Graphviz rendering of topology with pressures and pressure drops
- **Detailed results tables** – segment and node results with all hydraulic parameters

## Assumptions

- Steady-state, **isothermal** flow
- **Compressible gas** with constant Z per segment
- Kinetic-energy (velocity-head) term neglected (standard pipeline practice)
- All pressures are **absolute** (not gauge)
- Single-phase gas flow

## Installation

```bash
pip install streamlit pandas
