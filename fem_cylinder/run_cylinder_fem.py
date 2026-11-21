"""
run_cylinder_fem.py
====================
Runs the actual gamma_afsd.py FEM thermal solver against the generated
cylinder.k deck, following the EXACT domain_mgr/heat_solve_mgr usage
pattern already proven working in your own generate_fem_snapshots.py
(domain_mgr(filename=...) -> patch pressure/w if needed -> heat_solve_mgr(domain)
-> loop heat_solver.time_integration()).

I cannot run gamma_afsd.py myself (needs cupy/GPU, only exists on
TinkerCliffs) -- this script is built from reading the actual source, not
executed end-to-end. Sanity-check the first few printed lines (node/element
counts should match what afsd_generate_cylinder_mesh.py reported) before
trusting a long run.

Snapshots the full nodal temperature field at the end of each layer's dwell
(same convention as generate_fem_snapshots.py) to .npz, AND writes an
unstructured VTK per snapshot so you can compare directly against the
B-spline cylinder run's fields in ParaView.

Usage
-----
Run from the SAME directory as cylinder.k / toolpath_cylinder.crs /
0_properties/ (same convention thinwall.k relies on -- relative paths):

    python run_cylinder_fem.py \\
        --deck cylinder.k --out-dir fem_cylinder_results \\
        --n-layers 20 --dwell-s 60 --traverse-mm-min 100 \\
        --omega-rpm 600 --pressure-mpa 10
"""

from __future__ import annotations
import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
BASE = Path("/projects/mma/apurba/compare_afsd_thermal")
SOLVER_DIR = BASE / "AFSD-Code" / "Tutorial"
sys.path.insert(0, str(SOLVER_DIR))

import cupy as cp
from gamma_afsd import domain_mgr, heat_solve_mgr


def layer_end_times(n_layers, radius_m, traverse_mm_min, dwell_s):
    v_mm_s = traverse_mm_min / 60.0
    circumference_mm = 2 * math.pi * radius_m * 1000.0
    t_dep = circumference_mm / v_mm_s
    t = 0.0
    ends = []
    for i in range(n_layers):
        t += t_dep + dwell_s
        ends.append(t)
    return ends, t_dep


def build_snapshot_schedule(n_layers, t_dep, dwell_s, samples_per_layer=4):
    """
    Multiple snapshot times PER LAYER (early/mid/late within the deposit
    sweep, plus end-of-dwell), not just layer-end. A single per-layer
    snapshot only ever captures the state AFTER a full 360-degree
    revolution has already completed and had time to conduct/equilibrate
    -- which is why an angular temperature scan came back perfectly flat
    (verified: Tmean/Tmax identical across all 12 angular bins at layer 14).
    Mid-revolution snapshots are needed to actually see the moving,
    localized heat source, same reasoning as the paper's own multi-snapshot
    Fig.7.

    Returns list of (layer_idx, sample_idx, target_time, label) sorted by time.
    """
    schedule = []
    t_cursor = 0.0
    for li in range(n_layers):
        layer_start = t_cursor
        fracs = np.linspace(0.15, 0.95, samples_per_layer)  # spread across the deposit sweep
        for si, f in enumerate(fracs):
            schedule.append((li, si, layer_start + f * t_dep, f"L{li+1}_dep_s{si+1}"))
        t_cursor += t_dep
        schedule.append((li, samples_per_layer, t_cursor + dwell_s, f"L{li+1}_dwell_end"))
        t_cursor += dwell_s
    schedule.sort(key=lambda x: x[2])
    return schedule


def mask_mesh_by_birth(nodes, elements, element_mat, element_birth, t_now):
    """
    Filter to only elements actually deposited by t_now (birth <= t_now),
    remap node indices to a compact active-only set. Without this, every
    snapshot dumps the FULL final-height mesh regardless of build progress
    -- only color changes between frames, geometry never grows. This is
    what makes "the cylinder being built" visualization actually show
    growth, not just a static shape changing color.
    """
    active_el_mask = element_birth <= t_now
    elements_active = elements[active_el_mask]
    mat_active = element_mat[active_el_mask]

    used_nodes = np.unique(elements_active.ravel())
    remap = -np.ones(len(nodes), dtype=np.int64)
    remap[used_nodes] = np.arange(len(used_nodes))

    nodes_out = nodes[used_nodes]
    elements_out = remap[elements_active]
    T_index_map = used_nodes  # to slice the matching temperature subset

    return nodes_out, elements_out, mat_active, T_index_map


def write_snapshot_vtk(path: Path, nodes: np.ndarray, elements: np.ndarray,
                       element_mat: np.ndarray, T: np.ndarray, t_now: float):
    """elements: 0-based (n_e,8). T: (n_n,) nodal temperature.
    nodes/elements/element_mat/T here are assumed ALREADY filtered to the
    active-at-t_now subset by the caller (via mask_mesh_by_birth)."""
    n_pts = len(nodes)
    n_cells = len(elements)
    with open(path, "w") as f:
        f.write("# vtk DataFile Version 3.0\n")
        f.write(f"AFSD cylinder FEM T field t={t_now:.1f}s\n")
        f.write("ASCII\nDATASET UNSTRUCTURED_GRID\n")
        f.write(f"POINTS {n_pts} float\n")
        for x, y, z in nodes:
            f.write(f"{x*1e-3:.6e} {y*1e-3:.6e} {z*1e-3:.6e}\n")  # mm -> m
        f.write(f"CELLS {n_cells} {n_cells*9}\n")
        for e in elements:
            f.write("8 " + " ".join(str(int(v)) for v in e) + "\n")
        f.write(f"CELL_TYPES {n_cells}\n")
        for _ in range(n_cells):
            f.write("12\n")
        f.write(f"CELL_DATA {n_cells}\n")
        f.write("SCALARS material int 1\nLOOKUP_TABLE default\n")
        for m in element_mat:
            f.write(f"{int(m)}\n")
        f.write(f"POINT_DATA {n_pts}\n")
        f.write("SCALARS Temperature float 1\nLOOKUP_TABLE default\n")
        for v in T:
            f.write(f"{v:.6f}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--deck", type=str, default="cylinder.k")
    ap.add_argument("--out-dir", type=str, default="fem_cylinder_results")
    ap.add_argument("--n-layers", type=int, default=20)
    ap.add_argument("--radius", type=float, default=0.045, help="m, must match mesh generator")
    ap.add_argument("--traverse-mm-min", type=float, default=100.0)
    ap.add_argument("--dwell-s", type=float, default=60.0)
    ap.add_argument("--omega-rpm", type=float, default=None,
                    help="override; if omitted, uses value baked into the .k deck")
    ap.add_argument("--pressure-mpa", type=float, default=None,
                    help="override; if omitted, uses value baked into the .k deck")
    ap.add_argument("--log-every-s", type=float, default=30.0)
    ap.add_argument("--samples-per-layer", type=int, default=4,
                    help="intra-layer snapshots (mid-revolution), plus 1 dwell-end "
                         "snapshot, per layer. >1 needed to see the moving heat "
                         "source -- a single layer-end-only snapshot always shows "
                         "post-revolution equilibrated (angularly flat) state.")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  AFSD Cylinder FEM run (gamma_afsd.py)")
    print("=" * 60)

    t0 = time.perf_counter()
    domain = domain_mgr(filename=args.deck)
    print(f"[INIT] domain loaded: {len(domain.nodes):,} nodes, "
         f"{len(domain.elements):,} elements  ({time.perf_counter()-t0:.1f}s)")

    if args.omega_rpm is not None:
        domain.w = args.omega_rpm * 2 * math.pi / 60.0
        print(f"[INIT] overriding omega -> {args.omega_rpm} rpm ({domain.w:.4f} rad/s)")
    if args.pressure_mpa is not None:
        domain.pressure = args.pressure_mpa
        print(f"[INIT] overriding pressure -> {args.pressure_mpa} MPa")

    print(f"[INIT] pressure={domain.pressure} MPa  w={domain.w:.4f} rad/s  "
         f"v={domain.v:.4f} mm/s  dt={domain.dt:.4f}s  end_time={domain.end_time:.1f}s")

    heat_solver = heat_solve_mgr(domain)

    l_ends, t_dep = layer_end_times(args.n_layers, args.radius,
                                    args.traverse_mm_min, args.dwell_s)
    schedule = build_snapshot_schedule(args.n_layers, t_dep, args.dwell_s,
                                       samples_per_layer=args.samples_per_layer)
    print(f"[INIT] layer end times (s): {[f'{t:.0f}' for t in l_ends]}")
    print(f"[INIT] {len(schedule)} total snapshots scheduled "
         f"({args.samples_per_layer} intra-layer + 1 dwell-end per layer)")

    n_nodes = len(domain.nodes)
    elements_cpu = domain.elements  # already numpy, 0-based node indices
    element_mat_cpu = np.asarray(domain.element_mat) if hasattr(domain, "element_mat") else None
    element_birth_cpu = np.asarray(domain.element_birth) if hasattr(domain, "element_birth") else None
    nodes_cpu = domain.nodes.get() if hasattr(domain.nodes, "get") else np.asarray(domain.nodes)

    snapped_idx = 0  # pointer into schedule (sorted by time)
    last_log = 0.0
    w0 = time.perf_counter()

    while domain.current_time < domain.end_time - 0.5 * domain.dt:
        heat_solver.time_integration()
        t_now = float(domain.current_time)

        if t_now - last_log >= args.log_every_s:
            T_now = heat_solver.temperature.get() if hasattr(heat_solver.temperature, "get") \
                else np.asarray(heat_solver.temperature)
            print(f"  t={t_now:8.1f}s  Tmax={T_now.max():7.1f}K  "
                 f"wall={time.perf_counter()-w0:.0f}s")
            last_log = t_now

        while snapped_idx < len(schedule) and t_now >= schedule[snapped_idx][2] - 0.5 * domain.dt:
            li, si, t_target, label = schedule[snapped_idx]
            T_np = heat_solver.temperature.get() if hasattr(heat_solver.temperature, "get") \
                else np.asarray(heat_solver.temperature)
            np.savez_compressed(
                out_dir / f"fem_cylinder_{label}.npz",
                nodes_mm=nodes_cpu, T=T_np[:n_nodes], t=t_now)
            if element_mat_cpu is not None and element_birth_cpu is not None:
                nodes_f, elems_f, mat_f, idx_map = mask_mesh_by_birth(
                    nodes_cpu, elements_cpu, element_mat_cpu,
                    element_birth_cpu, t_now)
                T_f = T_np[:n_nodes][idx_map]
                write_snapshot_vtk(
                    out_dir / f"fem_cylinder_{label}_t{t_now:06.1f}s.vtk",
                    nodes_f, elems_f, mat_f, T_f, t_now)
            elif element_mat_cpu is not None:
                print("  [WARN] domain.element_birth not found -- writing "
                     "UNMASKED full-height mesh (geometry won't show growth)")
                write_snapshot_vtk(
                    out_dir / f"fem_cylinder_{label}_t{t_now:06.1f}s.vtk",
                    nodes_cpu, elements_cpu, element_mat_cpu,
                    T_np[:n_nodes], t_now)
            print(f"  [SNAPSHOT] {label}  t={t_now:.1f}s  "
                 f"Tmax={T_np[:n_nodes].max():.1f}K  -> {out_dir}/")
            snapped_idx += 1

        if snapped_idx >= len(schedule):
            break

    print(f"\n[DONE] total wall time: {time.perf_counter()-w0:.1f}s")
    print(f"       {snapped_idx}/{len(schedule)} snapshots written to {out_dir}/")


if __name__ == "__main__":
    main()
