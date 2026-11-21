"""
run_thinwall_fem_fig7fig8.py
==============================
Reproduces Ruvo et al.'s Fig 7 (transient field, 12 snapshots) and Fig 8
(Y-Z/X-Z sections, 4 of those same 12 snapshots) using the REAL FEM
solver (gamma_afsd.py) at 600rpm/10MPa/mu=0.3/60s dwell/4 layers -- the
exact process condition both papers describe for these two figures.

Reuses run_cylinder_fem.py's PROVEN domain_mgr/heat_solve_mgr pattern
verbatim (mask_mesh_by_birth, write_snapshot_vtk) -- only the geometry
(straight-line thin-wall instead of circular cylinder) and the snapshot
schedule change.

Fig 8's 4 timestamps (60,240,420,600s) are already a SUBSET of Fig 7's
12 (10,60,110 / 190,240,290 / 370,420,470 / 550,600,650) -- one run
covers both figures. Fig 8's sections are just a ParaView Slice filter
on 4 of these same snapshots, not a separate simulation.

IMPORTANT: run against a PATCHED copy of thinwall.k with the process
line changed to 10.0 15.0 5.0 62.831853 1.67 0.3 0.90 (P=10MPa,
w=600rpm, mu=0.3) -- the original thinwall.k is the Hu et al.
validation case (400rpm/5MPa/mu=0.2) and must NOT be overwritten.
Create the patched copy first:

    cp thinwall.k thinwall_600rpm_10MPa_mu03.k
    sed -i 's/^5.0 15.0 5.0 41.888 1.67 0.2 0.90$/10.0 15.0 5.0 62.831853 1.67 0.3 0.90/' \\
        thinwall_600rpm_10MPa_mu03.k
    grep -A1 P_Rs_Rp_w_v_mu_slip thinwall_600rpm_10MPa_mu03.k   # verify

Usage
-----
    python run_thinwall_fem_fig7fig8.py \\
        --deck thinwall_600rpm_10MPa_mu03.k --out-dir fem_fig7fig8_results
"""

from __future__ import annotations
import argparse
import sys
import time
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
BASE = Path("/projects/mma/apurba/compare_afsd_thermal")
SOLVER_DIR = BASE / "AFSD-Code" / "Tutorial"
sys.path.insert(0, str(SOLVER_DIR))
sys.path.insert(0, str(_HERE))

from gamma_afsd import domain_mgr, heat_solve_mgr

# Reuse the EXACT, already-proven functions from run_cylinder_fem.py --
# imported directly rather than re-typed, so any fix made there is
# automatically picked up here too.
from run_cylinder_fem import mask_mesh_by_birth, write_snapshot_vtk


# Fig 7's 12 timestamps exactly as captioned. Fig 8's 4 (60,240,420,600)
# are already a subset -- extract those from this same run in ParaView
# via a Slice filter, no second simulation needed.
TARGET_TIMES = {
    10:  "L1_early",  60:  "L1_mid",  110: "L1_late",
    190: "L2_early", 240: "L2_mid", 290: "L2_late",
    370: "L3_early", 420: "L3_mid", 470: "L3_late",
    550: "L4_early", 600: "L4_mid", 650: "L4_late",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--deck", type=str, default="thinwall_600rpm_10MPa_mu03.k")
    ap.add_argument("--out-dir", type=str, default="fem_fig7fig8_results")
    ap.add_argument("--omega-rpm", type=float, default=None,
                    help="override; leave unset to use the value baked into --deck")
    ap.add_argument("--pressure-mpa", type=float, default=None,
                    help="override; leave unset to use the value baked into --deck")
    ap.add_argument("--log-every-s", type=float, default=20.0)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  AFSD Thin-wall FEM run for Fig 7/8 reproduction (gamma_afsd.py)")
    print(f"  Target: 12 snapshots at t={sorted(TARGET_TIMES.keys())}")
    print("=" * 60)

    t0 = time.perf_counter()
    domain = domain_mgr(filename=args.deck)
    print(f"[INIT] domain loaded: {len(domain.nodes):,} nodes, "
         f"{len(domain.elements):,} elements  ({time.perf_counter()-t0:.1f}s)")

    if args.omega_rpm is not None:
        import math
        domain.w = args.omega_rpm * 2 * math.pi / 60.0
        print(f"[INIT] overriding omega -> {args.omega_rpm} rpm ({domain.w:.4f} rad/s)")
    if args.pressure_mpa is not None:
        domain.pressure = args.pressure_mpa
        print(f"[INIT] overriding pressure -> {args.pressure_mpa} MPa")

    print(f"[INIT] pressure={domain.pressure} MPa  w={domain.w:.4f} rad/s  "
         f"v={domain.v:.4f} mm/s  dt={domain.dt:.4f}s  end_time={domain.end_time:.1f}s")
    print(f"[CHECK] omega->rpm back-conversion: {domain.w*60/(2*3.14159265):.1f} rpm "
         f"(should read 600.0)")
    print(f"[CHECK] pressure (should read 10.0): {domain.pressure} MPa")

    if domain.end_time < max(TARGET_TIMES.keys()) + 1.0:
        print(f"[ERROR] deck's end_time ({domain.end_time:.1f}s) is shorter than the "
             f"last target snapshot ({max(TARGET_TIMES.keys())}s) -- deck may not "
             f"actually cover 4 layers at these process settings. Aborting before "
             f"wasting GPU time.")
        sys.exit(1)

    heat_solver = heat_solve_mgr(domain)

    n_nodes = len(domain.nodes)
    elements_cpu = domain.elements
    element_mat_cpu = np.asarray(domain.element_mat) if hasattr(domain, "element_mat") else None
    element_birth_cpu = np.asarray(domain.element_birth) if hasattr(domain, "element_birth") else None
    nodes_cpu = domain.nodes.get() if hasattr(domain.nodes, "get") else np.asarray(domain.nodes)

    remaining_times = sorted(TARGET_TIMES.keys())
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

        while remaining_times and t_now >= remaining_times[0] - 0.5 * domain.dt:
            t_target = remaining_times.pop(0)
            label = TARGET_TIMES[t_target]
            T_np = heat_solver.temperature.get() if hasattr(heat_solver.temperature, "get") \
                else np.asarray(heat_solver.temperature)

            np.savez_compressed(
                out_dir / f"fem_thinwall_t{t_target:04d}s_{label}.npz",
                nodes_mm=nodes_cpu, T=T_np[:n_nodes], t=t_now)

            if element_mat_cpu is not None and element_birth_cpu is not None:
                nodes_f, elems_f, mat_f, idx_map = mask_mesh_by_birth(
                    nodes_cpu, elements_cpu, element_mat_cpu,
                    element_birth_cpu, t_now)
                T_f = T_np[:n_nodes][idx_map]
                write_snapshot_vtk(
                    out_dir / f"fem_thinwall_t{t_target:04d}s_{label}.vtk",
                    nodes_f, elems_f, mat_f, T_f, t_now)
            else:
                print("  [WARN] birth-masking unavailable -- writing full mesh")
                write_snapshot_vtk(
                    out_dir / f"fem_thinwall_t{t_target:04d}s_{label}.vtk",
                    nodes_cpu, elements_cpu, element_mat_cpu,
                    T_np[:n_nodes], t_now)

            print(f"  [SNAPSHOT] t={t_target}s ({label})  actual_t={t_now:.1f}s  "
                 f"Tmax={T_np[:n_nodes].max():.1f}K  -> {out_dir}/", flush=True)

        if not remaining_times:
            break

    print(f"\n[DONE] total wall time: {time.perf_counter()-w0:.1f}s")
    print(f"       {12-len(remaining_times)}/12 snapshots written to {out_dir}/")
    if remaining_times:
        print(f"       [WARN] missed targets: {remaining_times} -- deck's end_time "
             f"may have been reached before these times.")


if __name__ == "__main__":
    main()
