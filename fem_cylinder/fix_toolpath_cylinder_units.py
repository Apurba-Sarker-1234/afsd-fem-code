"""
fix_toolpath_cylinder_units.py
================================
Fixes the confirmed unit mismatch in toolpath_cylinder.crs: the file's
x/y/z coordinates are in METERS (e.g., x=0.0450 for the 45mm centerline
radius), while the mesh (cylinder.k's *NODE section) is in MILLIMETERS
(confirmed via check_fem_nodes.py: X/Y range [-66,66], Z range [-5,60]).

This 1000x mismatch means the tool's true position, as seen by the
mesh's own distance calculations, collapses to near the origin for the
entire simulation -- explaining both symptoms found: the hollow center
running nearly as hot as the wall itself, and hot nodes spread
symmetrically about (0,0) instead of tracing the tool's actual
45mm-radius circular path.

Multiplies the x, y, z columns by 1000. Time and state columns are
left unchanged.
"""
import numpy as np

INPUT_FILE = "toolpath_cylinder.crs"
OUTPUT_FILE = "toolpath_cylinder_mm.crs"

data = np.loadtxt(INPUT_FILE)
print(f"Loaded {len(data)} waypoints from {INPUT_FILE}")
print(f"Original x range: {data[:,1].min():.4f} to {data[:,1].max():.4f}")
print(f"Original y range: {data[:,2].min():.4f} to {data[:,2].max():.4f}")
print(f"Original z range: {data[:,3].min():.4f} to {data[:,3].max():.4f}")

data[:, 1] *= 1000.0  # x: m -> mm
data[:, 2] *= 1000.0  # y: m -> mm
data[:, 3] *= 1000.0  # z: m -> mm

print(f"\nCorrected x range: {data[:,1].min():.2f} to {data[:,1].max():.2f}")
print(f"Corrected y range: {data[:,2].min():.2f} to {data[:,2].max():.2f}")
print(f"Corrected z range: {data[:,3].min():.2f} to {data[:,3].max():.2f}")
print(f"(should now match the mesh's own ranges: X/Y~[-66,66], Z~[-5,60])")

with open(OUTPUT_FILE, "w") as f:
    for row in data:
        f.write(f"{row[0]:<12.4f}{row[1]:<10.4f}{row[2]:<10.4f}{row[3]:<10.4f}{int(row[4])}\n")

print(f"\nWritten: {OUTPUT_FILE}")
