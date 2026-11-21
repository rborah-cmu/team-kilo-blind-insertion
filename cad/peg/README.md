# Peg CAD

Onshape export (`urdf/assembly_1.urdf`, `meshes/Part_1.stl`). Units: metres.
Origin at the joint between the square top and the shaft; +z points up toward
the top.

| Feature | Size |
| --- | --- |
| Square top (gripped) | 25 × 25 mm, 19.05 mm tall (z = 0 to +19.05 mm) |
| Shaft | 25 mm diameter cylinder, z = 0 to −37.1 mm |
| Chamfer | 1 mm × 45° at the tip; tip face 23 mm diameter at z = −38.1 mm |
| Total length | 57.15 mm |
| Volume | 30.6 cm³ (about 38 g if printed solid in PLA) |

## Known issues with the export

- **Mass and inertia are wrong.** The URDF mass (3.06e-5 kg) equals the
  volume in m³, so Onshape used a density of 1 kg/m³. Weigh the printed peg and
  scale the mass and inertia to match before using them in simulation.
- **No collision geometry.** The URDF has only a visual mesh. For MuJoCo, build
  the peg from two convex pieces instead of the whole mesh (whose convex hull
  would fill in the step between the top and the shaft):
  - a box, 12.5 × 12.5 × 9.525 mm half-sizes, centred at z = +9.525 mm;
  - the shaft as a cylinder (radius 12.5 mm), or the lower part of the STL
    (z < 0) as its own mesh to keep the chamfer exact.
