# Peg and base CAD

Onshape export of the peg seated in the base. Units: metres.

- `meshes/Part_1.stl`: the peg (identical to `cad/peg/meshes/Part_1.stl`;
  see `cad/peg/README.md`).
- `meshes/Part_1_1.stl`: the base.
- `urdf/assembly_1.urdf`: base fixed to `root`; peg on a prismatic joint
  (`slider_1`, along z) over the hole, from the Onshape mate.

## Base

Origin at the centre of the bottom face; +z up.

| Feature | Size |
| --- | --- |
| Block | 126.7 × 169.0 × 50.8 mm (x × y × z), solid |
| Hole | 25.4 mm (1 in) diameter, straight through the full 50.8 mm height |
| Hole chamfer | 1 mm × 45° at the top edge (27.4 mm diameter at the top face) |
| Hole centre | (+2.62, +3.02) mm from the block's centre, i.e. 4.0 mm off-centre |
| Volume | 1062 cm³ (about 1.3 kg if printed solid in PLA) |

## Fit with the peg

Peg shaft 25.0 mm, hole 25.4 mm: **0.4 mm diametral clearance** (0.2 mm per
side), before printing error. FDM-printed holes usually come out undersized by
a few tenths of a millimetre, so print a short test section of the hole first
and check the peg goes in.

## Known issues with the export

- Same as the peg: masses and inertias assume a density of 1 kg/m³ (the base's
  "mass" 0.00106 equals its volume in m³), and there is no collision geometry.
- For MuJoCo, build the base as a plate with a circular through-hole (mjfit) at
  the offset above, not from this mesh: a mesh's convex hull would fill the hole.
