# Plate search

## I/O

- Inputs: none
- Outputs: estimate of the plate center (x, y) and rotation angle theta

## Steps

1. Explore the space with the peg in hand, starting with a snake-shaped search.
2. At each contact point, estimate the normal force (`O_F_ext_hat_K`).
3. When the first point is found, go to the other side of the normal and sweep back to find the opposite side of the plate.
4. Straight-line sweep through the orthogonal midpoint.
5. Hit one side, then the other.

## Possible add-ons

- Normal-force convergence for the first contact point.
- Rotate the gripper so it stays perpendicular to the motion and more stable on contact.

The probe and the pin are the same part. The arm probes for the plate with the pin.
