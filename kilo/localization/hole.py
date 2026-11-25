"""Refine the hole estimate using measured vertical presses."""

import numpy as np

from kilo.localization.archive.hole_filter import HoleSearchFilter


def localize_hole(center, yaw_deg, nominal_top, cfg, press, run, *, rng=None):
    """Update the run with the measured presses and return the final particle filter.

    The press callback returns the measured 3D tip position and whether contact
    stopped the descent. The run record is updated after each measurement.
    """
    settings = cfg["hole_search"]
    depth = settings["drop_depth"]
    # At full chamfer engagement, only the shaft-to-bore clearance permits a drop.
    capture = cfg["hole"]["radius"] - cfg["peg"]["shaft_radius"]
    radius = cfg["hole"]["max_offset_from_center"]
    rng = np.random.default_rng(rng)
    belief = HoleSearchFilter(center, radius, capture, settings["num_particles"], rng=rng)
    yaw = np.radians(yaw_deg)
    axes = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
    cad_offset = axes @ cfg["hole"]["cad_offset"]

    # Take the surface reference outside the hole search disc, on the long axis.
    ref_xy = center + (cfg["box"]["size"][1] / 4) * axes[:, 1]
    # The extra millimetre leaves room for the joint-space IK positioning error.
    margin = .001
    tip, contact = press(ref_xy, nominal_top - depth - margin, "descent-reference")
    run.update(status="no_surface_reference", capture_radius_m=capture, drop_depth_m=depth,
               presses=[], reference={"tip": tip.tolist(), "contact": bool(contact)})
    if not contact:
        return belief
    surface_z = tip[2]
    run["surface_z_m"] = float(surface_z)
    run["status"] = "in_progress"

    # Plate yaw is modulo 180 degrees, so the CAD offset gives two initial probes.
    cad_candidates = [center + cad_offset, center - cad_offset]
    tried = []
    confirmed = False
    for i in range(settings["max_presses"]):
        point = cad_candidates[i] if i < len(cad_candidates) and not confirmed else belief.next_press(
            exclude=tried)
        if point is None:
            break
        tip, contact = press(point, surface_z - depth - margin, f"descent-hole-{i + 1}")
        progressed = float(surface_z - tip[2])
        dropped = progressed >= depth
        run["presses"].append(dict(target_xy=point.tolist(), tip=tip.tolist(), contact=bool(contact),
                                   depth_m=progressed, dropped=dropped))
        if not dropped and not contact:
            # No contact and inadequate measured depth is an incomplete press.
            run["status"] = "incomplete_press"
            return belief
        tried.extend((point, tip[:2]))
        if dropped and not confirmed:
            # The broad prior can have fewer than one particle inside the
            # submillimetre capture disc. Reinitialise within the measured hit.
            belief = HoleSearchFilter(tip[:2], capture, capture, settings["num_particles"], rng=rng)
            confirmed = True
        belief.update_press(tip[:2], dropped)
        mean, covariance = belief.estimate()
        std = float(np.sqrt(np.linalg.eigvalsh(covariance)[-1]))
        run.update(center=mean.tolist(), covariance=covariance.tolist(), std_m=std)
        print(f"Hole press {i + 1}: {'drop' if dropped else 'surface'}, {progressed * 1000:.1f} mm", flush=True)
        if confirmed and std <= settings["stop_std"]:
            run["status"] = "found"
            return belief
    run["status"] = "unconfirmed"
    return belief
