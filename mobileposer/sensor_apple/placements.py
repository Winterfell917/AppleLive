"""Explicit recording placement labels, separate from legacy model slots.

Handheld phones use the wrist-slot N-pose as a declared frame proxy, not a claim
that a hand-mounted sensor measures the forearm or matches a trained model.
"""
import numpy as np

WATCH_PLACEMENTS = ("left_wrist", "right_wrist")
PHONE_PLACEMENTS = ("left_hand", "right_hand", "left_thigh_pocket", "right_thigh_pocket")
PLACEMENT_SLOTS = {"left_wrist": 0, "right_wrist": 1, "left_hand": 0,
                   "right_hand": 1, "left_thigh_pocket": 2, "right_thigh_pocket": 3}
SLOT_FRAMES = ("left_wrist", "right_wrist", "left_thigh", "right_thigh", "head", "left_foot", "right_foot")
DEFAULT_PLACEMENTS = {"apple_watch": "left_wrist", "iphone": "right_thigh_pocket"}


def describe_placements(slots, placements=None, selection="numeric_slots_unspecified"):
    if not slots or any(type(v) is not int or v not in range(7) for v in slots.values()):
        raise ValueError("Body slots must be integers 0..6")
    names = {s: "unspecified" for s in slots} if placements is None else dict(placements)
    if set(names) != set(slots):
        raise ValueError("Placement labels must cover exactly the selected sources")
    for s, name in names.items():
        if name == "unspecified":
            continue
        allowed = {"apple_watch": WATCH_PLACEMENTS, "iphone": PHONE_PLACEMENTS}.get(s, ())
        if name not in allowed or PLACEMENT_SLOTS[name] != slots[s]:
            raise ValueError(f"Placement {name!r} is incompatible with {s} slot {slots[s]}")
    collisions = {str(slot): [s for s in slots if slots[s] == slot] for slot in set(slots.values())
                  if list(slots.values()).count(slot) > 1}
    proxies = [s for s, name in names.items() if name in ("left_hand", "right_hand")]
    warnings = ["Placement must remain fixed within each calibrated segment; changing hand/pocket/grip requires recalibration.",
                "Placement support in this exporter does not establish compatibility with a downstream trained model."]
    if selection != "explicit":
        warnings.append("Physical placement was not explicitly declared; verify labels/slots before using the calibrated data.")
    if proxies:
        warnings.append("Handheld phone uses a wrist-slot N-pose frame proxy. Keep wrists neutral during the calibration pose and grip fixed afterwards; it is not an interchangeable forearm sensor.")
    if collisions:
        warnings.append("Multiple devices share a legacy body slot. Keep the source axis; seven-slot RMI/RSB export is disabled to prevent overwriting a device.")
    return {"selection": selection, "source_slots": dict(slots), "source_placements": names,
            "source_target_frames": {s: SLOT_FRAMES[v] + ("_proxy" if s in proxies else "") for s, v in slots.items()},
            "hand_wrist_proxy_sources": proxies, "slot_collisions": collisions,
            "legacy_slot_export_available": not bool(collisions),
            "downstream_model_compatibility_verified": False, "warnings": warnings}


def resolve_placements(slots=None, watch=None, phone=None):
    if watch is not None or phone is not None:
        if slots is not None:
            raise ValueError("Do not mix named placement flags with --source-slot")
        if watch is None or phone is None:
            raise ValueError("Provide both --watch-placement and --phone-placement to label this recording explicitly")
        names = {"apple_watch": watch, "iphone": phone}
        if watch not in WATCH_PLACEMENTS or phone not in PHONE_PLACEMENTS:
            raise ValueError("Unsupported Watch/iPhone placement")
        slots = {s: PLACEMENT_SLOTS[name] for s, name in names.items()}
        return slots, describe_placements(slots, names, "explicit")
    if slots is not None:
        return slots, describe_placements(slots)
    slots = {s: PLACEMENT_SLOTS[name] for s, name in DEFAULT_PLACEMENTS.items()}
    return slots, describe_placements(slots, DEFAULT_PLACEMENTS, "legacy_defaults_assumed")


def placement_arrays(info, sources):
    """Unicode/scalar arrays only: readable with allow_pickle=False."""
    return {"source_placements": np.array([info["source_placements"][s] for s in sources]),
            "source_target_frames": np.array([info["source_target_frames"][s] for s in sources]),
            "placement_selection": np.array(info["selection"]),
            "legacy_slot_export_available": np.array(info["legacy_slot_export_available"])}


def source_rotations(cal, source):
    """Read authoritative per-device matrices, with a safe legacy fallback."""
    if "source_calibration" in cal:
        entry = cal["source_calibration"][source]
        if entry["slot"] != cal["source_slots"][source]:
            raise ValueError(f"Inconsistent calibration slot for {source}")
        return np.asarray(entry["RMI"], dtype=float), np.asarray(entry["RSB"], dtype=float)
    slots = cal["source_slots"]
    if len(set(slots.values())) != len(slots):
        raise ValueError("Legacy slot-only calibration cannot represent multiple devices in the same slot")
    return tuple(np.asarray(cal[key][slots[source]], dtype=float) for key in ("RMI", "RSB"))
