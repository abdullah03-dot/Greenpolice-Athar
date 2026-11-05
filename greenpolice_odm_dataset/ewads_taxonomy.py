"""
EWADS class taxonomy -- ONE place that says which of the 18 fine classes belong together.

Used by
  * 08_tiled_eval.py    hierarchical metrics (fine -> group -> "any e-waste")
  * ewads_fusion.py     cross-frame association may link two views whose fine classes differ only
                        if they are in the same group (a CRT seen as a flat screen from 40 m is the
                        same object; a CRT seen as a cable is not)
  * 11_evidence_pipeline.py   reports the group alongside the fine class

WHY GROUPS
  At survey altitude, several fine classes are not separable from pixels (a 9 px phone vs a 9 px
  battery).  Scoring "battery vs mobile_phone" there measures a coin flip, and the confusion costs
  mAP without telling the operator anything wrong: both are lithium-bearing small devices.  The
  group level asks the question the image CAN answer; the fine level is reported only where the
  object is big enough (see ewads_survey.resolvability).

Groups are by what a nadir camera sees (size + silhouette) AND by handling hazard, so a group-level
claim is still useful in an enforcement report.  Change them only BEFORE you look at test results
(the frozen targets file records this module's hash -- see tools/freeze_targets.py).
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Dict, List

CLASSES: List[str] = [
    "crt_monitor", "flat_screen", "circuit_board", "battery", "cable_bundle", "single_cable",
    "mobile_phone", "keyboard", "printer", "small_appliance", "large_appliance", "cpu_chip",
    "hard_drive", "power_supply", "ewaste_pile", "plastic_casing", "metal_component", "lighting",
]

GROUPS: Dict[str, List[str]] = {
    "display":            ["crt_monitor", "flat_screen"],                    # box/panel, glass; CRT = lead
    "appliance":          ["large_appliance", "small_appliance", "printer"],  # white goods / housings
    "computer_component": ["keyboard", "hard_drive", "power_supply", "plastic_casing", "metal_component"],
    "board_chip":         ["circuit_board", "cpu_chip"],                     # PCB material, heavy metals
    "cell_device":        ["battery", "mobile_phone"],                       # lithium / lead cells
    "cable":              ["cable_bundle", "single_cable"],                  # copper + PVC (burn sites)
    "lighting":           ["lighting"],                                      # tubes/CFL: mercury
    "pile":               ["ewaste_pile"],                                   # mixed heap
}

# Hazard tier: the single definition, used by 07's forensic log, 08 / 11 reports and target E3.
TIER_1 = {"battery", "circuit_board", "crt_monitor"}

FINE_TO_GROUP: Dict[str, str] = {c: g for g, cs in GROUPS.items() for c in cs}
GROUP_NAMES: List[str] = list(GROUPS)
FINE_TO_GROUP_ID: List[int] = [GROUP_NAMES.index(FINE_TO_GROUP[c]) for c in CLASSES]

assert sorted(FINE_TO_GROUP) == sorted(CLASSES), "every class must be in exactly one group"
assert len(FINE_TO_GROUP) == sum(len(v) for v in GROUPS.values()), "a class is in two groups"

LEVELS = ("fine", "group", "binary")


def n_classes(level: str) -> int:
    return {"fine": len(CLASSES), "group": len(GROUP_NAMES), "binary": 1}[level]


def level_names(level: str) -> List[str]:
    return {"fine": CLASSES, "group": GROUP_NAMES, "binary": ["ewaste"]}[level]


def map_ids(cls_ids, level: str):
    """fine class ids (array-like) -> ids at `level`."""
    import numpy as np
    c = np.asarray(cls_ids, dtype=int)
    if level == "fine":
        return c
    if level == "group":
        return np.asarray(FINE_TO_GROUP_ID, dtype=int)[c] if len(c) else c
    if level == "binary":
        return np.zeros_like(c)
    raise ValueError(level)


def same_group(a: int, b: int) -> bool:
    return FINE_TO_GROUP_ID[int(a)] == FINE_TO_GROUP_ID[int(b)]


def taxonomy_sha256() -> str:
    """Hash of this file -- recorded with frozen targets so a later regrouping is visible."""
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
