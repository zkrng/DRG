"""
Rule-based parsing of CMS IPPS \"DRG Definition\" strings.

Splits each row into DRG code, main clinical description, and a trailing
severity / complication tag (W MCC, W CC, W/O CC/MCC, OR CHEMOTHE, etc.).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional, Tuple

# Longest / most specific patterns first (end-anchored, case-insensitive).
_CC_TAG_SUFFIXES: Tuple[str, ...] = (
    r"W\s+MCC\s+OR\s+CHEMOTHE",
    r"W\s+CC\s+OR\s+TPA\s+IN\s+24\s+HRS",
    r"W\s+MCC\s+OR\s+INTESTINAL\s+TRANSPLANT",
    r"W\s+MCC\s+OR\s+4\+\s+VESSELS/STENTS",
    r"W\s+MCC\s+OR\s+4\+\s+VES/STENTS",
    r"W\s+MCC\s+OR\s+DISC\s+DEVICE/NEUROSTIM",
    r"W\s+CC\s+OR\s+SPINAL\s+NEUROSTIMULATORS",
    r"W\s+CC\s+OR\s+PERIPH\s+NEUROSTIM",
    r"W\s+CC\s+OR\s+HIGH\s+DOSE\s+CHEMO\s+AGENT",
    r"W/O\s+CC/MCC",
    r"W\s+CC/MCC\s+OR\s+MAJOR\s+DEVICE",
    r"W\s+CC/MCC",
    r"W/O\s+MCC",
    r"W\s+VENTILATOR\s+SUPPORT\s+>96\s+HOURS",
    r"W\s+VENTILATOR\s+SUPPORT\s+<=96\s+HOURS",
    r"W\s+COMPLICATING\s+DIAGNOSES",
    r"W/O\s+COMPLICATING\s+DIAGNOSES",
    r"W\s+MAJ\s+O\.R\.",
    r"W/O\s+MAJ\s+O\.R\.",
    r"W\s+MCC",
    r"W\s+CC",
    r"W\s+THROM",
    r"OR\s+CHEMOTHE",
    r"W\s+MEDICAL\s+COMPLICATIONS",
    r"W\s+MV\s+>96\s+HOURS",
    r"W\s+MV\s+<=96\s+HOURS",
    r"W\s+REHABILITATION\s+THERAPY",
    r"W\s+OR\s+W/O\s+OTHER\s+RELATED\s+CONDITION",
)

_COMPILED = tuple(
    re.compile(rf"\s+{pat}$", re.IGNORECASE) for pat in _CC_TAG_SUFFIXES
)

_CODE_DESC = re.compile(
    r"^\s*(\d{3})\s*-\s*(.+?)\s*$",
    re.DOTALL,
)

_OR_SPLIT = re.compile(r"\s+OR\s+", re.IGNORECASE)

# Fixed-width OR-segments (extra segments are truncated; increase if CMS adds longer tags).
MAX_CC_TAG_OR_PARTS = 4
CC_TAG_PART_COLUMN_NAMES: Tuple[str, ...] = tuple(
    f"cc_tag_part_{i}" for i in range(1, MAX_CC_TAG_OR_PARTS + 1)
)


def _cc_tag_or_into_four_slots(tag: Optional[str]) -> list[Optional[str]]:
    """Split ``cc_tag`` on `` OR `` into exactly four slots (None = unused)."""
    out: list[Optional[str]] = [None, None, None, None]
    if tag is None:
        return out
    s = str(tag).strip()
    if not s:
        return out

    raw = _OR_SPLIT.split(s)
    if (
        len(raw) == 2
        and raw[0] == "W"
        and raw[1].lstrip().startswith("W/")
    ):
        segments = [s]
    else:
        segments = [p.strip() for p in raw if p.strip()]

    for i in range(min(MAX_CC_TAG_OR_PARTS, len(segments))):
        out[i] = segments[i]
    return out


@dataclass(frozen=True)
class DRGParts:
    drg_code: str
    main_diagnosis: str
    cc_tag: Optional[str]
    cc_tag_part_1: Optional[str]
    cc_tag_part_2: Optional[str]
    cc_tag_part_3: Optional[str]
    cc_tag_part_4: Optional[str]
    raw_definition: str


def parse_drg_definition(text: object) -> DRGParts:
    """
    Parse a single DRG Definition cell into structured fields.

    * drg_code: three-digit string (leading zeros kept).
    * main_diagnosis: clinical text with trailing CC/MCC-style tag removed when matched.
    * cc_tag: trailing qualifier (e.g. \"W MCC\", \"W/O CC/MCC\") or None if none matched.
    * cc_tag_part_1 … cc_tag_part_4: segments after splitting ``cc_tag`` on `` OR ``
      (unused slots are None).
    """
    if text is None or (isinstance(text, float) and str(text) == "nan"):
        return DRGParts("", "", None, None, None, None, None, "")

    s = str(text).strip().strip('"').strip()
    if not s:
        return DRGParts("", "", None, None, None, None, None, "")

    m = _CODE_DESC.match(s)
    if not m:
        return DRGParts("", s, None, None, None, None, None, s)

    code, desc = m.group(1), m.group(2).strip()
    tag: Optional[str] = None
    remainder = desc

    for rx in _COMPILED:
        mm = rx.search(remainder)
        if mm:
            tag = remainder[mm.start() :].strip()
            remainder = remainder[: mm.start()].rstrip()
            break

    slots = _cc_tag_or_into_four_slots(tag)

    return DRGParts(
        drg_code=code,
        main_diagnosis=remainder,
        cc_tag=tag,
        cc_tag_part_1=slots[0],
        cc_tag_part_2=slots[1],
        cc_tag_part_3=slots[2],
        cc_tag_part_4=slots[3],
        raw_definition=s,
    )


def add_parsed_columns(df, column: str = "DRG Definition"):
    """Return a copy of df with drg_code, main_diagnosis, cc_tag, cc_tag_part_1..4 columns."""
    out = df.copy()
    parsed = out[column].map(parse_drg_definition)
    out["drg_code"] = parsed.map(lambda p: p.drg_code)
    out["main_diagnosis"] = parsed.map(lambda p: p.main_diagnosis)
    out["cc_tag"] = parsed.map(lambda p: p.cc_tag)
    out["cc_tag_part_1"] = parsed.map(lambda p: p.cc_tag_part_1)
    out["cc_tag_part_2"] = parsed.map(lambda p: p.cc_tag_part_2)
    out["cc_tag_part_3"] = parsed.map(lambda p: p.cc_tag_part_3)
    out["cc_tag_part_4"] = parsed.map(lambda p: p.cc_tag_part_4)
    return out
