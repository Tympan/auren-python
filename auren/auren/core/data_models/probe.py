""" Data model for the probe
"""
import typing as t
from pydantic import BaseModel

from auren.core.utils import GetUnitsMixin, IDMixin


class Probe(IDMixin, GetUnitsMixin, BaseModel):
    """Base class that describes an Auren probe

    This basically describes the location of mics, and probe diameter

    """

    # The units used to describe the probe geometry
    units: str = "mm"
    interior_diameter: float = 3.4
    exterior_diameter: float = 4.5
    length: float = 31.7
    length_with_tip: float = 35.4
    mic_positions: t.List[float] = [0.4, 5.9, 9.5, 13.4]
