"""
Data model for probe geometry
"""
import typing as t
from pydantic import BaseModel

class TubeGeometry(BaseModel):
    units : str = "mm"
    type : t.Literal["open", "closed"] = "closed"
    length : float
    interior_diameter : float = 3.4
    probe_locations : t.List[float] = [0.4]


# instantiate a set of standard geometries used by the Auren
tube_12mm = TubeGeometry(
    units="mm",
    type="closed",
    length=12,
    interior_diameter=3.4,
    probe_locations=[0.4],
)
tube_14p5mm = TubeGeometry(
    units="mm",
    type="closed",
    length=14.5,
    interior_diameter=3.4,
    probe_locations=[0.4],
)
tube_17p5mm = TubeGeometry(
    units="mm",
    type="closed",
    length=17.5,
    interior_diameter=3.4,
    probe_locations=[0.4],
)
tube_20mm = TubeGeometry(
    units="mm",
    type="closed",
    length=20,
    interior_diameter=3.4,
    probe_locations=[0.4],
)

simulator_1p26cc = TubeGeometry(
    units="mm",
    type="closed",
    length=28.5,
    interior_diameter=7.5,
    probe_locations=[0.4, 28.1],
)
simulator_2cc = TubeGeometry(
    units="mm",
    type="closed",
    length=28.5,
    interior_diameter=9.452,
    probe_locations=[0.4, 28.1],
)

