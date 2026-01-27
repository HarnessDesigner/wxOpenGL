from typing import TYPE_CHECKING

from . import base3d as _base3d
from ..model_loaders import dxf as _dxf

if TYPE_CHECKING:
    from ..geometry import angle as _angle
    from ..geometry import point as _point
    from .. import Canvas as _Canvas
    from .. import gl_materials as _glm


class MeshDXF(_base3d.Base3D):
    """
    When the file is exported from whatever model program the model needs to
    be tesselated so the proper information will be available in the .dxf file.
    """

    def __init__(self, canvas: "_Canvas", material: "_glm.GLMaterial",
                 selected_material: "_glm.GLMaterial", smooth: bool,
                 file_path: str, position: _point.Point | None = None,
                 angle: _angle.Angle | None = None):

        data = _dxf.load(file_path)

        _base3d.Base3D.__init__(self, canvas, material, selected_material,
                                smooth, data, position, angle)
