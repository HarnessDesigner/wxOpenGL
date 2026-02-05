
import wx

from . import config as _config
from . import canvas as _canvas
from . import mouse_handler as _mouse_handler
from .geometry import point as _point
from .geometry import angle as _angle
from . import gl_materials as _gl_materials
from .objects import mesh_model as _mesh_model
from .objects import mesh_generic as _mesh_generic
from .objects import base3d as _base3d

Config = _config.Config

GenericMaterial = _gl_materials.GenericMaterial
PlasticMaterial = _gl_materials.PlasticMaterial
BlackPlasticMaterial = _gl_materials.BlackPlasticMaterial
CyanPlasticMaterial = _gl_materials.CyanPlasticMaterial
GreenPlasticMaterial = _gl_materials.GreenPlasticMaterial
RedPlasticMaterial = _gl_materials.RedPlasticMaterial
WhitePlasticMaterial = _gl_materials.WhitePlasticMaterial
YellowPlasticMaterial = _gl_materials.YellowPlasticMaterial
RubberMaterial = _gl_materials.RubberMaterial
MetallicMaterial = _gl_materials.MetallicMaterial
PolishedMaterial = _gl_materials.PolishedMaterial
GLMaterial = _gl_materials.GLMaterial

Point = _point.Point
Angle = _angle.Angle

MeshGeneric = _mesh_generic.MeshGeneric
MeshModel = _mesh_model.MeshModel
Base3D = _base3d.Base3D

CONFIG_MOUSE_NONE = _config.MOUSE_NONE
CONFIG_MOUSE_LEFT = _config.MOUSE_LEFT
CONFIG_MOUSE_MIDDLE = _config.MOUSE_MIDDLE
CONFIG_MOUSE_RIGHT = _config.MOUSE_RIGHT
CONFIG_MOUSE_AUX1 = _config.MOUSE_AUX1
CONFIG_MOUSE_AUX2 = _config.MOUSE_AUX2
CONFIG_MOUSE_WHEEL = _config.MOUSE_WHEEL

CONFIG_MOUSE_REVERSE_X_AXIS = _config.MOUSE_REVERSE_X_AXIS
CONFIG_MOUSE_REVERSE_Y_AXIS = _config.MOUSE_REVERSE_Y_AXIS
CONFIG_MOUSE_REVERSE_WHEEL_AXIS = _config.MOUSE_REVERSE_WHEEL_AXIS
CONFIG_MOUSE_SWAP_AXIS = _config.MOUSE_SWAP_AXIS


wxEVT_GL_OBJECT_SELECTED = _mouse_handler.wxEVT_GL_OBJECT_SELECTED
EVT_GL_OBJECT_SELECTED = _mouse_handler.EVT_GL_OBJECT_SELECTED

wxEVT_GL_OBJECT_UNSELECTED = _mouse_handler.wxEVT_GL_OBJECT_UNSELECTED
EVT_GL_OBJECT_UNSELECTED = _mouse_handler.EVT_GL_OBJECT_UNSELECTED

wxEVT_GL_OBJECT_ACTIVATED = _mouse_handler.wxEVT_GL_OBJECT_ACTIVATED
EVT_GL_OBJECT_ACTIVATED = _mouse_handler.EVT_GL_OBJECT_ACTIVATED

wxEVT_GL_OBJECT_RIGHT_CLICK = _mouse_handler.wxEVT_GL_OBJECT_RIGHT_CLICK
EVT_GL_OBJECT_RIGHT_CLICK = _mouse_handler.EVT_GL_OBJECT_RIGHT_CLICK

wxEVT_GL_OBJECT_RIGHT_DCLICK = _mouse_handler.wxEVT_GL_OBJECT_RIGHT_DCLICK
EVT_GL_OBJECT_RIGHT_DCLICK = _mouse_handler.EVT_GL_OBJECT_RIGHT_DCLICK

wxEVT_GL_OBJECT_MIDDLE_CLICK = _mouse_handler.wxEVT_GL_OBJECT_MIDDLE_CLICK
EVT_GL_OBJECT_MIDDLE_CLICK = _mouse_handler.EVT_GL_OBJECT_MIDDLE_CLICK

wxEVT_GL_OBJECT_MIDDLE_DCLICK = _mouse_handler.wxEVT_GL_OBJECT_MIDDLE_DCLICK
EVT_GL_OBJECT_MIDDLE_DCLICK = _mouse_handler.EVT_GL_OBJECT_MIDDLE_DCLICK

wxEVT_GL_OBJECT_AUX1_CLICK = _mouse_handler.wxEVT_GL_OBJECT_AUX1_CLICK
EVT_GL_OBJECT_AUX1_CLICK = _mouse_handler.EVT_GL_OBJECT_AUX1_CLICK

wxEVT_GL_OBJECT_AUX1_DCLICK = _mouse_handler.wxEVT_GL_OBJECT_AUX1_DCLICK
EVT_GL_OBJECT_AUX1_DCLICK = _mouse_handler.EVT_GL_OBJECT_AUX1_DCLICK

wxEVT_GL_OBJECT_AUX2_CLICK = _mouse_handler.wxEVT_GL_OBJECT_AUX2_CLICK
EVT_GL_OBJECT_AUX2_CLICK = _mouse_handler.EVT_GL_OBJECT_AUX2_CLICK

wxEVT_GL_OBJECT_AUX2_DCLICK = _mouse_handler.wxEVT_GL_OBJECT_AUX2_DCLICK
EVT_GL_OBJECT_AUX2_DCLICK = _mouse_handler.EVT_GL_OBJECT_AUX2_DCLICK

GLObjectEvent = _mouse_handler.GLObjectEvent


class Canvas(wx.Panel):

    def __init__(self, parent):
        wx.Panel.__init__(self, parent, wx.ID_ANY, style=wx.BORDER_NONE)
        view_size = _canvas.Canvas.GetViewSize()
        self._ref_count = 0

        self._panel = wx.Panel(self, wx.ID_ANY, pos=(0, 0))
        self._canvas = _canvas.Canvas(self._panel, size=view_size.as_int[:-1], pos=(0, 0))

        self.Bind(wx.EVT_ERASE_BACKGROUND, self._on_erase_background)
        self.Bind(wx.EVT_SIZE, self._on_size)

    def _on_size(self, evt):
        w, h = evt.GetSize()
        self._panel.SetSize((w, h))
        cw, ch = self._canvas.GetSize()

        x = (w - cw) // 2
        y = (h - ch) // 2

        self._canvas.Move(x, y)

    @staticmethod
    def GetViewSize() -> _point.Point:
        return _canvas.Canvas.GetViewSize()

    def AddObject(self, obj):
        self._canvas.AddObject(obj)

    def RemoveObject(self, obj):
        self._canvas.RemoveObject(obj)

    def __enter__(self):
        self._ref_count += 1
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._ref_count -= 1

    def Refresh(self, *args, **kwargs):
        if self._ref_count:
            return

        self._canvas.Refresh(*args, **kwargs)

    def Truck(self, delta) -> None:
        self._canvas.TruckPedestal(delta, 0.0)

    def Pedestal(self, delta) -> None:
        self._canvas.TruckPedestal(0.0, delta)

    def TruckPedestal(self, truck_delta, pedestal_delta) -> None:
        self._canvas.TruckPedestal(truck_delta, pedestal_delta)

    def Zoom(self, delta):
        self._canvas.Zoom(delta, None)

    def RotateAbout(self, delta_x, delta_y) -> None:
        self._canvas.Rotate(delta_x, delta_y)

    def Dolly(self, delta):
        self._canvas.Walk(delta, 0.0)

    def Walk(self, delta_z, delta_x) -> None:
        self._canvas.Walk(delta_z, delta_x)

    def Pan(self, delta):
        self._canvas.PanTilt(delta, 0.0)

    def Tilt(self, delta) -> None:
        self._canvas.PanTilt(0.0, delta)

    def PanTilt(self, pan_delta, tilt_delta):
        self._canvas.PanTilt(pan_delta, tilt_delta)

    # def _on_size(self, evt):
    #     w, h = evt.GetSize()
    #     view_size = _canvas.Canvas.get_view_size()
    #     size = _point.Point(w, h)
    #     pos = (size - view_size) / 2.0
    #
    #     self._canvas.Move(pos.as_int[:-1])
    #
    #     evt.Skip()

    def _on_erase_background(self, _):
        pass
