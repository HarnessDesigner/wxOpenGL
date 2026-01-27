import wx

from . import canvas as _canvas
from . import dragging as _dragging
from . import object_picker as _object_picker
from . import free_rotate as _free_rotate
from .geometry import point as _point

from . import config as _config

Config = _config.Config

MOUSE_NONE = _config.MOUSE_NONE
MOUSE_LEFT = _config.MOUSE_LEFT
MOUSE_MIDDLE = _config.MOUSE_MIDDLE
MOUSE_RIGHT = _config.MOUSE_RIGHT
MOUSE_AUX1 = _config.MOUSE_AUX1
MOUSE_AUX2 = _config.MOUSE_AUX2
MOUSE_WHEEL = _config.MOUSE_WHEEL

MOUSE_REVERSE_X_AXIS = _config.MOUSE_REVERSE_X_AXIS
MOUSE_REVERSE_Y_AXIS = _config.MOUSE_REVERSE_Y_AXIS
MOUSE_REVERSE_WHEEL_AXIS = _config.MOUSE_REVERSE_WHEEL_AXIS
MOUSE_SWAP_AXIS = _config.MOUSE_SWAP_AXIS


wxEVT_GL_OBJECT_SELECTED = wx.NewEventType()
EVT_GL_OBJECT_SELECTED = wx.PyEventBinder(wxEVT_GL_OBJECT_SELECTED, 0)

wxEVT_GL_OBJECT_UNSELECTED = wx.NewEventType()
EVT_GL_OBJECT_UNSELECTED = wx.PyEventBinder(wxEVT_GL_OBJECT_UNSELECTED, 0)

wxEVT_GL_OBJECT_ACTIVATED = wx.NewEventType()
EVT_GL_OBJECT_ACTIVATED = wx.PyEventBinder(wxEVT_GL_OBJECT_ACTIVATED, 0)

wxEVT_GL_OBJECT_RIGHT_CLICK = wx.NewEventType()
EVT_GL_OBJECT_RIGHT_CLICK = wx.PyEventBinder(wxEVT_GL_OBJECT_RIGHT_CLICK, 0)

wxEVT_GL_OBJECT_RIGHT_DCLICK = wx.NewEventType()
EVT_GL_OBJECT_RIGHT_DCLICK = wx.PyEventBinder(wxEVT_GL_OBJECT_RIGHT_DCLICK, 0)

wxEVT_GL_OBJECT_MIDDLE_CLICK = wx.NewEventType()
EVT_GL_OBJECT_MIDDLE_CLICK = wx.PyEventBinder(wxEVT_GL_OBJECT_MIDDLE_CLICK, 0)

wxEVT_GL_OBJECT_MIDDLE_DCLICK = wx.NewEventType()
EVT_GL_OBJECT_MIDDLE_DCLICK = wx.PyEventBinder(wxEVT_GL_OBJECT_MIDDLE_DCLICK, 0)

wxEVT_GL_OBJECT_AUX1_CLICK = wx.NewEventType()
EVT_GL_OBJECT_AUX1_CLICK = wx.PyEventBinder(wxEVT_GL_OBJECT_AUX1_CLICK, 0)

wxEVT_GL_OBJECT_AUX1_DCLICK = wx.NewEventType()
EVT_GL_OBJECT_AUX1_DCLICK = wx.PyEventBinder(wxEVT_GL_OBJECT_AUX1_DCLICK, 0)

wxEVT_GL_OBJECT_AUX2_CLICK = wx.NewEventType()
EVT_GL_OBJECT_AUX2_CLICK = wx.PyEventBinder(wxEVT_GL_OBJECT_AUX2_CLICK, 0)

wxEVT_GL_OBJECT_AUX2_DCLICK = wx.NewEventType()
EVT_GL_OBJECT_AUX2_DCLICK = wx.PyEventBinder(wxEVT_GL_OBJECT_AUX2_DCLICK, 0)


class GLObjectEvent(wx.CommandEvent):

    def __init__(self, evtType):
        wx.CommandEvent.__init__(self, evtType)
        self._gl_object = None

    def GetGLObject(self):
        return not self._gl_object

    def SetGLObject(self, obj):
        self._gl_object = obj


class MouseHandler:

    def __init__(self, canvas: _canvas.Canvas):
        self.canvas = canvas

        self._drag_obj: _dragging.DragObject = None
        self.is_motion = False
        self.mouse_pos = None
        self._free_rot: _free_rotate.FreeRotate = None

        canvas.Bind(wx.EVT_LEFT_UP, self.on_left_up)
        canvas.Bind(wx.EVT_LEFT_DOWN, self.on_left_down)
        canvas.Bind(wx.EVT_LEFT_DCLICK, self.on_left_dclick)

        canvas.Bind(wx.EVT_MIDDLE_UP, self.on_middle_up)
        canvas.Bind(wx.EVT_MIDDLE_DOWN, self.on_middle_down)
        canvas.Bind(wx.EVT_MIDDLE_DCLICK, self.on_middle_dclick)

        canvas.Bind(wx.EVT_RIGHT_UP, self.on_right_up)
        canvas.Bind(wx.EVT_RIGHT_DOWN, self.on_right_down)
        canvas.Bind(wx.EVT_RIGHT_DCLICK, self.on_right_dclick)

        canvas.Bind(wx.EVT_MOUSE_AUX1_UP, self.on_aux1_up)
        canvas.Bind(wx.EVT_MOUSE_AUX1_DOWN, self.on_aux1_down)
        canvas.Bind(wx.EVT_MOUSE_AUX1_DCLICK, self.on_aux1_dclick)

        canvas.Bind(wx.EVT_MOUSE_AUX2_UP, self.on_aux2_up)
        canvas.Bind(wx.EVT_MOUSE_AUX2_DOWN, self.on_aux2_down)
        canvas.Bind(wx.EVT_MOUSE_AUX2_DCLICK, self.on_aux2_dclick)

        canvas.Bind(wx.EVT_MOTION, self.on_mouse_motion)
        canvas.Bind(wx.EVT_MOUSEWHEEL, self.on_mouse_wheel)

    def _process_mouse(self, code):
        for config, func in (
            (Config.walk, self.canvas.walk),
            (Config.truck_pedistal, self.canvas.truck_pedistal),
            (Config.reset, self.canvas.camera.reset),
            (Config.rotate, self.canvas.rotate),
            (Config.pan_tilt, self.canvas.pan_tilt),
            (Config.zoom, self.canvas.zoom)
        ):
            if config.mouse is None:
                continue

            if config.mouse & code:

                def _wrapper(dx, dy):
                    if config.mouse & MOUSE_SWAP_AXIS:
                        func(dy, dx)
                    else:
                        func(dx, dy)

                return _wrapper

        def _do_nothing_func(_, __):
            pass

        return _do_nothing_func

    def on_left_down(self, evt: wx.MouseEvent):
        x, y = evt.GetPosition()
        mouse_pos = _point.Point(x, y)
        self.mouse_pos = mouse_pos
        self.is_motion = False

        selected = _object_picker.find_object(mouse_pos, self.canvas.objects)

        if selected:
            event = GLObjectEvent(wxEVT_GL_OBJECT_SELECTED)
            event.SetId(self.canvas.GetId())
            event.SetEventObject(self.canvas)
            event.SetGLObject(selected)
            self.canvas.GetEventHandler().ProcessEvent(event)

            if not event.ShouldPropagate():
                evt.Skip()
                return

            with self.canvas:
                if self._drag_obj is not None:
                    self._drag_obj.owner.is_selected = False

                # prepare exact drag using project/unproject anchor approach
                if not self.canvas.HasCapture():
                    self.canvas.CaptureMouse()

                # compute object's center world point from its hit_test_rect

                if selected == self.canvas.selected:
                    if selected.is_move_shown:
                        selected.stop_move()
                        self._drag_obj = None
                    elif selected.is_angle_shown:
                        self._free_rot = _free_rotate.FreeRotate(
                            self.canvas, self.canvas.selected, x, y)
                    else:
                        selected.is_selected = False
                        self.canvas.selected = None
                else:
                    self._free_rot = None

                    # project center to screen
                    win_point = self.canvas.camera.project_point(selected.position)

                    # compute pick-world and offsets
                    pick_world = self.canvas.camera.unproject_point(win_point)
                    obj = selected.get_parent_object()
                    pick_offset = obj.position - pick_world

                    selected.is_selected = True

                    # store drag state on canvas
                    self._drag_obj = _dragging.DragObject(
                        obj, selected, anchor_screen=win_point,
                        pick_offset=pick_offset, mouse_start=mouse_pos,
                        start_obj_pos=obj.position.copy(),
                        last_pos=obj.position.copy())

                self.canvas.Refresh(True)

            self.canvas.Refresh(False)

    def on_left_up(self, evt: wx.MouseEvent):
        self._process_mouse_release(evt)
        with self.canvas:
            self._free_rot = None

            if self._drag_obj is not None:
                self._drag_obj.obj.is_selected = False
                self._drag_obj = None

                if self.canvas.HasCapture():
                    self.canvas.ReleaseMouse()

                self.canvas.Refresh(False)

            evt.Skip()

            if not self.is_motion:
                x, y = evt.GetPosition()
                mouse_pos = _point.Point(x, y)  # NOQA

            if not evt.RightIsDown():
                if self.canvas.HasCapture():
                    self.canvas.ReleaseMouse()

                self.mouse_pos = None

            self.is_motion = False

        evt.Skip()

    def on_left_dclick(self, evt: wx.MouseEvent):
        x, y = evt.GetPosition()
        mouse_pos = _point.Point(x, y)
        selected = _object_picker.find_object(mouse_pos, self.canvas.objects)

        if selected:
            event = GLObjectEvent(wxEVT_GL_OBJECT_ACTIVATED)
            event.SetId(self.canvas.GetId())
            event.SetEventObject(self.canvas)
            event.SetGLObject(selected)
            self.canvas.GetEventHandler().ProcessEvent(event)

        self._process_mouse_release(evt)
        evt.Skip()

    def on_middle_up(self, evt: wx.MouseEvent):
        if not self.is_motion:
            x, y = evt.GetPosition()
            mouse_pos = _point.Point(x, y)
            selected = _object_picker.find_object(mouse_pos, self.canvas.objects)

            if selected:
                event = GLObjectEvent(wxEVT_GL_OBJECT_MIDDLE_CLICK)
                event.SetId(self.canvas.GetId())
                event.SetEventObject(self.canvas)
                event.SetGLObject(selected)
                self.canvas.GetEventHandler().ProcessEvent(event)

        self._process_mouse_release(evt)
        evt.Skip()

    def on_middle_down(self, evt: wx.MouseEvent):
        self.is_motion = False

        if not self.canvas.HasCapture():
            self.canvas.CaptureMouse()

        x, y = evt.GetPosition()
        self.mouse_pos = _point.Point(x, y)

        evt.Skip()

    def on_middle_dclick(self, evt: wx.MouseEvent):
        x, y = evt.GetPosition()
        mouse_pos = _point.Point(x, y)
        selected = _object_picker.find_object(mouse_pos, self.canvas.objects)

        if selected:
            event = GLObjectEvent(wxEVT_GL_OBJECT_MIDDLE_DCLICK)
            event.SetId(self.canvas.GetId())
            event.SetEventObject(self.canvas)
            event.SetGLObject(selected)
            self.canvas.GetEventHandler().ProcessEvent(event)

        self._process_mouse_release(evt)
        evt.Skip()

    def on_right_up(self, evt: wx.MouseEvent):
        if not self.is_motion:
            x, y = evt.GetPosition()
            mouse_pos = _point.Point(x, y)
            selected = _object_picker.find_object(mouse_pos, self.canvas.objects)

            if selected:
                event = GLObjectEvent(wxEVT_GL_OBJECT_RIGHT_CLICK)
                event.SetId(self.canvas.GetId())
                event.SetEventObject(self.canvas)
                event.SetGLObject(selected)
                self.canvas.GetEventHandler().ProcessEvent(event)

        self._process_mouse_release(evt)
        evt.Skip()

    def on_right_down(self, evt: wx.MouseEvent):
        self.is_motion = False

        if not self.canvas.HasCapture():
            self.canvas.CaptureMouse()

        x, y = evt.GetPosition()
        self.mouse_pos = _point.Point(x, y)

        evt.Skip()

    def on_right_dclick(self, evt: wx.MouseEvent):
        x, y = evt.GetPosition()
        mouse_pos = _point.Point(x, y)
        selected = _object_picker.find_object(mouse_pos, self.canvas.objects)

        if selected:
            event = GLObjectEvent(wxEVT_GL_OBJECT_RIGHT_DCLICK)
            event.SetId(self.canvas.GetId())
            event.SetEventObject(self.canvas)
            event.SetGLObject(selected)
            self.canvas.GetEventHandler().ProcessEvent(event)

        self._process_mouse_release(evt)
        evt.Skip()

    def on_mouse_wheel(self, evt: wx.MouseEvent):
        if evt.GetWheelRotation() > 0:
            delta = 1.0
        else:
            delta = -1.0

        self._process_mouse(MOUSE_WHEEL)(delta, 0.0)

        self.canvas.Refresh(False)
        evt.Skip()

    def on_mouse_motion(self, evt: wx.MouseEvent):
        if evt.Dragging():
            x, y = evt.GetPosition()
            new_mouse_pos = _point.Point(x, y)

            if self.mouse_pos is None:
                self.mouse_pos = new_mouse_pos

            delta = new_mouse_pos - self.mouse_pos
            self.mouse_pos = new_mouse_pos

            with self.canvas:
                if evt.LeftIsDown():
                    if self._drag_obj is not None:
                        if self._drag_obj.owner.is_move_shown:
                            self._drag_obj.move(self, new_mouse_pos)

                        elif self._drag_obj.owner.is_angle_shown:
                            self._drag_obj.rotate(self, new_mouse_pos)

                    elif self._free_rot is not None:
                        self._free_rot(x, y)
                    else:
                        self.is_motion = True
                        self._process_mouse(MOUSE_LEFT)(*list(delta)[:-1])

                if evt.MiddleIsDown():
                    self.is_motion = True
                    self._process_mouse(MOUSE_MIDDLE)(*list(delta)[:-1])
                if evt.RightIsDown():
                    self.is_motion = True
                    self._process_mouse(MOUSE_RIGHT)(*list(delta)[:-1])
                if evt.Aux1IsDown():
                    self.is_motion = True
                    self._process_mouse(MOUSE_AUX1)(*list(delta)[:-1])
                if evt.Aux2IsDown():
                    self.is_motion = True
                    self._process_mouse(MOUSE_AUX2)(*list(delta)[:-1])

            self.canvas.Refresh(False)

        evt.Skip()

    def _process_mouse_release(self, evt: wx.MouseEvent):
        if True not in (
            evt.LeftIsDown(),
            evt.MiddleIsDown(),
            evt.RightIsDown(),
            evt.Aux1IsDown(),
            evt.Aux2IsDown()
        ):
            if self.canvas.HasCapture():
                self.canvas.ReleaseMouse()

    def on_aux1_up(self, evt: wx.MouseEvent):
        if not self.is_motion:
            x, y = evt.GetPosition()
            mouse_pos = _point.Point(x, y)
            selected = _object_picker.find_object(mouse_pos, self.canvas.objects)

            if selected:
                event = GLObjectEvent(wxEVT_GL_OBJECT_AUX1_CLICK)
                event.SetId(self.canvas.GetId())
                event.SetEventObject(self.canvas)
                event.SetGLObject(selected)
                self.canvas.GetEventHandler().ProcessEvent(event)

        self._process_mouse_release(evt)
        evt.Skip()

    def on_aux1_down(self, evt: wx.MouseEvent):
        self.is_motion = False

        if not self.canvas.HasCapture():
            self.canvas.CaptureMouse()

        x, y = evt.GetPosition()
        self.mouse_pos = _point.Point(x, y)

        evt.Skip()

    def on_aux1_dclick(self, evt: wx.MouseEvent):
        x, y = evt.GetPosition()
        mouse_pos = _point.Point(x, y)
        selected = _object_picker.find_object(mouse_pos, self.canvas.objects)

        if selected:
            event = GLObjectEvent(wxEVT_GL_OBJECT_AUX1_DCLICK)
            event.SetId(self.canvas.GetId())
            event.SetEventObject(self.canvas)
            event.SetGLObject(selected)
            self.canvas.GetEventHandler().ProcessEvent(event)

        self._process_mouse_release(evt)
        evt.Skip()

    def on_aux2_up(self, evt: wx.MouseEvent):
        if not self.is_motion:
            x, y = evt.GetPosition()
            mouse_pos = _point.Point(x, y)
            selected = _object_picker.find_object(mouse_pos, self.canvas.objects)

            if selected:
                event = GLObjectEvent(wxEVT_GL_OBJECT_AUX2_CLICK)
                event.SetId(self.canvas.GetId())
                event.SetEventObject(self.canvas)
                event.SetGLObject(selected)
                self.canvas.GetEventHandler().ProcessEvent(event)

        self._process_mouse_release(evt)
        evt.Skip()

    def on_aux2_down(self, evt: wx.MouseEvent):
        self.is_motion = False

        if not self.canvas.HasCapture():
            self.canvas.CaptureMouse()

        x, y = evt.GetPosition()
        self.mouse_pos = _point.Point(x, y)

        evt.Skip()

    def on_aux2_dclick(self, evt: wx.MouseEvent):
        x, y = evt.GetPosition()
        mouse_pos = _point.Point(x, y)
        selected = _object_picker.find_object(mouse_pos, self.canvas.objects)

        if selected:
            event = GLObjectEvent(wxEVT_GL_OBJECT_AUX2_DCLICK)
            event.SetId(self.canvas.GetId())
            event.SetEventObject(self.canvas)
            event.SetGLObject(selected)
            self.canvas.GetEventHandler().ProcessEvent(event)

        self._process_mouse_release(evt)
        evt.Skip()
