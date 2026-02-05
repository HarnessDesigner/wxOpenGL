import wx
from wx.lib import scrolledpanel
from wx.lib.agw import cubecolourdialog

from . import config as _config


Config = _config.Config


class ColorButton(wx.Button):

    @staticmethod
    def cc(color):
        r = color.Red()
        g = color.Green()
        b = color.Blue()

        return wx.Colour(255 - r, 255 - g, 255 - b)

    def __init__(self, parent, color, id=wx.ID_ANY):  # NOQA
        wx.Button.__init__(self, parent, id, label='Set Color')

        color = wx.Colour(*[int(item * 255) for item in color])
        self.SetBackgroundColour(color)

    def SetBackgroundColour(self, colour):
        color = wx.Colour(colour.Red(), colour.Green(), colour.Blue())
        wx.Button.SetBackgroundColour(self, color)
        self.SetForegroundColour(self.cc(color))


def HSizer(parent, label, ctrl) -> wx.BoxSizer:

    sizer = wx.BoxSizer(wx.HORIZONTAL)

    st = wx.StaticText(parent, wx.ID_ANY, label=label)
    sizer.Add(st, 0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 5)
    sizer.Add(ctrl, 0, wx.ALIGN_CENTER_VERTICAL | wx.ALIGN_RIGHT)
    return sizer


class HeadlightTab(scrolledpanel.ScrolledPanel):

    def __init__(self, parent):

        scrolledpanel.ScrolledPanel.__init__(self, parent, wx.ID_ANY, style=wx.BORDER_NONE)
        self.SetupScrolling()

        enable = Config.headlight.turn_on

        self.enable = wx.CheckBox(self, wx.ID_ANY, label='')
        self.enable.SetValue(enable)
        self.enable.Bind(wx.EVT_CHECKBOX, self.on_enable)

        self.size = wx.SpinCtrlDouble(self, wx.ID_ANY, value=str(round(Config.headlight.cutoff, 1)),
                                      initial=round(Config.headlight.cutoff, 1),
                                      min=1.0, max=1.0, inc=0.1)

        self.size.Bind(wx.EVT_SPINCTRLDOUBLE, self.on_size)
        self.size.Enable(enable)

        self.dissipate = wx.SpinCtrlDouble(self, wx.ID_ANY, value=str(round(Config.headlight.dissipate, 1)),
                                           initial=round(Config.headlight.dissipate, 1),
                                           min=1.0, max=100.0, inc=0.5)

        self.dissipate.Bind(wx.EVT_SPINCTRLDOUBLE, self.on_dissipate)
        self.dissipate.Enable(enable)

        self.color = ColorButton(self, Config.headlight.color)
        self.color.Bind(wx.EVT_BUTTON, self.on_color)
        self.color.Enable(enable)

        vsizer = wx.BoxSizer(wx.VERTICAL)

        enable_sizer = HSizer(self, 'Enable:', self.enable)
        size_sizer = HSizer(self, 'Size:', self.size)
        dissipate_sizer = HSizer(self, 'Dissipate:', self.dissipate)
        color_sizer = HSizer(self, 'Beam Color:', self.color)

        vsizer.Add(enable_sizer, 0, wx.ALL, 5)
        vsizer.Add(size_sizer, 0, wx.ALL, 5)
        vsizer.Add(dissipate_sizer, 0, wx.ALL, 5)
        vsizer.Add(color_sizer, 0, wx.ALL, 5)

        hsizer = wx.BoxSizer(wx.HORIZONTAL)
        hsizer.Add(vsizer, 0, wx.ALL, 10)
        self.SetSizerAndFit(hsizer)

    def on_enable(self, evt):
        value = self.enable.GetValue()
        self.dissipate.Enable(value)
        self.size.Enable(value)
        self.color.Enable(value)
        Config.headlight.turn_on = value
        evt.Skip()

    def on_dissipate(self, evt):
        Config.headlight.dissipate = self.dissipate.GetValue()
        evt.Skip()

    def on_size(self, evt):
        Config.headlight.cutoff = self.size.GetValue()
        evt.Skip()

    def on_color(self, evt):
        color_data = wx.ColourData()
        color_data.FromString(Config.colors.custom_colors)

        color = wx.Colour(*[int(item * 255) for item in Config.headlight.color])
        color_data.SetColour(color)
        color_data.SetChooseAlpha(True)

        dialog = cubecolourdialog.CubeColourDialog(self, color_data)
        dialog.ShowModal()
        color_data = dialog.GetColourData()
        new_color = color_data.GetColour()
        color_data.SetColour(wx.Colour(255, 255, 255, 255))

        Config.colors.custom_colors = color_data.ToString()

        Config.headlight.color = [item / 255.0 for item in
                                  (new_color.Red(), new_color.Green(),
                                   new_color.Blue(), new_color.GetAlpha())]

        self.color.SetBackgroundColour(new_color)
        evt.Skip()


class DebugTab(scrolledpanel.ScrolledPanel):

    def __init__(self, parent):

        scrolledpanel.ScrolledPanel.__init__(self, parent, wx.ID_ANY, style=wx.BORDER_NONE)
        self.SetupScrolling()

        enable = Config.debug.bypass

        self.enable = wx.CheckBox(self, wx.ID_ANY, label='')
        self.enable.SetValue(enable)
        self.enable.Bind(wx.EVT_CHECKBOX, self.on_enable)

        self.arguments = wx.CheckBox(self, wx.ID_ANY, label='')
        self.arguments.SetValue(Config.debug.log_args)
        self.arguments.Bind(wx.EVT_CHECKBOX, self.on_arguments)
        self.arguments.Enable(enable)

        self.func_speed = wx.CheckBox(self, wx.ID_ANY, label='')
        self.func_speed.SetValue(Config.debug.call_duration)
        self.func_speed.Bind(wx.EVT_CHECKBOX, self.on_func_speed)
        self.arguments.Enable(enable)

        vsizer = wx.BoxSizer(wx.VERTICAL)

        enable_sizer = HSizer(self, 'Enable:', self.enable)
        arguments_sizer = HSizer(self, 'Call Arguments:', self.arguments)
        func_speed_sizer = HSizer(self, 'Function Speed:', self.func_speed)

        vsizer.Add(enable_sizer, 0, wx.ALL, 5)
        vsizer.Add(arguments_sizer, 0, wx.ALL, 5)
        vsizer.Add(func_speed_sizer, 0, wx.ALL, 5)

        hsizer = wx.BoxSizer(wx.HORIZONTAL)
        hsizer.Add(vsizer, 0, wx.ALL, 10)
        self.SetSizerAndFit(hsizer)

    def on_enable(self, evt):
        value = self.enable.GetValue()
        self.arguments.Enable(value)
        self.func_speed.Enable(value)
        Config.debug.bypass = value
        evt.Skip()

    def on_arguments(self, evt):
        Config.debug.log_args = self.arguments.GetValue()
        evt.Skip()

    def on_func_speed(self, evt):
        Config.debug.call_duration = self.func_speed.GetValue()
        evt.Skip()


class MouseButtonCtrl(wx.Panel):
    def __init__(self, parent, config):

        self.config = config
        wx.Panel.__init__(self, parent, wx.ID_ANY, style=wx.BORDER_NONE)

        choices = ['None', 'Left', 'Middle', 'Right', 'Aux1', 'Aux2', 'Wheel']
        self.button_choice = wx.RadioBox(self, wx.ID_ANY, label='Mouse Button',
                                         choices=choices, majorDimension=3,
                                         style=wx.RA_SPECIFY_COLS)

        self.radio_mapping = {}

        for i in range(len(choices)):
            label = self.button_choice.GetString(i)
            self.radio_mapping[label] = i

        self.reverse_x = wx.CheckBox(self, wx.ID_ANY, label='')
        self.reverse_y = wx.CheckBox(self, wx.ID_ANY, label='')
        self.reverse_wheel = wx.CheckBox(self, wx.ID_ANY, label='')
        self.swap_axis = wx.CheckBox(self, wx.ID_ANY, label='')

        rx_sizer = HSizer(self, 'Reverse X Axis:', self.reverse_x)
        ry_sizer = HSizer(self, 'Reverse Y Axis:', self.reverse_y)
        rw_sizer = HSizer(self, 'Reverse Wheel:', self.reverse_wheel)
        swap_sizer = HSizer(self, 'Swap X and Y Axis:', self.swap_axis)

        col1_sizer = wx.BoxSizer(wx.VERTICAL)
        col1_sizer.Add(rx_sizer, 0, wx.BOTTOM, 5)
        col1_sizer.Add(ry_sizer, 0, wx.TOP, 5)

        col2_sizer = wx.BoxSizer(wx.VERTICAL)
        col2_sizer.Add(rw_sizer, 0, wx.BOTTOM, 5)
        col2_sizer.Add(swap_sizer, 0, wx.TOP, 5)

        hsizer = wx.BoxSizer(wx.HORIZONTAL)
        hsizer.Add(col1_sizer, 0, wx.RIGHT, 5)
        hsizer.Add(col2_sizer, 0, wx.LEFT, 5)

        vsizer = wx.BoxSizer(wx.VERTICAL)
        vsizer.Add(self.button_choice, 0, wx.EXPAND | wx.ALL, 5)
        vsizer.Add(hsizer, 0, wx.EXPAND | wx.ALL, 5)

        hsizer = wx.BoxSizer(wx.HORIZONTAL)
        hsizer.Add(vsizer, 0, wx.EXPAND | wx.ALL, 5)

        self.SetSizerAndFit(hsizer)

        if config.mouse & _config.MOUSE_WHEEL:
            self.button_choice.SetSelection(self.radio_mapping['Wheel'])
            self.reverse_x.Enable(False)
            self.reverse_y.Enable(False)
            self.swap_axis.Enable(False)
            self.reverse_wheel.Enable(True)

        elif config.mouse & _config.MOUSE_LEFT:
            self.button_choice.SetSelection(self.radio_mapping['Left'])
            self.reverse_x.Enable(True)
            self.reverse_y.Enable(True)
            self.swap_axis.Enable(True)
            self.reverse_wheel.Enable(False)
        elif config.mouse & _config.MOUSE_MIDDLE:
            self.button_choice.SetSelection(self.radio_mapping['Middle'])
            self.reverse_x.Enable(False)
            self.reverse_y.Enable(False)
            self.swap_axis.Enable(False)
            self.reverse_wheel.Enable(True)

        elif config.mouse & _config.MOUSE_RIGHT:
            self.button_choice.SetSelection(self.radio_mapping['Right'])
            self.reverse_x.Enable(False)
            self.reverse_y.Enable(False)
            self.swap_axis.Enable(False)
            self.reverse_wheel.Enable(True)

        elif config.mouse & _config.MOUSE_AUX1:
            self.button_choice.SetSelection(self.radio_mapping['Aux1'])
            self.reverse_x.Enable(False)
            self.reverse_y.Enable(False)
            self.swap_axis.Enable(False)
            self.reverse_wheel.Enable(True)

        elif config.mouse & _config.MOUSE_AUX2:
            self.button_choice.SetSelection(self.radio_mapping['Aux2'])
            self.reverse_x.Enable(False)
            self.reverse_y.Enable(False)
            self.swap_axis.Enable(False)
            self.reverse_wheel.Enable(True)

        elif config.mouse & _config.MOUSE_NONE:
            self.button_choice.SetSelection(self.radio_mapping['None'])
            self.reverse_x.Enable(False)
            self.reverse_y.Enable(False)
            self.swap_axis.Enable(False)
            self.reverse_wheel.Enable(False)
        else:
            raise ValueError()

        self.reverse_wheel.SetValue(bool(config.mouse & _config.MOUSE_REVERSE_WHEEL_AXIS))
        self.swap_axis.SetValue(bool(config.mouse & _config.MOUSE_SWAP_AXIS))
        self.reverse_y.SetValue(bool(config.mouse & _config.MOUSE_REVERSE_Y_AXIS))
        self.reverse_x.SetValue(bool(config.mouse & _config.MOUSE_REVERSE_X_AXIS))

        self.button_choice.Bind(wx.EVT_RADIOBOX, self.on_mouse_choice)
        self.reverse_x.Bind(wx.EVT_CHECKBOX, self.on_reverse_x)
        self.reverse_y.Bind(wx.EVT_CHECKBOX, self.on_reverse_y)
        self.reverse_wheel.Bind(wx.EVT_CHECKBOX, self.on_reverse_wheel)
        self.swap_axis.Bind(wx.EVT_CHECKBOX, self.on_swap_axis)

    def on_reverse_wheel(self, evt):
        value = self.reverse_wheel.GetValue()
        if value:
            self.config.mouse |= _config.MOUSE_REVERSE_WHEEL_AXIS
        else:
            self.config.mouse &= ~_config.MOUSE_REVERSE_WHEEL_AXIS
        evt.Skip()

    def on_reverse_x(self, evt):
        value = self.reverse_x.GetValue()
        if value:
            self.config.mouse |= _config.MOUSE_REVERSE_X_AXIS
        else:
            self.config.mouse &= ~_config.MOUSE_REVERSE_X_AXIS
        evt.Skip()

    def on_reverse_y(self, evt):
        value = self.reverse_y.GetValue()
        if value:
            self.config.mouse |= _config.MOUSE_REVERSE_Y_AXIS
        else:
            self.config.mouse &= ~_config.MOUSE_REVERSE_Y_AXIS
        evt.Skip()

    def on_swap_axis(self, evt):
        value = self.swap_axis.GetValue()
        if value:
            self.config.mouse |= _config.MOUSE_SWAP_AXIS
        else:
            self.config.mouse &= ~_config.MOUSE_SWAP_AXIS
        evt.Skip()

    def on_mouse_choice(self, evt):
        index = self.button_choice.GetSelection()
        value = self.button_choice.GetString(index)

        if value == 'None':
            self.reverse_x.SetValue(False)
            self.reverse_y.SetValue(False)
            self.swap_axis.SetValue(False)
            self.reverse_wheel.SetValue(False)
            self.reverse_x.Enable(False)
            self.reverse_y.Enable(False)
            self.swap_axis.Enable(False)
            self.reverse_wheel.Enable(False)
        if value == 'Wheel':
            self.reverse_x.SetValue(False)
            self.reverse_y.SetValue(False)
            self.swap_axis.SetValue(False)
            self.reverse_x.Enable(False)
            self.reverse_y.Enable(False)
            self.swap_axis.Enable(False)
            self.reverse_wheel.Enable(True)
        else:
            self.reverse_wheel.SetValue(False)
            self.reverse_x.Enable(True)
            self.reverse_y.Enable(True)
            self.swap_axis.Enable(True)
            self.reverse_wheel.Enable(False)
        mapping = dict(
            Left=_config.MOUSE_LEFT,
            Middle=_config.MOUSE_MIDDLE,
            Right=_config.MOUSE_RIGHT,
            Aux1=_config.MOUSE_AUX1,
            Aux2=_config.MOUSE_AUX2,
            Wheel=_config.MOUSE_WHEEL
        )

        self.config.mouse = mapping[value]

        if self.reverse_wheel.GetValue():
            self.config.mouse |= _config.MOUSE_REVERSE_WHEEL_AXIS
        if self.reverse_x.GetValue():
            self.config.mouse |= _config.MOUSE_REVERSE_X_AXIS
        if self.reverse_y.GetValue():
            self.config.mouse |= _config.MOUSE_REVERSE_Y_AXIS
        if self.swap_axis.GetValue():
            self.config.mouse |= _config.MOUSE_SWAP_AXIS

        evt.Skip()


SPECIAL_KEYS = {
    wx.WXK_UP: '{UP}',
    wx.WXK_NUMPAD_UP: '{NUMPAD_UP}',
    wx.WXK_DOWN: '{DOWN}',
    wx.WXK_NUMPAD_DOWN: '{NUMPAD_DOWN}',
    wx.WXK_LEFT: '{LEFT}',
    wx.WXK_NUMPAD_LEFT: '{NUMPAD_LEFT}',
    wx.WXK_RIGHT: '{RIGHT}',
    wx.WXK_NUMPAD_RIGHT: '{NUMPAD_RIGHT}',
    wx.WXK_SUBTRACT: '{SUBTRACT}',
    wx.WXK_NUMPAD_SUBTRACT: '{NUMPAD_SUBTRACT}',
    wx.WXK_ADD: '{ADD}',
    wx.WXK_NUMPAD_ADD: '{NUMPAD_ADD}',
    wx.WXK_DIVIDE: '{DIVIDE}',
    wx.WXK_NUMPAD_DIVIDE: '{NUMPAD_DIVIDE}',
    wx.WXK_MULTIPLY: '{MULTIPLY}',
    wx.WXK_NUMPAD_MULTIPLY: '{NUMPAD_MULTIPLY}',
    wx.WXK_DECIMAL: '{DECIMAL}',
    wx.WXK_NUMPAD_DECIMAL: '{NUMPAD_DECIMAL}',
    wx.WXK_SEPARATOR: '{SEPARATOR}',
    wx.WXK_NUMPAD_SEPARATOR: '{NUMPAD_SEPARATOR}',
    wx.WXK_SPACE: '{SPACE}',
    wx.WXK_NUMPAD_SPACE: '{NUMPAD_SPACE}',
    wx.WXK_NUMPAD_EQUAL: '{NUMPAD_EQUAL}',
    wx.WXK_HOME: '{HOME}',
    wx.WXK_NUMPAD_HOME: '{NUMPAD_HOME}',
    wx.WXK_END: '{END}',
    wx.WXK_NUMPAD_END: '{NUMPAD_END}',
    wx.WXK_PAGEUP: '{PAGEUP}',
    wx.WXK_NUMPAD_PAGEUP: '{NUMPAD_PAGEUP}',
    wx.WXK_PAGEDOWN: '{PAGEDOWN}',
    wx.WXK_NUMPAD_PAGEDOWN: '{NUMPAD_PAGEDOWN}',
    wx.WXK_RETURN: '{RETURN}',
    wx.WXK_NUMPAD_ENTER: '{NUMPAD_ENTER}',
    wx.WXK_INSERT: '{INSERT}',
    wx.WXK_NUMPAD_INSERT: '{NUMPAD_INSERT}',
    wx.WXK_TAB: '{TAB}',
    wx.WXK_NUMPAD_TAB: '{NUMPAD_TAB}',
    wx.WXK_DELETE: '{DELETE}',
    wx.WXK_NUMPAD_DELETE: '{NUMPAD_DELETE}',
    wx.WXK_NUMPAD0: '{NUMPAD0}',
    wx.WXK_NUMPAD1: '{NUMPAD1}',
    wx.WXK_NUMPAD2: '{NUMPAD2}',
    wx.WXK_NUMPAD3: '{NUMPAD3}',
    wx.WXK_NUMPAD4: '{NUMPAD4}',
    wx.WXK_NUMPAD5: '{NUMPAD5}',
    wx.WXK_NUMPAD6: '{NUMPAD6}',
    wx.WXK_NUMPAD7: '{NUMPAD7}',
    wx.WXK_NUMPAD8: '{NUMPAD8}',
    wx.WXK_NUMPAD9: '{NUMPAD9}'
}


class KeyCtrl(wx.Panel):

    def __init__(self, parent, label, value):
        wx.Panel.__init__(self, parent, wx.ID_ANY, style=wx.BORDER_NONE)

        if value in SPECIAL_KEYS:
            value = SPECIAL_KEYS[value]
        elif 31 < value < 127:
            value = chr(value)
        else:
            raise ValueError

        self.ctrl = wx.TextCtrl(self, wx.ID_ANY, value=value, size=(150, -1))
        hsizer = HSizer(self, label, self.ctrl)
        vsizer = wx.BoxSizer(wx.VERTICAL)
        vsizer.Add(hsizer, 0, wx.EXPAND)

        self.SetSizer(vsizer)

        self.ctrl.Bind(wx.EVT_CHAR_HOOK, self.on_text)

    def Bind(self, *args, **kwargs):
        self.ctrl.Bind(*args, **kwargs)

    def on_text(self, evt: wx.KeyEvent):
        key = evt.GetKeyCode()
        if key in SPECIAL_KEYS:
            self.ctrl.SetValue(SPECIAL_KEYS[key])

            event = wx.KeyEvent(wx.wxEVT_KEY_DOWN)
            event.SetId(self.ctrl.GetId())
            event.SetEventObject(self.ctrl)
            event.SetKeyCode(key)
            self.ctrl.GetEventHandler().ProcessEvent(event)

        elif 31 < key < 127:
            self.ctrl.SetValue(chr(key).lower())
            event = wx.KeyEvent(wx.wxEVT_KEY_DOWN)
            event.SetId(self.ctrl.GetId())
            event.SetEventObject(self.ctrl)
            event.SetKeyCode(key)
            self.ctrl.GetEventHandler().ProcessEvent(event)
        else:
            event = wx.KeyEvent(wx.wxEVT_KEY_DOWN)
            event.SetId(self.ctrl.GetId())
            event.SetEventObject(self.ctrl)
            event.SetKeyCode(None)
            self.ctrl.GetEventHandler().ProcessEvent(event)
            self.ctrl.SetValue('')


class ResetTab(scrolledpanel.ScrolledPanel):

    def __init__(self, parent):

        scrolledpanel.ScrolledPanel.__init__(self, parent, wx.ID_ANY, style=wx.BORDER_NONE)

        self.mouse = MouseButtonCtrl(self, Config.reset)
        self.key = KeyCtrl(self, 'Key:', Config.reset.key)

        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(self.mouse, 0)
        sizer.Add(self.key, 0)

        self.SetSizer(sizer)

        self.key.Bind(wx.EVT_KEY_DOWN, self.on_key)

    @staticmethod
    def on_key(evt: wx.KeyEvent):
        Config.reset.key = evt.GetKeyCode()
        evt.Skip()


class ZoomTab(scrolledpanel.ScrolledPanel):

    def __init__(self, parent):

        scrolledpanel.ScrolledPanel.__init__(self, parent, wx.ID_ANY, style=wx.BORDER_NONE)

        self.mouse = MouseButtonCtrl(self, Config.zoom)
        self.in_key = KeyCtrl(self, 'Zoom In Key:', Config.zoom.in_key)
        self.out_key = KeyCtrl(self, 'Zoom Out Key:', Config.zoom.out_key)
        self.sensitivity = wx.SpinCtrlDouble(self, wx.ID_ANY,
                                             value=str(round(Config.zoom.sensitivity, 1)),
                                             min=0.1, max=50.0, inc=0.1,
                                             initial=round(Config.zoom.sensitivity, 2))

        sens_sizer = HSizer(self, 'Sensitivity:', self.sensitivity)

        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(self.mouse, 0)
        sizer.Add(self.in_key, 0, wx.ALL | 5)
        sizer.Add(self.out_key, 0, wx.ALL | 5)
        sizer.Add(sens_sizer, 0, wx.ALL | 5)

        self.SetSizer(sizer)

        self.in_key.Bind(wx.EVT_KEY_DOWN, self.on_in_key)
        self.out_key.Bind(wx.EVT_KEY_DOWN, self.on_out_key)
        self.sensitivity.Bind(wx.EVT_SPINCTRLDOUBLE, self.on_sensitivity)

    def on_sensitivity(self, evt):
        Config.zoom.sensitivity = self.sensitivity.GetValue()
        evt.Skip()

    @staticmethod
    def on_in_key(evt: wx.KeyEvent):
        Config.zoom.in_key = evt.GetKeyCode()
        evt.Skip()

    @staticmethod
    def on_out_key(evt: wx.KeyEvent):
        Config.zoom.out_key = evt.GetKeyCode()
        evt.Skip()


class WalkTab(scrolledpanel.ScrolledPanel):

    def __init__(self, parent):

        scrolledpanel.ScrolledPanel.__init__(self, parent, wx.ID_ANY, style=wx.BORDER_NONE)

        self.mouse = MouseButtonCtrl(self, Config.walk)
        self.forward_key = KeyCtrl(self, 'Forward Key:', Config.walk.forward_key)
        self.backward_key = KeyCtrl(self, 'Backward Key:', Config.walk.backward_key)
        self.left_key = KeyCtrl(self, 'Left Key:', Config.walk.left_key)
        self.right_key = KeyCtrl(self, 'Right Key:', Config.walk.right_key)

        self.sensitivity = wx.SpinCtrlDouble(self, wx.ID_ANY,
                                             value=str(round(Config.walk.sensitivity, 1)),
                                             min=0.1, max=50.0, inc=0.1,
                                             initial=round(Config.walk.sensitivity, 2))

        self.speed = wx.SpinCtrlDouble(self, wx.ID_ANY,
                                       value=str(round(Config.walk.speed, 1)),
                                       min=0.1, max=50.0, inc=0.1,
                                       initial=round(Config.walk.speed, 2))

        sens_sizer = HSizer(self, 'Sensitivity:', self.sensitivity)
        speed_sizer = HSizer(self, 'Speed:', self.sensitivity)

        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(self.mouse, 0)
        sizer.Add(self.forward_key, 0, wx.ALL | 5)
        sizer.Add(self.backward_key, 0, wx.ALL | 5)
        sizer.Add(self.left_key, 0, wx.ALL | 5)
        sizer.Add(self.right_key, 0, wx.ALL | 5)
        sizer.Add(sens_sizer, 0, wx.ALL | 5)
        sizer.Add(speed_sizer, 0, wx.ALL | 5)
        self.SetSizer(sizer)

        self.forward_key.Bind(wx.EVT_KEY_DOWN, self.on_forward_key)
        self.backward_key.Bind(wx.EVT_KEY_DOWN, self.on_backward_key)
        self.left_key.Bind(wx.EVT_KEY_DOWN, self.on_left_key)
        self.right_key.Bind(wx.EVT_KEY_DOWN, self.on_right_key)

        self.sensitivity.Bind(wx.EVT_SPINCTRLDOUBLE, self.on_sensitivity)
        self.speed.Bind(wx.EVT_SPINCTRLDOUBLE, self.on_speed)

    def on_sensitivity(self, evt):
        Config.walk.sensitivity = self.sensitivity.GetValue()
        evt.Skip()

    def on_speed(self, evt):
        Config.walk.speed = self.speed.GetValue()
        evt.Skip()

    @staticmethod
    def on_forward_key(evt: wx.KeyEvent):
        Config.walk.forward_key = evt.GetKeyCode()
        evt.Skip()

    @staticmethod
    def on_backward_key(evt: wx.KeyEvent):
        Config.walk.backward_key = evt.GetKeyCode()
        evt.Skip()

    @staticmethod
    def on_left_key(evt: wx.KeyEvent):
        Config.walk.left_key = evt.GetKeyCode()
        evt.Skip()

    @staticmethod
    def on_right_key(evt: wx.KeyEvent):
        Config.walk.right_key = evt.GetKeyCode()
        evt.Skip()


class TruckPedestalTab(scrolledpanel.ScrolledPanel):

    def __init__(self, parent):

        scrolledpanel.ScrolledPanel.__init__(self, parent, wx.ID_ANY, style=wx.BORDER_NONE)

        self.mouse = MouseButtonCtrl(self, Config.truck_pedestal)
        self.up_key = KeyCtrl(self, 'Up Key:', Config.truck_pedestal.up_key)
        self.down_key = KeyCtrl(self, 'Down Key:', Config.truck_pedestal.down_key)
        self.left_key = KeyCtrl(self, 'Left Key:', Config.truck_pedestal.left_key)
        self.right_key = KeyCtrl(self, 'Right Key:', Config.truck_pedestal.right_key)

        self.sensitivity = wx.SpinCtrlDouble(self, wx.ID_ANY,
                                             value=str(round(Config.truck_pedestal.sensitivity, 1)),
                                             min=0.1, max=50.0, inc=0.1,
                                             initial=round(Config.truck_pedestal.sensitivity, 2))

        sens_sizer = HSizer(self, 'Sensitivity:', self.sensitivity)

        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(self.mouse, 0)
        sizer.Add(self.up_key, 0, wx.ALL | 5)
        sizer.Add(self.down_key, 0, wx.ALL | 5)
        sizer.Add(self.left_key, 0, wx.ALL | 5)
        sizer.Add(self.right_key, 0, wx.ALL | 5)
        sizer.Add(sens_sizer, 0, wx.ALL | 5)
        self.SetSizer(sizer)

        self.up_key.Bind(wx.EVT_KEY_DOWN, self.on_up_key)
        self.down_key.Bind(wx.EVT_KEY_DOWN, self.on_down_key)
        self.left_key.Bind(wx.EVT_KEY_DOWN, self.on_left_key)
        self.right_key.Bind(wx.EVT_KEY_DOWN, self.on_right_key)

        self.sensitivity.Bind(wx.EVT_SPINCTRLDOUBLE, self.on_sensitivity)

    def on_sensitivity(self, evt):
        Config.truck_pedestal.sensitivity = self.sensitivity.GetValue()
        evt.Skip()

    @staticmethod
    def on_up_key(evt: wx.KeyEvent):
        Config.truck_pedestal.up_key = evt.GetKeyCode()
        evt.Skip()

    @staticmethod
    def on_down_key(evt: wx.KeyEvent):
        Config.truck_pedestal.down_key = evt.GetKeyCode()
        evt.Skip()

    @staticmethod
    def on_left_key(evt: wx.KeyEvent):
        Config.truck_pedestal.left_key = evt.GetKeyCode()
        evt.Skip()

    @staticmethod
    def on_right_key(evt: wx.KeyEvent):
        Config.truck_pedestal.right_key = evt.GetKeyCode()
        evt.Skip()


class PanTiltTab(scrolledpanel.ScrolledPanel):

    def __init__(self, parent):

        scrolledpanel.ScrolledPanel.__init__(self, parent, wx.ID_ANY, style=wx.BORDER_NONE)

        self.mouse = MouseButtonCtrl(self, Config.pan_tilt)
        self.up_key = KeyCtrl(self, 'Up Key:', Config.pan_tilt.up_key)
        self.down_key = KeyCtrl(self, 'Down Key:', Config.pan_tilt.down_key)
        self.left_key = KeyCtrl(self, 'Left Key:', Config.pan_tilt.left_key)
        self.right_key = KeyCtrl(self, 'Right Key:', Config.pan_tilt.right_key)

        self.sensitivity = wx.SpinCtrlDouble(self, wx.ID_ANY,
                                             value=str(round(Config.pan_tilt.sensitivity, 1)),
                                             min=0.1, max=50.0, inc=0.1,
                                             initial=round(Config.pan_tilt.sensitivity, 2))

        sens_sizer = HSizer(self, 'Sensitivity:', self.sensitivity)

        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(self.mouse, 0)
        sizer.Add(self.up_key, 0, wx.ALL | 5)
        sizer.Add(self.down_key, 0, wx.ALL | 5)
        sizer.Add(self.left_key, 0, wx.ALL | 5)
        sizer.Add(self.right_key, 0, wx.ALL | 5)
        sizer.Add(sens_sizer, 0, wx.ALL | 5)
        self.SetSizer(sizer)

        self.up_key.Bind(wx.EVT_KEY_DOWN, self.on_up_key)
        self.down_key.Bind(wx.EVT_KEY_DOWN, self.on_down_key)
        self.left_key.Bind(wx.EVT_KEY_DOWN, self.on_left_key)
        self.right_key.Bind(wx.EVT_KEY_DOWN, self.on_right_key)

        self.sensitivity.Bind(wx.EVT_SPINCTRLDOUBLE, self.on_sensitivity)

    def on_sensitivity(self, evt):
        Config.pan_tilt.sensitivity = self.sensitivity.GetValue()
        evt.Skip()

    @staticmethod
    def on_up_key(evt: wx.KeyEvent):
        Config.pan_tilt.up_key = evt.GetKeyCode()
        evt.Skip()

    @staticmethod
    def on_down_key(evt: wx.KeyEvent):
        Config.pan_tilt.down_key = evt.GetKeyCode()
        evt.Skip()

    @staticmethod
    def on_left_key(evt: wx.KeyEvent):
        Config.pan_tilt.left_key = evt.GetKeyCode()
        evt.Skip()

    @staticmethod
    def on_right_key(evt: wx.KeyEvent):
        Config.pan_tilt.right_key = evt.GetKeyCode()
        evt.Skip()


class RotateTab(scrolledpanel.ScrolledPanel):

    def __init__(self, parent):

        scrolledpanel.ScrolledPanel.__init__(self, parent, wx.ID_ANY, style=wx.BORDER_NONE)

        self.mouse = MouseButtonCtrl(self, Config.rotate)
        self.up_key = KeyCtrl(self, 'Up Key:', Config.rotate.up_key)
        self.down_key = KeyCtrl(self, 'Down Key:', Config.rotate.down_key)
        self.left_key = KeyCtrl(self, 'Left Key:', Config.rotate.left_key)
        self.right_key = KeyCtrl(self, 'Right Key:', Config.rotate.right_key)

        self.sensitivity = wx.SpinCtrlDouble(self, wx.ID_ANY,
                                             value=str(round(Config.rotate.sensitivity, 1)),
                                             min=0.1, max=50.0, inc=0.1,
                                             initial=round(Config.rotate.sensitivity, 2))

        sens_sizer = HSizer(self, 'Sensitivity:', self.sensitivity)

        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(self.mouse, 0)
        sizer.Add(self.up_key, 0, wx.ALL | 5)
        sizer.Add(self.down_key, 0, wx.ALL | 5)
        sizer.Add(self.left_key, 0, wx.ALL | 5)
        sizer.Add(self.right_key, 0, wx.ALL | 5)
        sizer.Add(sens_sizer, 0, wx.ALL | 5)
        self.SetSizer(sizer)

        self.up_key.Bind(wx.EVT_KEY_DOWN, self.on_up_key)
        self.down_key.Bind(wx.EVT_KEY_DOWN, self.on_down_key)
        self.left_key.Bind(wx.EVT_KEY_DOWN, self.on_left_key)
        self.right_key.Bind(wx.EVT_KEY_DOWN, self.on_right_key)

        self.sensitivity.Bind(wx.EVT_SPINCTRLDOUBLE, self.on_sensitivity)

    def on_sensitivity(self, evt):
        Config.rotate.sensitivity = self.sensitivity.GetValue()
        evt.Skip()

    @staticmethod
    def on_up_key(evt: wx.KeyEvent):
        Config.rotate.up_key = evt.GetKeyCode()
        evt.Skip()

    @staticmethod
    def on_down_key(evt: wx.KeyEvent):
        Config.rotate.down_key = evt.GetKeyCode()
        evt.Skip()

    @staticmethod
    def on_left_key(evt: wx.KeyEvent):
        Config.rotate.left_key = evt.GetKeyCode()
        evt.Skip()

    @staticmethod
    def on_right_key(evt: wx.KeyEvent):
        Config.rotate.right_key = evt.GetKeyCode()
        evt.Skip()


class KeyboardInputTab(scrolledpanel.ScrolledPanel):

    def __init__(self, parent):
        scrolledpanel.ScrolledPanel.__init__(self, parent, wx.ID_ANY, style=wx.BORDER_NONE)

        self.max_repeat_factor = wx.SpinCtrlDouble(
            self, wx.ID_ANY, value=str(round(Config.keyboard_settings.max_speed_factor, 1)),
            min=0.1, max=50.0, inc=0.1, initial=round(Config.keyboard_settings.max_speed_factor, 1))

        self.min_repeat_factor = wx.SpinCtrlDouble(
            self, wx.ID_ANY, value=str(round(Config.keyboard_settings.start_speed_factor, 1)),
            min=0.1, max=50.0, inc=0.1, initial=round(Config.keyboard_settings.start_speed_factor, 1))

        self.repeat_increment = wx.SpinCtrlDouble(
            self, wx.ID_ANY, value=str(round(Config.keyboard_settings.speed_factor_increment, 1)),
            min=0.1, max=50.0, inc=0.1, initial=round(Config.keyboard_settings.speed_factor_increment, 1))

        max_sizer = HSizer(self, 'Max Repeat Factor:', self.max_repeat_factor)
        min_sizer = HSizer(self, 'Min Repeat Factor:', self.min_repeat_factor)
        inc_sizer = HSizer(self, 'Repeat Factor Increment:', self.repeat_increment)

        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(min_sizer, 0, wx.ALL | 5)
        sizer.Add(max_sizer, 0, wx.ALL | 5)
        sizer.Add(inc_sizer, 0, wx.ALL | 5)

        self.SetSizer(sizer)

        self.max_repeat_factor.Bind(wx.EVT_SPINCTRLDOUBLE, self.on_max)
        self.min_repeat_factor.Bind(wx.EVT_SPINCTRLDOUBLE, self.on_min)
        self.repeat_increment.Bind(wx.EVT_SPINCTRLDOUBLE, self.on_inc)

    def on_min(self, evt):
        Config.keyboard_settings.start_speed_factor = self.min_repeat_factor.GetValue()
        evt.Skip()

    def on_max(self, evt):
        Config.keyboard_settings.max_speed_factor = self.max_repeat_factor.GetValue()
        evt.Skip()

    def on_inc(self, evt):
        Config.keyboard_settings.speed_factor_increment = self.repeat_increment.GetValue()
        evt.Skip()


class VirtualCanvasTab(scrolledpanel.ScrolledPanel):

    def __init__(self, parent):
        scrolledpanel.ScrolledPanel.__init__(self, parent, wx.ID_ANY, style=wx.BORDER_NONE)

        self.width = wx.SpinCtrl(
            self, wx.ID_ANY, value=str(Config.virtual_canvas.width),
            min=300, max=9999999, initial=Config.virtual_canvas.width)

        self.height = wx.SpinCtrl(
            self, wx.ID_ANY, value=str(Config.virtual_canvas.height),
            min=200, max=9999999, initial=Config.virtual_canvas.height)

        width_sizer = HSizer(self, 'Width:', self.width)
        height_sizer = HSizer(self, 'Height:', self.height)

        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(width_sizer, 0, wx.ALL | 5)
        sizer.Add(height_sizer, 0, wx.ALL | 5)

        self.SetSizer(sizer)

        self.width.Bind(wx.EVT_SPINCTRLDOUBLE, self.on_width)
        self.height.Bind(wx.EVT_SPINCTRLDOUBLE, self.on_height)

    def on_width(self, evt):
        Config.virtual_canvas.width = self.width.GetValue()
        evt.Skip()

    def on_height(self, evt):
        Config.virtual_canvas.height = self.height.GetValue()
        evt.Skip()


class FloorTab(scrolledpanel.ScrolledPanel):

    def __init__(self, parent):
        scrolledpanel.ScrolledPanel.__init__(self, parent, wx.ID_ANY, style=wx.BORDER_NONE)

        show_grid = Config.floor.show_grid

        self.show_grid = wx.CheckBox(self, wx.ID_ANY, label='')
        self.show_grid.SetValue(show_grid)
        self.show_grid.Bind(wx.EVT_CHECKBOX, self.on_show_grid)

        reflections = Config.floor.reflections

        self.reflections = wx.CheckBox(self, wx.ID_ANY, label='')
        self.reflections.SetValue(reflections)
        self.reflections.Bind(wx.EVT_CHECKBOX, self.on_reflections)

        self.reflection_strength = wx.SpinCtrlDouble(
            self, wx.ID_ANY, value=str(round(Config.floor.reflection_strength, 1)),
            min=0.1, max=100.0, inc=0.1, initial=round(Config.floor.reflection_strength, 1))

        self.reflection_strength.Enable(reflections)
        self.reflection_strength.Bind(wx.EVT_SPINCTRLDOUBLE, self.on_reflection_strength)

        self.distance = wx.SpinCtrl(
            self, wx.ID_ANY, value=str(Config.floor.distance),
            min=100, max=10000, initial=Config.floor.distance)

        self.distance.Bind(wx.EVT_SPINCTRL, self.on_distance)

        self.grid_size = wx.SpinCtrl(
            self, wx.ID_ANY, value=str(Config.floor.grid_size),
            min=2, max=1000, initial=Config.floor.grid_size)

        self.grid_size.Enable(show_grid)

        self.grid_size.Bind(wx.EVT_SPINCTRL, self.on_grid_size)

        self.primary_color = ColorButton(self, Config.floor.primary_color)
        self.secondary_color = ColorButton(self, Config.floor.secondary_color)

        self.primary_color.Bind(wx.EVT_BUTTON, self.on_primary_color)
        self.secondary_color.Bind(wx.EVT_BUTTON, self.on_secondary_color)

        self.secondary_color.Enable(show_grid)

        self.ground_height = wx.SpinCtrlDouble(
            self, wx.ID_ANY, value=str(round(Config.floor.ground_height, 1)),
            min=-1000.0, max=1000.0, inc=0.1, initial=round(Config.floor.ground_height, 1))

        self.ground_height.Bind(wx.EVT_SPINCTRLDOUBLE, self.on_ground_height)

        ground_height_sizer = HSizer(self, 'Ground Height:', self.ground_height)
        distance_sizer = HSizer(self, 'Floor Size:', self.distance)
        primary_color_sizer = HSizer(self, 'Primary Color:', self.primary_color)
        show_grid_sizer = HSizer(self, 'Show Grid:', self.show_grid)
        grid_size_sizer = HSizer(self, 'Grid Size:', self.grid_size)
        secondary_color_sizer = HSizer(self, 'Secondary Color:', self.secondary_color)
        reflections_sizer = HSizer(self, 'Reflections:', self.reflections)
        reflection_strength_sizer = HSizer(self, 'Reflection Strength:', self.reflection_strength)

        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(ground_height_sizer, 0, wx.ALL | 5)
        sizer.Add(distance_sizer, 0, wx.ALL | 5)
        sizer.Add(primary_color_sizer, 0, wx.ALL | 5)
        sizer.Add(show_grid_sizer, 0, wx.ALL | 5)
        sizer.Add(grid_size_sizer, 0, wx.ALL | 5)
        sizer.Add(secondary_color_sizer, 0, wx.ALL | 5)
        sizer.Add(reflections_sizer, 0, wx.ALL | 5)
        sizer.Add(reflection_strength_sizer, 0, wx.ALL | 5)

        self.SetSizer(sizer)

    def on_show_grid(self, evt):
        value = self.show_grid.GetValue()
        self.grid_size.Enable(value)
        self.secondary_color.Enable(value)
        Config.floor.show_grid = value
        evt.Skip()

    def on_reflections(self, evt):
        value = self.reflections.GetValue()
        self.reflection_strength.Enable(value)
        Config.floor.reflections = value
        evt.Skip()

    def on_primary_color(self, evt):
        color_data = wx.ColourData()
        color_data.FromString(Config.colors.custom_colors)

        color = wx.Colour(*[int(item * 255) for item in Config.floor.primary_color])
        color_data.SetColour(color)
        color_data.SetChooseAlpha(True)

        dialog = cubecolourdialog.CubeColourDialog(self, color_data)
        dialog.ShowModal()
        color_data = dialog.GetColourData()
        new_color = color_data.GetColour()
        color_data.SetColour(wx.Colour(255, 255, 255, 255))

        Config.colors.custom_colors = color_data.ToString()

        Config.floor.primary_color = [item / 255.0 for item in
                                      (new_color.Red(), new_color.Green(),
                                       new_color.Blue(), new_color.GetAlpha())]

        self.primary_color.SetBackgroundColour(new_color)
        evt.Skip()

    def on_secondary_color(self, evt):
        color_data = wx.ColourData()
        color_data.FromString(Config.colors.custom_colors)

        color = wx.Colour(*[int(item * 255) for item in Config.floor.secondary_color])
        color_data.SetColour(color)
        color_data.SetChooseAlpha(True)

        dialog = cubecolourdialog.CubeColourDialog(self, color_data)
        dialog.ShowModal()
        color_data = dialog.GetColourData()
        new_color = color_data.GetColour()
        color_data.SetColour(wx.Colour(255, 255, 255, 255))

        Config.colors.custom_colors = color_data.ToString()

        Config.floor.secondary_color = [item / 255.0 for item in
                                        (new_color.Red(), new_color.Green(),
                                         new_color.Blue(), new_color.GetAlpha())]

        self.secondary_color.SetBackgroundColour(new_color)
        evt.Skip()

    def on_reflection_strength(self, evt):
        Config.floor.reflection_strength = self.reflection_strength.GetValue()
        evt.Skip()

    def on_grid_size(self, evt):
        Config.floor.grid_size = self.grid_size.GetValue()
        evt.Skip()

    def on_ground_height(self, evt):
        Config.floor.ground_height = self.ground_height.GetValue()
        evt.Skip()

    def on_distance(self, evt):
        Config.floor.distance = self.distance.GetValue()
        evt.Skip()


class CameraTab(scrolledpanel.ScrolledPanel):

    def __init__(self, parent):
        scrolledpanel.ScrolledPanel.__init__(self, parent, wx.ID_ANY, style=wx.BORDER_NONE)

        focal_target_visible = Config.camera.focal_target_visible

        self.focal_target_visible = wx.CheckBox(self, wx.ID_ANY, label='')
        self.focal_target_visible.SetValue(focal_target_visible)
        self.focal_target_visible.Bind(wx.EVT_CHECKBOX, self.on_focal_target_visible)

        self.focal_target_radius = wx.SpinCtrlDouble(
            self, wx.ID_ANY, value=str(round(Config.camera.focal_target_radius, 2)),
            min=0.25, max=5.0, inc=0.05, initial=round(Config.camera.focal_target_radius, 2))

        self.focal_target_radius.Enable(focal_target_visible)
        self.focal_target_radius.Bind(wx.EVT_SPINCTRLDOUBLE, self.on_focal_target_radius)

        self.focal_target_color = ColorButton(self, Config.camera.focal_target_color)
        self.focal_target_color.Bind(wx.EVT_BUTTON, self.on_focal_target_color)

        self.focal_target_color.Enable(focal_target_visible)

        focal_sizer = HSizer(self, 'Enable Focal Target:', self.focal_target_visible)
        focal_radius_sizer = HSizer(self, 'Focal Target Radius:', self.focal_target_radius)
        focal_color_sizer = HSizer(self, 'Focal Target Color:', self.focal_target_color)

        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(focal_sizer, 0, wx.ALL | 5)
        sizer.Add(focal_radius_sizer, 0, wx.ALL | 5)
        sizer.Add(focal_color_sizer, 0, wx.ALL | 5)

        self.SetSizer(sizer)

    def on_focal_target_visible(self, evt):
        value = self.focal_target_visible.GetValue()
        self.focal_target_radius.Enable(value)
        self.focal_target_color.Enable(value)
        Config.camera.focal_target_visible = value
        evt.Skip()

    def on_focal_target_color(self, evt):
        color_data = wx.ColourData()
        color_data.FromString(Config.colors.custom_colors)

        color = wx.Colour(*[int(item * 255) for item in Config.camera.focal_target_color])
        color_data.SetColour(color)
        color_data.SetChooseAlpha(True)

        dialog = cubecolourdialog.CubeColourDialog(self, color_data)
        dialog.ShowModal()
        color_data = dialog.GetColourData()
        new_color = color_data.GetColour()
        color_data.SetColour(wx.Colour(255, 255, 255, 255))

        Config.colors.custom_colors = color_data.ToString()

        Config.camera.focal_target_color = [item / 255.0 for item in
                                            (new_color.Red(), new_color.Green(),
                                             new_color.Blue(), new_color.GetAlpha())]

        self.focal_target_color.SetBackgroundColour(new_color)
        evt.Skip()

    def on_focal_target_radius(self, evt):
        Config.camera.focal_target_radius = self.focal_target_radius.GetValue()
        evt.Skip()


class ConfigPanel(wx.Notebook):

    def __init__(self, parent, id=wx.ID_ANY, size=wx.DefaultSize,  # NOQA
                 pos=wx.DefaultPosition,
                 style=wx.NB_TOP | wx.NB_FIXEDWIDTH | wx.NB_MULTILINE):

        wx.Notebook.__init__(self, parent, id, size=size, pos=pos, style=style)

        if not Config.colors.custom_colors:
            color_data = wx.ColourData()

            for i in range(12):
                color_data.SetCustomColour(i, wx.Colour(255, 255, 255, 255))

            color_data.SetColour(wx.Colour(255, 255, 255, 255))

            Config.colors.custom_colors = color_data.ToString()

        self.keyboard_input = KeyboardInputTab(self)
        self.walk = WalkTab(self)
        self.zoom = ZoomTab(self)
        self.rotate = RotateTab(self)
        self.pan_tilt = PanTiltTab(self)
        self.truck_pedestal = TruckPedestalTab(self)
        self.reset = ResetTab(self)
        self.camera = CameraTab(self)
        self.headlight = HeadlightTab(self)
        self.floor = FloorTab(self)
        self.virtual_canvas = VirtualCanvasTab(self)
        self.debug = DebugTab(self)

        self.AddPage(self.keyboard_input, 'Keyboard Input')
        self.AddPage(self.walk, 'Walk')
        self.AddPage(self.zoom, 'Zoom')
        self.AddPage(self.rotate, 'Rotate')
        self.AddPage(self.pan_tilt, 'Pan/Tilt')
        self.AddPage(self.truck_pedestal, 'Truck/Pedestal')
        self.AddPage(self.reset, 'Reset View')
        self.AddPage(self.camera, 'Camera')
        self.AddPage(self.headlight, 'Headlight')
        self.AddPage(self.floor, 'Floor')
        self.AddPage(self.virtual_canvas, 'Virtual Canvas')
        self.AddPage(self.debug, 'Debug')
