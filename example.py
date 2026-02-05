import sys

# Uncomment these to run using an installed version of the library ** PYCHARM **
# sys.path.pop(0)
# sys.path.pop(0)

# Set this to True to load upwards of 6 million triangles
STRESS_TEST = False

import wx
import wxOpenGL
import threading
import time
import random

material = wxOpenGL.PlasticMaterial([0.4, 0.4, 0.4, 1.0])
selected_material = wxOpenGL.PlasticMaterial([1.0, 0.5, 0.5, 1.0])
angle = wxOpenGL.Angle.from_euler(90.0, 0, 0)

selected_material.x_ray = True


class Frame(wx.Frame):

    def __init__(self):
        wx.Frame.__init__(self, None, wx.ID_ANY, size=(1600, 900))
        self.canvas = wxOpenGL.Canvas(self)

        vsizer = wx.BoxSizer(wx.VERTICAL)
        hsizer = wx.BoxSizer(wx.HORIZONTAL)
        hsizer.Add(self.canvas, 1, wx.EXPAND)
        vsizer.Add(hsizer, 1, wx.EXPAND)
        self.SetSizer(vsizer)
        self.model = None

        t = threading.Thread(target=self._load_model)
        t.daemon = True
        t.start()

    def _load_model(self):
        time.sleep(2)
        self.models = []

        if STRESS_TEST:
            for _ in range(100):
                point = wxOpenGL.Point(random.randrange(-1000, 1000), 0, random.randrange(-1000, 1000))
                model = wxOpenGL.MeshModel(self.canvas, material, selected_material, True, r'examples/c-045788-000-a-3d.stp', point, angle)

                wx.CallAfter(self.canvas.Refresh, False)
                self.models.append(model)
        else:
            point = wxOpenGL.Point(0, 0, 0)
            model = wxOpenGL.MeshModel(self.canvas, material, selected_material, True, r'examples/c-045788-000-a-3d.stp', point, angle)
            wx.CallAfter(self.canvas.Refresh, False)
            self.models.append(model)


class App(wx.App):

    def OnInit(self):
        self.frame = Frame()
        self.frame.Show()

        return True


app = App()
app.MainLoop()

