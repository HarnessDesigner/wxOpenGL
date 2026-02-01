import wx
import wxOpenGL
import threading
import time

material = wxOpenGL.PlasticMaterial([0.4, 0.4, 0.4, 1.0])
selected_material = wxOpenGL.PlasticMaterial([1.0, 0.5, 0.5, 1.0])
point = wxOpenGL.Point(0, 0, 0)
angle = wxOpenGL.Angle.from_euler(0.0, 0, 0)

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
        self.model = wxOpenGL.MeshModel(self.canvas, material, selected_material, True, r'examples/final_asm_asm.stp', point, angle)


class App(wx.App):

    def OnInit(self):
        self.frame = Frame()
        self.frame.Show()

        return True


app = App()
app.MainLoop()

