from OpenGL import GL
from . import utils as _utils


class GLMaterial:

    _ambient = (0.0, 0.0, 0.0)

    # color is the "color" we tend to think of, tends to be white for metals
    _diffuse = (0.0, 0.0, 0.0)

    # plastics white, metals darker color
    _specular = (0.0, 0.0, 0.0)

    # polished metals has the highest shine, rubber type materials will have a rally low shine. plastics are in between
    _shine = 0.0  # 0.0 to 128.0

    def __init__(self, color):
        self._color = color
        self._saved_emission = []
        self._saved_ambient = []
        self._saved_diffuse = []
        self._saved_specular = []
        self._saved_shine = []

        self.x_ray = False
        self.x_ray_color = [0.2, 0.2, 1.0, 0.35]

    @property
    def is_opaque(self):
        if self.x_ray:
            return False

        return self._color[-1] == 1.0

    def set(self):
        self._saved_emission = GL.glGetMaterialfv(GL.GL_FRONT, GL.GL_EMISSION)
        self._saved_ambient = GL.glGetMaterialfv(GL.GL_FRONT, GL.GL_AMBIENT)
        self._saved_diffuse = GL.glGetMaterialfv(GL.GL_FRONT, GL.GL_DIFFUSE)
        self._saved_specular = GL.glGetMaterialfv(GL.GL_FRONT, GL.GL_SPECULAR)
        self._saved_shine = GL.glGetMaterialfv(GL.GL_FRONT, GL.GL_SHININESS)

        a = tuple(self._color[:-1])

        if self.x_ray:
            GL.glColor4f(*self.x_ray_color)
            GL.glMaterialfv(GL.GL_FRONT, GL.GL_EMISSION, self.x_ray_color)
            GL.glMaterialfv(GL.GL_FRONT, GL.GL_AMBIENT, self.x_ray_color)
            GL.glMaterialfv(GL.GL_FRONT, GL.GL_DIFFUSE, self.x_ray_color)
            GL.glMaterialfv(GL.GL_FRONT, GL.GL_SPECULAR, self.x_ray_color)
            GL.glMaterialf(GL.GL_FRONT, GL.GL_SHININESS, 110.0)
        else:
            GL.glColor4f(*self._color)
            GL.glMaterialfv(GL.GL_FRONT, GL.GL_AMBIENT, self._ambient + a)
            GL.glMaterialfv(GL.GL_FRONT, GL.GL_DIFFUSE, self._diffuse + a)
            GL.glMaterialfv(GL.GL_FRONT, GL.GL_SPECULAR, self._specular + a)
            GL.glMaterialf(GL.GL_FRONT, GL.GL_SHININESS, self._shine)

    def unset(self):
        GL.glMaterialfv(GL.GL_FRONT, GL.GL_EMISSION, self._saved_emission)
        GL.glMaterialfv(GL.GL_FRONT, GL.GL_AMBIENT, self._saved_ambient)
        GL.glMaterialfv(GL.GL_FRONT, GL.GL_DIFFUSE, self._saved_diffuse)
        GL.glMaterialfv(GL.GL_FRONT, GL.GL_SPECULAR, self._saved_specular)
        GL.glMaterialf(GL.GL_FRONT, GL.GL_SHININESS, self._saved_shine)


class GenericMaterial(GLMaterial):
    _ambient = (0.3, 0.3, 0.3, 0.5)
    _diffuse = (0.5, 0.5, 0.5, 0.5)
    _specular = (0.8, 0.8, 0.8, 0.5)
    _shine = 100.0


class PlasticMaterial(GLMaterial):
    _ambient = (0.0, 0.0, 0.0)
    _diffuse = (0.0, 0.0, 0.0)
    _specular = (0.898039, 0.898039, 0.898039)
    _shine = 110.0

    def __init__(self, color):
        super().__init__(color)
        self._ambient = tuple(color[:-1])
        self._diffuse = tuple(color[:-1])


class BlackPlasticMaterial(GLMaterial):
    _ambient = (0.0, 0.0, 0.0)
    _diffuse = (0.01, 0.01, 0.01)
    _specular = (0.50, 0.50, 0.50)
    _shine = 32.0


class CyanPlasticMaterial(GLMaterial):
    _ambient = (0.0, 0.1, 0.06)
    _diffuse = (0.0, 0.50980392, 0.50980392)
    _specular = (0.50196078, 0.50196078, 0.50196078)
    _shine = 32.0


class GreenPlasticMaterial(GLMaterial):
    _ambient = (0.0, 0.0, 0.0)
    _diffuse = (0.1, 0.35, 0.1)
    _specular = (0.45, 0.55, 0.45)
    _shine = 32.0
          

class RedPlasticMaterial(GLMaterial):
    _ambient = (0.0, 0.0, 0.0)
    _diffuse = (0.5, 0.0, 0.0)
    _specular = (0.7, 0.6, 0.6)
    _shine = 32.0
          

class WhitePlasticMaterial(GLMaterial):
    _ambient = (0.0, 0.0, 0.0)
    _diffuse = (0.55, 0.55, 0.55)
    _specular = (0.70, 0.70, 0.70)
    _shine = 32.0
          

class YellowPlasticMaterial(GLMaterial):
    _ambient = (0.0, 0.0, 0.0)
    _diffuse = (0.5, 0.5, 0.0)
    _specular = (0.60, 0.60, 0.50)
    _shine = 32.0


class RubberMaterial(GLMaterial):
    _ambient = (0.0, 0.0, 0.0)
    _diffuse = (0.0, 0.0, 0.0)
    _specular = (0.0, 0.0, 0.0)
    _shine = 10.0

    def __init__(self, color):
        super().__init__(color)
        r, g, b, _ = color

        if r == g == b == 0.0:
            ar, ag, ab = (0.02, 0.02, 0.02)
            dr, dg, db = (0.01, 0.01, 0.01)
            sr, sg, sb = (0.4, 0.4, 0.4)
        else:
            ar = _utils.remap(r, 0.0, 1.0, 0.0, 0.05)
            ag = _utils.remap(g, 0.0, 1.0, 0.0, 0.05)
            ab = _utils.remap(b, 0.0, 1.0, 0.0, 0.05)

            dr = _utils.remap(r, 0.0, 1.0, 0.4, 0.5)
            dg = _utils.remap(g, 0.0, 1.0, 0.4, 0.5)
            db = _utils.remap(b, 0.0, 1.0, 0.4, 0.5)

            sr = _utils.remap(r, 0.0, 1.0, 0.04, 0.7)
            sg = _utils.remap(g, 0.0, 1.0, 0.04, 0.7)
            sb = _utils.remap(b, 0.0, 1.0, 0.04, 0.7)

        self._ambient = (ar, ag, ab)
        self._diffuse = (dr, dg, db)
        self._specular = (sr, sg, sb)


class MetallicMaterial(GLMaterial):
    _ambient = (0.0, 0.0, 0.0)
    _diffuse = (0.0, 0.0, 0.0)
    _specular = (0.0, 0.0, 0.0)
    _shine = 51.2

    def __init__(self, color):
        super().__init__(color)
        r, g, b, _ = color

        ar = _utils.remap(r, 0.75294, 1.0, 0.19215, 0.24705)
        ag = _utils.remap(g, 0.75294, 0.843137, 0.19215, 0.19607)
        ab = _utils.remap(b, 0.0, 0.75294, 0.07058, 0.19215)

        dr = _utils.remap(r, 0.75294, 1.0, 0.50588, 0.3451)
        dg = _utils.remap(g, 0.75294, 0.843137, 0.50588, 0.3137)
        db = _utils.remap(b, 0.0, 0.75294, 0.09019, 0.50588)

        sr = _utils.remap(r, 0.75294, 1.0, 0.50588, 0.79607)
        sg = _utils.remap(g, 0.75294, 0.843137, 0.50588, 0.72156)
        sb = _utils.remap(b, 0.0, 0.75294, 0.2078, 0.50588)

        self._ambient = (ar, ag, ab)
        self._diffuse = (dr, dg, db)
        self._specular = (sr, sg, sb)


class PolishedMaterial(GLMaterial):
    _ambient = (0.0, 0.0, 0.0)
    _diffuse = (0.0, 0.0, 0.0)
    _specular = (0.0, 0.0, 0.0)
    _shine = 0.0

    def __init__(self, color):
        super().__init__(color)
        r, g, b, _ = color

        ar = _utils.remap(r, 0.75294, 1.0, 0.22745, 0.24705)
        ag = _utils.remap(g, 0.75294, 0.843137, 0.22745, 0.22352)
        ab = _utils.remap(b, 0.0, 0.75294, 0.06274, 0.22745)

        dr = _utils.remap(r, 0.75294, 1.0, 0.27450, 0.34509)
        dg = _utils.remap(g, 0.75294, 0.843137, 0.27450, 0.31372)
        db = _utils.remap(b, 0.0, 0.75294, 0.09019, 0.27450)

        sr = _utils.remap(r, 0.75294, 1.0, 0.77254, 0.79607)
        sg = _utils.remap(g, 0.75294, 0.843137, 0.77254, 0.72156)
        sb = _utils.remap(b, 0.0, 0.75294, 0.20784, 0.77254)

        self._ambient = (ar, ag, ab)
        self._diffuse = (dr, dg, db)
        self._specular = (sr, sg, sb)

        self._shine = _utils.remap(r + g + b, 1.843137, 2.25882, 83.2, 89.6)
