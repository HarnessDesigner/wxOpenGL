import ezdxf
import numpy as np


def load(file):
    doc = ezdxf.readfile(file)
    msp = doc.modelspace()

    vertices = []
    faces = []
    vertex_map = {}

    def get_index(v):
        if v not in vertex_map:
            vertex_map[v] = len(vertices)
            vertices.append(v)
        return vertex_map[v]

    for entity in msp:
        if entity.dxftype() == "3DFACE":
            pts = [
                tuple(entity.dxf.vtx0),
                tuple(entity.dxf.vtx1),
                tuple(entity.dxf.vtx2),
                tuple(entity.dxf.vtx3),
            ]

            # Some DXF faces repeat the last vertex (triangles)
            unique = list(dict.fromkeys(pts))
            if len(unique) < 3:
                continue

            # Triangulate quad → 2 triangles
            if len(unique) == 3:
                i0, i1, i2 = [get_index(v) for v in unique]
                faces.append((i0, i1, i2))
            elif len(unique) == 4:
                i0, i1, i2, i3 = [get_index(v) for v in unique]
                faces.append((i0, i1, i2))
                faces.append((i0, i2, i3))

    return np.array(vertices, dtype=np.float64), np.array(faces, dtype=np.uint32)
