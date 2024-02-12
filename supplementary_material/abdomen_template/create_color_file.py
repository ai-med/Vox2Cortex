import trimesh
import numpy as np

meshes = [
    'avg_template_abdomenct1k_registered/registered_liver.ply',
    'avg_template_abdomenct1k_registered/registered_kidney_left.ply',
    'avg_template_abdomenct1k_registered/registered_kidney_right.ply',
    'avg_template_abdomenct1k_registered/registered_spleen.ply',
    'avg_template_abdomenct1k_registered/registered_pancreas.ply'
]
out_dir = "avg_template_abdomenct1k_registered/"

for mn in meshes:
    m = trimesh.load(mn, process=False)
    v, f = m.vertices, m.faces
    v = v - v.mean(axis=0)

    # Cartesian -> angular coordinates
    xy = v[:, 0] ** 2 + v[:, 1] ** 2
    r = np.sqrt(xy + v[:, 2] ** 2)
    theta = np.arctan2(np.sqrt(xy), v[:, 2])
    phi = np.arctan2(v[:, 1], v[:, 0])

    v_color = 10 * theta/(2 * np.pi) + phi/(2 * np.pi)
    np.save(mn.replace(".ply", "_color.npy"), v_color)
