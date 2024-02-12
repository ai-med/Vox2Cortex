import trimesh
import numpy as np
import nibabel.freesurfer.io as fsio

name = "/mnt/nas/Software/Freesurfer70/subjects/fsaverage/surf/rh.sphere"
out_file = "fsaverage/rh_color.npy"

v, f = fsio.read_geometry(name)
v = v / 100

# Cartesian -> angular coordinates
xy = v[:, 0] ** 2 + v[:, 1] ** 2
r = np.sqrt(xy + v[:, 2] ** 2)
theta = np.arctan2(np.sqrt(xy), v[:, 2])
phi = np.arctan2(v[:, 1], v[:, 0])

v_color = 10 * theta/(2 * np.pi) + phi/(2 * np.pi)
np.save(out_file, v_color)
