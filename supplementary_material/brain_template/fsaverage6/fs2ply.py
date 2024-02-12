import nibabel as nib
import trimesh
(v, f) = nib.freesurfer.io.read_geometry('lh.white')
trimesh.Trimesh(v, f, process=False).export('lh_white.ply')
(v, f) = nib.freesurfer.io.read_geometry('rh.white')
trimesh.Trimesh(v, f, process=False).export('rh_white.ply')
(v, f) = nib.freesurfer.io.read_geometry('lh.pial')
trimesh.Trimesh(v, f, process=False).export('lh_pial.ply')
(v, f) = nib.freesurfer.io.read_geometry('rh.pial')
trimesh.Trimesh(v, f, process=False).export('rh_pial.ply')
