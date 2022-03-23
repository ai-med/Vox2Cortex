'''
pre-processing script for Vox2Cortex
This script is adapted from DeepCSR https://bitbucket.csiro.au/projects/CRCPMAX/repos/deepcsr/browse/preprop.py

This script registers an image (orig.mgz file from FreeSurfer outputs) to MNI space (using niftyreg),
the corresponding affine transformation matrix is then used to warp the FreeSurfer surfaces to the same space.
It further reduces the mesh resolution by quadratic edge collapse (using pymeshlab) 

To adapt the script for your data you need to set a few paths:
path for the MNI template file,
a .txt file with subject IDs
save directory 
directory with original freesurfer outputs

Requirements:
pymeshlab - for downsampling the meshes
niftyreg - for registration to MNI space
freesurfer - call to some freesurfer commands, e.g. mri_convert 
'''

import nibabel as nib
import numpy as np
import os
import trimesh
import subprocess
import multiprocessing as mp
import logging
import pymeshlab
ms = pymeshlab.MeshSet()

##### SET PATHs here: ######################################
# FreeSurfer output directory
OASIS_DIR = '/path/to/OASIS/FS_full/'
# Output directory
SAVE_DIR = '/path/to/OASIS/CSR_data'
SURFACES = ['lh_pial', 'lh_white', 'rh_pial', 'rh_white']
MNI_TEMPLATE_FILE = '/path/to/template/MNI152_T1_1mm.nii.gz'

###########################################################

with open('/path/to/OASIS/FS_full/sublist.txt') as f:
    SUBLIST = f.read().splitlines()

if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)


def register(pat, input_dir):
    print('register ID ', pat)

    # registering input images to MNI template
    sample_output_dir = os.path.join(SAVE_DIR,pat)
    if not os.path.exists(sample_output_dir):
            os.makedirs(sample_output_dir)
    out_mri_vol_path = os.path.join(sample_output_dir, 'mri.nii.gz')
    out_affine_path = os.path.join(sample_output_dir, 'transform_affine.txt')
    # this is a FreeSurfer command to convert Freesurfer .mgz data format to nifti
    subprocess.call(['mri_convert', os.path.join(input_dir,pat,'mri','orig.mgz'), out_mri_vol_path])
    # niftyreg registration:
    reg_aladin_cmd = ['reg_aladin', '-ref', MNI_TEMPLATE_FILE, '-flo', out_mri_vol_path, '-aff', out_affine_path, '-voff']       
    subprocess.call(reg_aladin_cmd)
    if os.path.exists(os.path.join(sample_output_dir, 'outputResult.nii.gz')): os.remove(os.path.join(sample_output_dir, 'outputResult.nii.gz'))
    subprocess.call(['reg_resample', '-ref', MNI_TEMPLATE_FILE, '-flo', out_mri_vol_path, 
        '-trans', out_affine_path, '-res', out_mri_vol_path, '-inter', '3', '-voff'])


# in case segmentation is needed:
def transform_segmentations(pat, input_dir):
    print('transform segm ID ', pat)
    sample_output_dir = os.path.join(SAVE_DIR,pat)
    if not os.path.exists(sample_output_dir):
            os.makedirs(sample_output_dir)

    out_aseg_vol_path = os.path.join(sample_output_dir, 'aseg.nii.gz')
    # FreeSurfer command
    subprocess.call(['mri_convert', os.path.join(input_dir,pat,'mri','aseg.mgz'), out_aseg_vol_path])
    
    out_affine_path = os.path.join(sample_output_dir, 'transform_affine.txt')
    # niftyreg registration with given transformation matrix (computed in register function   
    subprocess.call(['reg_resample', '-ref', MNI_TEMPLATE_FILE, '-flo', out_aseg_vol_path, 
        '-trans', out_affine_path, '-res', out_aseg_vol_path, '-inter', '0', '-voff'])
  
 
def convert_warp_surfaces(pat, input_dir):
    meshes = []

    print('warping surfaces ID ', pat)
    # converting and warping surfaces
    sample_output_dir = os.path.join(SAVE_DIR,pat)
    out_affine_path = os.path.join(sample_output_dir, 'transform_affine.txt')
    surf_path = os.path.join(input_dir,pat,'surf')
    lh_pial_path = os.path.join(surf_path,'lh.pial')
    lh_white_path = os.path.join(surf_path,'lh.white')
    rh_pial_path = os.path.join(surf_path,'rh.pial')
    rh_white_path = os.path.join(surf_path,'rh.white')
    # read fixed to moving transformation and invert to moving to fixed        
    T = np.linalg.inv(np.loadtxt(out_affine_path))
    
    for surf_name, surf_in_path in zip(SURFACES, [lh_pial_path, lh_white_path, rh_pial_path, rh_white_path]):        
        surf_scanner_path = os.path.join(sample_output_dir, "{}.scanner".format(surf_name))
        surf_out_path = os.path.join(sample_output_dir, "{}.ply".format(surf_name))
        # convert freesurfer mesh to scanner cordinates       
        subprocess.call(['mris_convert', '--to-scanner', surf_in_path, surf_scanner_path])
        # read fs mesh and save as .ply file
        verts, faces = nib.freesurfer.io.read_geometry(surf_scanner_path)
        # process=False is important as otherwise the ordering of vertices might be changed
        mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)

        # apply transformation
        mesh = mesh.apply_transform(T)
        mesh.export(surf_out_path)
        meshes.append(mesh)
        
        # delete .scanner file
        os.remove(surf_scanner_path)
    
    return meshes
  
        
def simplify_meshes(pat_id, reduction):
    '''
    param pat_id: patient id e.g. 1000_3
    param reduction: list of reduction parameters. e.g. 0.5 meaning 50% reduction, 0.3 meaning 70% reduction
    '''
    
    surfaces = ['lh_pial', 'lh_white', 'rh_pial', 'rh_white']
    
    # define paths of cortex meshes:
    meshes_path = os.path.join(SAVE_DIR, pat_id)
    for red in reduction:
        for surf in surfaces:
            if os.path.isfile(os.path.join(meshes_path, surf+'.ply')):
                ms.load_new_mesh(os.path.join(meshes_path, surf+'.ply'))
            else:
                print('skipping patient {}, mesh {} does not exist'.format(pat, surf))
                break

            # apply quadratic edge collapse filter:
            ms.simplification_quadric_edge_collapse_decimation(targetperc=red, preserveboundary=True, preservetopology=True)
            
            
            # save the current selected mesh
            ms.save_current_mesh(os.path.join(meshes_path, surf+'_reduced_{}.ply'.format(red)))

            # get a reference to the current selected mesh
            m = ms.current_mesh()
            print('Patient {}: {} reduction {}, resulting vertex number: {}'.format(pat_id, surf, red, m.vertex_number()))
            
    pass

def process_all(args):
    pat = args[0]
    input_dir = args[1]
    sample_output_dir = os.path.join(SAVE_DIR,pat)
    print('PROCESSING PAT ID ', pat)
    if not os.path.isfile(os.path.join(sample_output_dir, 'transform_affine.txt')):
        try:
            register(pat, input_dir)
        except Exception as ex:
            logging.exception('exception in registration occured. stop processing this patient')
            return
    if not os.path.isfile(os.path.join(sample_output_dir, 'aseg.nii.gz')):    
        try: 
            transform_segmentations(pat, input_dir)
        except Exception as ex:
            logging.exception('exception in transform segmentation occured. stop processing this patient')
            return
    if not os.path.isfile(os.path.join(sample_output_dir, 'rh_white.ply')):    
        try:
            meshes = convert_warp_surfaces(pat, input_dir)
            simplify_meshes(pat, [0.3])
        except Exception as ex:
            logging.exception('exception in warp surfaces occured. stop processing this patient')
            return



def main():
    pool = mp.Pool(mp.cpu_count())
    print('PROCESSING OASIS')
    oasis_iter = [(pat,OASIS_DIR ) for pat in SUBLIST]
    pool.map(process_all,oasis_iter)

    print('FINISHED OASIS')
 
 
	
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    np.seterr(all="warn")
    main()
