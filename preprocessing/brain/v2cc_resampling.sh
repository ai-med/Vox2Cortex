#!/bin/bash

export SUBJECTS_DIR=/path/to/FS_out
export SAVE_DIR=/path/to/FS_out_fsaverage_remapped
ID_FILE=/path/to/ids.txt
N=10

resample(){
  sub=${1}
  echo $sub
    mkdir -p $SAVE_DIR/$sub
    mkdir $SAVE_DIR/$sub/mri
    mkdir $SAVE_DIR/$sub/surf
    mkdir $SAVE_DIR/$sub/label
    cp $SUBJECTS_DIR/$sub/mri/orig.mgz $SAVE_DIR/$sub/mri/orig.mgz
    for hemi in lh rh ; do
      for surf in white pial ; do
	# sphere.reg file is missing for UKB so we need to create it. If this file exists, the next block can be skipped
        mris_smooth -n 3 -nw $SUBJECTS_DIR/$sub/surf/${hemi}.white $SUBJECTS_DIR/$sub/surf/${hemi}.smoothwm
        # Only needed to create *.sulc
        mris_inflate $SUBJECTS_DIR/$sub/surf/${hemi}.smoothwm $SUBJECTS_DIR/$sub/surf/${hemi}.inflated
        mris_curvature -w -min -max -a 10 $SUBJECTS_DIR/$sub/surf/${hemi}.smoothwm
        mris_curvature -w $SUBJECTS_DIR/$sub/surf/${hemi}.white # Probably not required
	mris_sphere $SUBJECTS_DIR/$sub/surf/${hemi}.inflated $SUBJECTS_DIR/$sub/surf/${hemi}.sphere
        mris_register -curv $SUBJECTS_DIR/$sub/surf/${hemi}.sphere $FREESURFER_HOME/average/${hemi}.average.curvature.filled.buckner40.tif $SUBJECTS_DIR/$sub/surf/${hemi}.sphere.reg
        mris_ca_label $sub ${hemi} $SUBJECTS_DIR/$sub/surf/${hemi}.sphere.reg $FREESURFER_HOME/average/${hemi}.DKTatlas40.gcs $SUBJECTS_DIR/$sub/label/${hemi}.aparc.DKTatlas40.annot
	###############

        # resample surface coordinates
        mri_surf2surf --s $sub --hemi $hemi --sval-xyz $surf --trgsubject fsaverage --tval $SAVE_DIR/$sub/surf/$hemi.$surf --tval-xyz $SAVE_DIR/$sub/mri/orig.mgz &
        
        # thickness
        mri_surf2surf --hemi $hemi --srcsubject $sub --srcsurfval thickness --src_type curv --trgsubject fsaverage --tval $SAVE_DIR/$sub/surf/$hemi.thickness --trg_type curv &

        # curvature

        mri_surf2surf --hemi $hemi --srcsubject $sub --srcsurfval curv --src_type curv --trgsubject fsaverage --tval $SAVE_DIR/$sub/surf/$hemi.curv --trg_type curv &

        # annot DKT
        mri_surf2surf --srcsubject $sub --trgsubject fsaverage --hemi $hemi --sval-annot $SUBJECTS_DIR/$sub/label/$hemi.aparc.DKTatlas.annot --tval $SAVE_DIR/$sub/label/$hemi.aparc.DKTatlas.annot &

        # annot Destrieux

        mri_surf2surf --srcsubject $sub --trgsubject fsaverage --hemi $hemi --sval-annot $SUBJECTS_DIR/$sub/label/$hemi.aparc.a2009s.annot --tval $SAVE_DIR/$sub/label/$hemi.aparc.a2009s.annot
      done
    done
  sleep 3 
}

(
for ID in `cat ${ID_FILE}`; do
    ((i=i%N)); ((i++==0)) && wait
    resample "${ID}" &
done
)
