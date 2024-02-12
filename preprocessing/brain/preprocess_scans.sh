#!/bin/bash
#
# Process a directory of input scans. Each scan should be named "<UID>.nii.gz".
#
if [ $# -ne 2 ]
	then
	echo "Wrong number of arguments."
	exit 0
fi

INPUT_FILE_DIR=$1
SUBJECT_DIR=$2

MNI_TEMPLATE_FILE=/mnt/nas/Data_Neuro/mni_templates/MNI152_T1_1mm.nii.gz
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
### Set number of parallel jobs
N=10

register_mri(){
	file=${1}
	id=$(basename "${file}" .nii.gz)
	echo $file
	out_dir="${SUBJECT_DIR}/$id"
	if [ ! -d "${out_dir}" ]; then
		mkdir $out_dir
	fi
	affine_file=$out_dir/niftyreg_affine.txt
	registered_file=$out_dir/mri_mni152.nii.gz
	reg_aladin -ref $MNI_TEMPLATE_FILE -flo $file -aff $affine_file
	reg_resample -ref $MNI_TEMPLATE_FILE -flo $file -trans $affine_file -res $registered_file -inter 3 -voff

	sleep 0.5
}

(
for file in $INPUT_FILE_DIR/*.nii.gz; do
	echo $file
	((i=i%N)); ((i++==0)) && wait
	register_mri "${file}" &
done
)

