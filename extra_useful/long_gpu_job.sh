echo "#!/bin/bash
### Set the job name (for your reference)
#PBS -N iadb_cond_seg
### Set the project name, your department code by default
#PBS -P sit
### Priority set 
#PBS -q high
### Request email when job begins and ends
#PBS -m bea
### Specify email address to use for notification.
#PBS -M $USER@iitd.ac.in
####
#PBS -l select=2:ncpus=4:ngpus=4:centos=icelake
### Specify "wallclock time" required for this job, hhh:mm:ss
#PBS -l walltime=24:00:00

### #PBS -l software=python3
# After job starts, must goto working directory.
# $PBS_O_WORKDIR is the directory from where the job is fired.
echo "==============================="
echo \$PBS_JOBID
cat \$PBS_NODEFILE
echo "==============================="
cd $PWD
pwd
source $HOME/.bashrc
. $HOME/anaconda3/etc/profile.d/conda.sh
#job
conda activate py38tor18
cd Diffusion_conditional_prior_segmentation/IADB/tools
python multi_gpu_demo_testing.py
$*" > temp.sh
qsub temp.sh
rm temp.sh
