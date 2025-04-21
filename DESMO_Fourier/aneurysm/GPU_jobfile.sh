#!/bin/csh
#SBATCH --job-name=DESMOF2_v2
###SBATCH --nodes=1      # number of cluster nodes, abbreviated by -N
#SBATCH -o slurm-%j.out-%N # name of the stdout, using the job number (%j) and the first node (%N)
#SBATCH -e slurm-%j.err-%N # name of the stderr, using job and first node values
###SBATCH --nodes=1
###SBATCH --ntasks=1
#SBATCH --mem=20000
#SBATCH --partition=arzani-gpu-grn
#SBATCH --account=arzani
#SBATCH --qos=arzani-gpu-grn
#SBATCH --gres=gpu:1
#SBATCH --time=10:00:00                                                                                                                                                                                                                                                                     


setenv WORKDIR $HOME/Phase4/DESMOFourier/modes2_v2/
setenv SCRDIR /scratch/general/nfs1/u1447794/Phase4/DESMOFourier/modes2_v2/
mkdir -p $SCRDIR
cp -r $WORKDIR/* $SCRDIR
cd $SCRDIR
                   
module use $HOME/MyModules
module load miniconda3/latest
conda activate pytorch_env
								   
srun python DESMOFourier_v2.py > DESMO.out     





