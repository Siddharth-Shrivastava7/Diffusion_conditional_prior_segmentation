import os
import yaml

import torch
import ignite.distributed as idist 

import ddpm
import sys


def main(argv):
    params_file = "/home/sidd_s/Diffusion_conditional_prior_segmentation/ccdm-stochastic-segmentation_mod/params.yml" ## changing to absolute path for debugging and understanding the code
    if len(argv) == 2:
        if "params" in argv[1]:
            params_file = argv[1]
            print(f"Overriding params file with {params_file}...")
        else:
            print(f"ERROR: Unrecognized parameter: {argv[1]}")
            sys.exit(-1)

    with open(params_file, 'r') as f:
        params = yaml.safe_load(f)

    # Remove SLURM_JOBID to prevent ignite assume we are using SLURM to run multiple tasks.
    os.environ.pop("SLURM_JOBID", None)
    os.environ['WANDB_MODE'] = params['wandb_mode']

    params['num_gpus'] = torch.cuda.device_count()  # to be passed to run_train

    if params['distributed']:
        # Run distributed
        with idist.Parallel(
                backend="nccl",
                nproc_per_node=torch.cuda.device_count(),
                master_addr="127.0.0.1",
                master_port=27182) as parallel:
            parallel.run(ddpm.run_train, params)
    else:
        # Run in a single node
        ddpm.run_train(0, params)


if __name__ == "__main__":
    main(sys.argv)
