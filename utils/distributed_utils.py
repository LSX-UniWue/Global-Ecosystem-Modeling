import os
import torch.distributed as dist


def setup_distributed(rank, world_size, master_port=12355):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = f'{master_port}'

    # initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)


def cleanup_distributed():
    dist.destroy_process_group()
