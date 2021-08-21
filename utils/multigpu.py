
# Utility Fns from RePRI Repo https://github.com/mboudiaf/RePRI-for-Few-Shot-Segmentation
import os
import torch.distributed as dist

def find_free_port():
    """
    Used for distributed learning
    """
    import socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port

def main_process(args):
    if args.distributed:
        rank = dist.get_rank()
        if rank == 0:
            return True
        else:
            return False
    else:
        return True


def setup(args, rank, world_size):
    """
    Used for distributed learning
    """
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = str(args.port)

    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup():
    """
    Used for distributed learning
    """
    dist.destroy_process_group()
