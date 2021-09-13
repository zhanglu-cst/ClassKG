import torch
import torch.distributed as dist

from compent.comm import synchronize,get_world_size


def set_multi_GPUs_envs(rank,world_size):
    # print('rank:{}, world size:{}'.format(rank,world_size))
    # print(torch.cuda.is_available())
    torch.cuda.set_device(rank)
    dist.init_process_group('nccl', rank = rank, world_size = world_size)
    synchronize()
    assert get_world_size() > 1


