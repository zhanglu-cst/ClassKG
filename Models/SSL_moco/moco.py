import torch
import torch.nn as nn

from Models.Graph_SSL.GIN_model import GIN
from compent.comm import accumulate_results_from_multiple_gpus
from compent.comm import get_rank


class MoCo(nn.Module):
    """
    Build a MoCo model with: a query encoder, a key encoder, and a queue
    https://arxiv.org/abs/1911.05722
    """

    def __init__(self, cfg, logger, input_dim, dim = 20, m = 0.999, T = 0.2, mlp = False):
        """
        dim: feature dimension (default: 128)
        K: queue size; number of negative keys (default: 65536)
        m: moco momentum of updating key encoder (default: 0.999)
        T: softmax temperature (default: 0.07)
        """

        super(MoCo, self).__init__()

        self.K = cfg.SSL.queue_size
        self.m = m
        self.T = T
        self.logger = logger

        self.rank = get_rank()
        self.device = torch.device('cuda:{}'.format(self.rank))
        # create the encoders
        # num_classes is the output fc dimension
        self.encoder_q = GIN(cfg, input_dim = input_dim)
        self.encoder_k = GIN(cfg, input_dim = input_dim)

        if mlp:  # hack: brute-force replacement
            dim_mlp = self.encoder_q.fc.weight.shape[1]
            self.encoder_q.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_q.fc)
            self.encoder_k.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_k.fc)

        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        # create the queue
        self.register_buffer("queue", torch.randn(dim, self.K))
        self.queue = nn.functional.normalize(self.queue, dim = 0)

        self.register_buffer("queue_ptr", torch.zeros(1, dtype = torch.long))

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        # gather keys before updating queue
        keys = concat_all_gather(keys)

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0, 'self.K:{},batch_size:{}'.format(self.K, batch_size)  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr:ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.logger.plot_record(value = ptr, win_name = 'SSL_ptr', X_value = None)

        self.queue_ptr[0] = ptr

    @torch.no_grad()
    def _batch_shuffle_ddp(self, x):
        """
        Batch shuffle, for making use of BatchNorm.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = len(x)
        x_gather = accumulate_results_from_multiple_gpus(x)
        # x_gather = concat_all_gather(x)
        batch_size_all = len(x_gather)

        num_gpus = batch_size_all // batch_size_this

        # random shuffle index
        idx_shuffle = torch.randperm(batch_size_all).cuda()

        # broadcast to all gpus
        torch.distributed.broadcast(idx_shuffle, src = 0)

        # index for restoring
        idx_unshuffle = torch.argsort(idx_shuffle)

        # shuffled index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_shuffle.view(num_gpus, -1)[gpu_idx]

        return get_elements_from_list(x_gather, idx_this), idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_ddp(self, x, idx_unshuffle):
        """
        Undo batch shuffle.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # restored index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_unshuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this]

    def forward(self, im_q, im_k):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
        Output:
            logits, targets
        """

        # compute query features
        q = self.encoder_q(im_q)  # queries: NxC
        # print('q1 feature shape:{}'.format(q.shape))  #[16,20]

        q = nn.functional.normalize(q, dim = 1)
        # print('q2 feature shape:{}'.format(q.shape))

        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder

            # shuffle for making use of BN
            # im_k, idx_unshuffle = self._batch_shuffle_ddp(im_k)
            # im_k = dgl.batch(im_k)
            # im_k = im_k.to(self.device)
            k = self.encoder_k(im_k)  # keys: NxC
            k = nn.functional.normalize(k, dim = 1)

            # undo shuffle
            # k = self._batch_unshuffle_ddp(k, idx_unshuffle)

        # compute logits
        # Einstein sum is more intuitive
        # positive logits: Nx1
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)  # [16,1]
        # negative logits: NxK
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])  # [16,2000]

        # logits: Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim = 1)  # [16,2001]

        # apply temperature
        logits /= self.T

        # labels: positive key indicators
        # 0 is the first one
        labels = torch.zeros(logits.shape[0], dtype = torch.long).cuda()

        # dequeue and enqueue
        self._dequeue_and_enqueue(k)

        return logits, labels


# utils
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
                      for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op = False)

    output = torch.cat(tensors_gather, dim = 0)
    return output


def get_elements_from_list(L, indexs):
    ans = []
    for index in indexs:
        ans.append(L[index])
    return ans
