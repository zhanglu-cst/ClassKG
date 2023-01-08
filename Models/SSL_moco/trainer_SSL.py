import datetime
import time

import torch
from torch import nn

from Models.SSL_moco.dataset_SSL import SSL_Graph_Moco_Dataset
from Models.SSL_moco.dataset_SSL import collate_fn
from Models.SSL_moco.moco import MoCo
from compent.checkpoint import CheckPointer_Normal
from compent.comm import get_rank
from compent.metric_logger import MetricLogger
from compent.utils import move_to_device, reduce_loss_dict


class Trainer_SSL():
    def __init__(self, cfg, logger, keywords, graph):
        self.rank = get_rank()
        self.cfg = cfg
        self.logger = logger
        self.keywords = keywords
        self.graph = graph

        model_moco = MoCo(cfg, logger, input_dim = len(self.keywords) + self.cfg.model.number_classes)

        model_moco.train()
        model_moco = model_moco.cuda()
        model_moco = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model_moco)
        self.model_moco = torch.nn.parallel.DistributedDataParallel(
                model_moco, device_ids = [self.rank], output_device = self.rank,  # find_unused_parameters=True
        )
        self.checkpointer = CheckPointer_Normal(cfg, logger, rank = self.rank)

    def do_train(self):
        self.logger.info('start SSL')
        self.logger.visdom_text(text = 'start SSL', win_name = 'state')
        dataset_train = SSL_Graph_Moco_Dataset(self.cfg, self.graph, keywords = self.keywords, max_length = 5)
        train_sampler = torch.utils.data.distributed.DistributedSampler(dataset_train)
        train_loader = torch.utils.data.DataLoader(
                dataset_train, batch_size = self.cfg.SSL.batch_size, pin_memory = True, sampler = train_sampler,
                drop_last = True, collate_fn = collate_fn)

        criterion = nn.CrossEntropyLoss().cuda(self.rank)
        optimizer = torch.optim.Adam(self.model_moco.parameters(), lr = 1e-3)

        meters = MetricLogger(delimiter = "  ")

        end = time.time()
        for iteration, batch in enumerate(train_loader):
            data_time = time.time() - end
            optimizer.zero_grad()
            batch = move_to_device(batch, rank = self.rank)
            output, target = self.model_moco(im_q = batch['batch_g1'], im_k = batch['batch_g2'])
            loss = criterion(output, target)

            loss_dict_reduced = reduce_loss_dict({'loss_all': loss})
            losses_reduced = sum(loss for loss in loss_dict_reduced.values())
            meters.update(loss = losses_reduced, **loss_dict_reduced)
            loss.backward()
            optimizer.step()
            batch_time = time.time() - end
            end = time.time()
            meters.update(time = batch_time, data = data_time)

            eta_seconds = meters.time.global_avg * (len(train_loader) - iteration)
            eta_string = str(datetime.timedelta(seconds = int(eta_seconds)))

            if iteration % 1 == 0:
                self.logger.info(
                        meters.delimiter.join(
                                [
                                    "SSL train",
                                    "eta: {eta}",
                                    "iter: {iter}",
                                    "total_itr: {total_itr}",
                                    "{meters}",
                                    "lr: {lr:.6f}",
                                    "max mem: {memory:.0f}",
                                ]
                        ).format(
                                eta = eta_string,
                                iter = iteration,
                                total_itr = len(train_loader),
                                meters = str(meters),
                                lr = optimizer.param_groups[0]["lr"],
                                memory = torch.cuda.max_memory_allocated() / 1024.0 / 1024.0,
                        ),
                        show_one = True
                )

                self.logger.plot_record(value = meters.loss.median,
                                        win_name = 'SSL loss,ITR:{}'.format(self.logger.get_value('ITR')),
                                        X_value = iteration)
            if iteration % 500 == 0:
                self.checkpointer.save_to_checkpoint_file_with_name(self.model_moco.module.encoder_q,
                                                                    filename = 'SSL_GIN_ITR{}_itr{}_loss_{}'.format(
                                                                            self.logger.get_value('ITR'), iteration,
                                                                            meters.loss.median)
                                                                    )

        self.checkpointer.save_to_checkpoint_file_with_name(self.model_moco.module.encoder_q,
                                                            filename = 'SSL_GIN_ITR{}'.format(
                                                                    self.logger.get_value('ITR')))
