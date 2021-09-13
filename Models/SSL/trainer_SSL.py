import datetime
import time

import torch
from torch import nn
from torch.optim import AdamW

from Models.SSL.dataset_DA import DA_dataset
from Models.SSL.dataset_DA import collate_fn
from compent.checkpoint import CheckPointer_Normal
from compent.comm import get_rank, synchronize, is_main_process
from compent.metric_logger import MetricLogger
from compent.utils import move_to_device, reduce_loss_dict, get_memory_used


class Trainer_SSL():
    def __init__(self, cfg, logger, keywords, graph, evaler_labeling_quality, eval_on_origin, model):
        self.rank = get_rank()
        self.cfg = cfg
        self.logger = logger
        self.keywords = keywords
        self.graph = graph
        self.evaler_labeling_quality = evaler_labeling_quality
        self.model = model
        self.eval_on_origin = eval_on_origin

        # model = UnsupervisedGIN(cfg, input_dim = len(self.keywords) + self.cfg.model.number_classes)
        #
        # model.train()
        # model = model.cuda()
        # model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        # self.model = torch.nn.parallel.DistributedDataParallel(
        #         model, device_ids = [self.rank], output_device = self.rank,  # find_unused_parameters=True
        # )

        self.checkpointer = CheckPointer_Normal(cfg, logger, rank = self.rank)

    def do_train(self):
        self.logger.info('start SSL')
        self.logger.visdom_text(text = 'start DA', win_name = 'state')
        dataset_train = DA_dataset(self.cfg, self.logger, self.graph, keywords = self.keywords)
        train_sampler = torch.utils.data.distributed.DistributedSampler(dataset_train)
        train_loader = torch.utils.data.DataLoader(
                dataset_train, batch_size = self.cfg.SSL.batch_size, pin_memory = True, sampler = train_sampler,
                drop_last = True, collate_fn = collate_fn)

        criterion = nn.CrossEntropyLoss().cuda(self.rank)
        # optimizer = AdamW(self.model.parameters(), lr = 1e-3, weight_decay = 5e-4)
        optimizer = AdamW(self.model.parameters(), lr = 1e-3)
        # optimizer = make_optimizer(cfg = self.cfg, model = self.model)
        # lr_scheduler = make_lr_scheduler(cfg = self.cfg, optimizer = optimizer)

        meters = MetricLogger(delimiter = "  ")
        end = time.time()
        for iteration, batch in enumerate(train_loader):

            data_time = time.time() - end
            optimizer.zero_grad()
            batch = move_to_device(batch, rank = self.rank)
            output = self.model(batch['batch_graphs'])
            target = batch['labels']
            loss = criterion(output, target)

            loss_dict_reduced = reduce_loss_dict({'loss_all': loss})
            losses_reduced = sum(loss for loss in loss_dict_reduced.values())
            meters.update(loss = losses_reduced, **loss_dict_reduced)
            loss.backward()
            optimizer.step()
            batch_time = time.time() - end
            end = time.time()
            meters.update(time = batch_time, data = data_time)

            # lr_scheduler.step(iteration)

            eta_seconds = meters.time.global_avg * (len(train_loader) - iteration)
            eta_string = str(datetime.timedelta(seconds = int(eta_seconds)))

            if iteration % 20 == 0:
                GPU_memory = torch.cuda.max_memory_allocated() / 1024.0 / 1024.0
                memory = get_memory_used()
                self.logger.info(
                        meters.delimiter.join(
                                [
                                    "SSL train",
                                    "eta: {eta}",
                                    "iter: {iter}",
                                    "total_itr: {total_itr}",
                                    "{meters}",
                                    "lr: {lr:.6f}",
                                    "max mem: {GPU_memory:.0f}",
                                    "memory: {memory:.1f}",
                                ]
                        ).format(
                                eta = eta_string,
                                iter = iteration,
                                total_itr = len(train_loader),
                                meters = str(meters),
                                lr = optimizer.param_groups[0]["lr"],
                                GPU_memory = GPU_memory,
                                memory = memory,
                        ),
                        show_one = True
                )
                self.logger.plot_record(value = GPU_memory, win_name = 'GPU')
                self.logger.plot_record(value = memory, win_name = 'memory')

                self.logger.plot_record(value = meters.loss.median,
                                        win_name = 'SSL loss')
                self.logger.plot_record(value = optimizer.param_groups[0]["lr"], win_name = 'lr')


        synchronize()
