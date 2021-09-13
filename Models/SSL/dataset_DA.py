import random

import dgl
import numpy
import torch
from torch.utils.data import Dataset


class DA_dataset(Dataset):
    def __init__(self, cfg, logger, graph, keywords):
        super(DA_dataset, self).__init__()
        self.graph = graph
        self.cfg = cfg
        self.num_nodes = graph.num_nodes()
        self.restart_prob = cfg.SSL.restart_prob
        self.keywords = keywords
        self.number_classes = cfg.model.number_classes
        self.number_itr = cfg.SSL.number_itr
        self.logger = logger

    def get_node_number(self):
        loc = self.logger.get_value('loc')
        std = self.logger.get_value('std')
        x = numpy.random.normal(loc = loc, scale = std)
        number = int(x)
        if (number <= 0):
            number = 1
        return number

    def generate_subgraph_with_start_node(self, start_node):
        node_num = self.get_node_number()
        g_traces, g_types = dgl.sampling.random_walk(self.graph, start_node, length = node_num, prob = 'ef')
        concat_vids, concat_types, lengths, offsets = dgl.sampling.pack_traces(g_traces, g_types)
        subgraph = self.graph.subgraph(concat_vids)
        return subgraph

    def __getitem__(self, index):
        cur_classes = random.randint(0, self.number_classes - 1)

        all_keywords_cur_class = self.keywords.label_to_keywords[cur_classes]

        node_1_keyword = random.sample(all_keywords_cur_class, 1)[0]
        node_1_ID = self.keywords.keywords_to_index[node_1_keyword]

        g1 = self.generate_subgraph_with_start_node(node_1_ID)

        return g1, torch.tensor(cur_classes).long()

    def __len__(self):
        return self.number_itr


def collate_fn(samples):
    graphs, labels = map(list, zip(*samples))
    batched_graph_1 = dgl.batch(graphs)
    labels = torch.tensor(labels).long()
    return {'batch_graphs': batched_graph_1, 'labels': labels}
