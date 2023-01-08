import random

import dgl
import numpy
from torch.utils.data import Dataset
from compent.comm import broadcast_data

class SSL_Graph_Moco_Dataset(Dataset):
    def __init__(self, cfg, graph, keywords, max_length):
        super(SSL_Graph_Moco_Dataset, self).__init__()
        self.graph = graph
        self.cfg = cfg
        self.num_nodes = graph.num_nodes()
        self.max_length = max_length
        self.restart_prob = cfg.SSL.restart_prob
        self.keywords = keywords
        self.number_classes = cfg.model.number_classes
        self.number_itr = cfg.SSL.number_itr

    def get_node_number(self):
        x = numpy.random.normal(loc = 5, scale = 1)
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

        node_2_keyword = random.sample(all_keywords_cur_class, 1)[0]
        node_2_ID = self.keywords.keywords_to_index[node_2_keyword]

        g1 = self.generate_subgraph_with_start_node(node_1_ID)
        g2 = self.generate_subgraph_with_start_node(node_2_ID)

        return g1, g2

    def __len__(self):
        return self.number_itr


def collate_fn(samples):
    # The input `samples` is a list of pairs
    #  (graphs, labels).
    g1, g2 = map(list, zip(*samples))
    # print('len g1:{}, g1[0]:{}'.format(len(g1),g1[0]))
    batched_graph_1 = dgl.batch(g1)
    batched_graph_2 = dgl.batch(g2)
    return {'batch_g1': batched_graph_1, 'batch_g2': batched_graph_2}
