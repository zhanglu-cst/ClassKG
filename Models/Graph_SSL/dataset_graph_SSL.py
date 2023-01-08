import dgl
import torch
from dgl.data import DGLDataset

from keyword_sentence.keywords import KeyWords
from keyword_sentence.sentence_process import get_sentence_hit_keywords


class Edges():
    def __init__(self, cfg, keywords: KeyWords):
        super(Edges, self).__init__()
        self.cfg = cfg
        self.keywords = keywords
        self._edge_src = []
        self._edge_dst = []

        self.edge_total_number = 0
        self.edge_hash_to_ID = {}

        # ---- edge feature  -------
        self.edge_appear_counts = []
        self.edge_fragment_lists = []
        if (self.cfg.use_edge_cat_feature):
            self.edge_cat_lists = []

    def get_edge_index(self, src_word, dst_word):
        # print('src_word:{}, dst_word:{}'.format(src_word, dst_word))
        hash = src_word + " " + dst_word
        if (hash not in self.edge_hash_to_ID):
            self.edge_hash_to_ID[hash] = self.edge_total_number
            self.edge_total_number += 1
            src_node_ID = self.keywords.keywords_to_index[src_word]
            dst_node_ID = self.keywords.keywords_to_index[dst_word]
            self._edge_src.append(src_node_ID)
            self._edge_dst.append(dst_node_ID)
            if (self.cfg.use_edge_cat_feature):
                self.edge_cat_lists.append([])
            self.edge_appear_counts.append(0)
            return self.edge_total_number - 1
        else:
            return self.edge_hash_to_ID[hash]

    @property
    def edge_src(self):
        return self._edge_src  # + self._edge_dst

    @property
    def edge_dst(self):
        return self._edge_dst  # + self._edge_src

    def add_edge(self, src_word: str, dst_word: str, edge_cat = None):
        edge_index = self.get_edge_index(src_word, dst_word)
        self.edge_appear_counts[edge_index] += 1
        if (edge_cat):
            self.edge_cat_lists[edge_index].append(edge_cat)

    def edge_class_to_soft_label(self):
        ans = torch.zeros(self.edge_total_number, self.cfg.model.number_classes)
        for edge_index, one_edge_cats in enumerate(self.edge_cat_lists):
            for cur_edge_one_cat in one_edge_cats:
                ans[edge_index][cur_edge_one_cat] += 1
        row_sum = torch.sum(ans, dim = 1)
        assert torch.sum(row_sum == 0) == 0
        row_sum = row_sum.unsqueeze(1)
        ans = torch.div(ans, row_sum)
        return ans

    @property
    def feature(self):
        # :  edge class, edge count, edge fragment embedding
        # if (self.cfg.use_edge_cat_feature):
        #     edge_soft_label = self.edge_class_to_soft_label()  # [num_edges, num_class]
        #     # return torch.cat([edge_soft_label, edge_soft_label], dim = 0)
        #     return edge_soft_label
        # else:
        #     return None
        count_feature = torch.tensor(self.edge_appear_counts).float()
        return {'count': count_feature}



class Graph_Keywords_Dataset_SSL(DGLDataset):

    def __init__(self, cfg, logger, keywords: KeyWords, sentences_vote, labels_vote, GT_labels):

        self.keywords = keywords
        self.cfg = cfg
        self.logger = logger
        self.self_loop = self.cfg.self_loop

        self.build_large_graph(sentences_vote, labels_vote)

        filter_sentences = []
        filter_labels = []
        filter_GTs = []
        for cur_sentence, cur_labels, cur_GT in zip(sentences_vote, labels_vote, GT_labels):
            words, origin_idx = get_sentence_hit_keywords(cur_sentence, self.keywords, return_origin_index = True)
            if (len(words) > 0):
                filter_sentences.append(cur_sentence)
                filter_labels.append(cur_labels)
                filter_GTs.append(cur_GT)
        self.sentences = filter_sentences
        self.labels = filter_labels
        self.GTs = filter_GTs

        super(Graph_Keywords_Dataset_SSL, self).__init__(name = 'KG', hash_key = (
            len(keywords.label_to_keywords[0]), len(keywords.label_to_keywords[1]),
            len(sentences_vote)))

    def process(self):
        pass

    def build_large_graph(self, sentences, labels):

        self.logger.info('build graphs, total number keywords:{}'.format(len(self.keywords)))
        # self.logger.info(str(self.keywords))
        self.edges = Edges(self.cfg, self.keywords)

        for cur_sentence, cur_label in zip(sentences, labels):
            hit_words, origin_index = get_sentence_hit_keywords(cur_sentence, keywords = self.keywords,
                                                                return_labels = False,
                                                                return_origin_index = True)
            for i in range(len(hit_words)):
                start_index = i if self.self_loop else i + 1
                cur_i_index = origin_index[i]
                for j in range(start_index, len(hit_words)):  # self loop
                    cur_j_index = origin_index[j]
                    dis = cur_j_index - cur_i_index
                    # print('dis:{}'.format(dis))
                    # if (dis < 500):
                    self.edges.add_edge(hit_words[i], hit_words[j], edge_cat = cur_label)

        src = torch.tensor(self.edges.edge_src).long()
        dst = torch.tensor(self.edges.edge_dst).long()
        print('edge_number:{}'.format(len(src)))

        Large_G = dgl.graph((src, dst))
        Large_G.ndata['nf'] = self.keywords.feature
        Large_G.edata['ef'] = self.edges.feature['count']
        # draw(Large_G)

        self.Large_G = Large_G


    def __getitem__(self, index):
        cur_sentence = self.sentences[index]
        cur_label = self.labels[index]
        cur_GT = self.GTs[index]
        words, origin_idx = get_sentence_hit_keywords(cur_sentence, self.keywords, return_origin_index = True)
        assert len(words) > 0

        set_words = list(set(words))

        node_IDs = []
        # subgraph_nodeID_to_word = {}
        # cur_subgraph_word_to_index = {}
        for i, word in enumerate(set_words):
            one_node_ID = self.keywords.keywords_to_index[word]
            node_IDs.append(one_node_ID)
            # subgraph_nodeID_to_word[i] = word
            # cur_subgraph_word_to_index[word] = i

        subgraph = dgl.node_subgraph(self.Large_G, node_IDs)

        return subgraph, cur_label, cur_GT, cur_sentence

    def __len__(self):
        return len(self.sentences)


def collate_fn(samples):
    # The input `samples` is a list of pairs
    #  (graphs, labels).
    graphs, labels, GTs, sentences = map(list, zip(*samples))
    batched_graph = dgl.batch(graphs)
    GTs = torch.tensor(GTs).long()
    return {'graphs': batched_graph, 'labels': torch.tensor(labels).long(), 'GT_labels': GTs, 'sentences': sentences}

# if __name__ == '__main__':
#
#     def collate(samples):
#         graphs, labels = map(list, zip(*samples))
#         batched_graph = dgl.batch(graphs)
#         batched_labels = torch.tensor(labels)
#         return batched_graph, batched_labels
#
#
#     # dataloader = DataLoader(
#     #         dataset,
#     #         batch_size = 1024,
#     #         collate_fn = collate,
#     #         drop_last = False,
#     #         shuffle = True)
#
#     import os
#     import torch
#     from PROJECT_ROOT import ROOT_DIR
#     from compent.logger import setup_logger
#     from config import cfg
#     from matplotlib import pyplot as plt
#     import networkx as nx
#
#     GPUs = '7'
#     cfg_file = 'SMS.yaml'
#     os.environ['CUDA_VISIBLE_DEVICES'] = GPUs
#     device = torch.device('cuda')
#     cfg_file = os.path.join(ROOT_DIR, 'config_files', cfg_file)
#     cfg.merge_from_file(cfg_file)
#     logger = setup_logger(name = 'train', save_dir = cfg.file_path.log_dir, distributed_rank = 0)
#
#     keywords = KeyWords(cfg = cfg, logger = logger)
#     sentence_all = Sentence_ALL(cfg)
#     dataset = Graph_Keywords_Dataset(cfg = cfg, logger = logger, keywords = keywords, sentence_all = sentence_all)
#     print(dataset.edges.feature)
#     for i, item in enumerate(dataset):
#         print(item)
#         graphs, labels, sentence, nodeID_to_word = item
#         print(graphs.ndata['nf'])
#         print(graphs.edata['ef'])
#         print(sentence)
#         nx.draw(graphs.to_networkx(), with_labels = True, labels = nodeID_to_word)
#         plt.show()
#         if (i == 5):
#             break
