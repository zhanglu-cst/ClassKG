import dgl
import torch
from dgl.data import DGLDataset
from transformers import LongformerTokenizer

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

        if (self.cfg.use_edge_cat_feature):
            edge_soft_label = self.edge_class_to_soft_label()

            return edge_soft_label
        else:
            return None




class Graph_Keywords_Dataset(DGLDataset):

    def __init__(self, cfg, logger, keywords: KeyWords, sentences_vote, labels_vote, GT_labels, sentences_eval,
                 labels_eval,
                 for_train):

        self.keywords = keywords
        self.cfg = cfg
        self.logger = logger
        self.self_loop = self.cfg.self_loop

        self.istrain = for_train

        self.build_all(sentences_vote, labels_vote, GT_labels, sentences_eval, labels_eval)

        super(Graph_Keywords_Dataset, self).__init__(name = 'KG', hash_key = (
            len(keywords.label_to_keywords[0]), len(keywords.label_to_keywords[1]),
            len(sentences_vote)))

    def process(self):
        pass

    def build_all(self, sentences_vote, labels_vote, GT_labels, sentences_eval, labels_eval):
        self.build_large_graph(sentences_vote, labels_vote)
        self.pretrain_large_graph()
        if (self.istrain):
            self.subgraphs, self.labels, self.sentences = self.__build_subgraph_set_with_sentence__(sentences_vote,
                                                                                                    labels_vote)
            assert len(GT_labels) == len(self.labels), 'GT label number:{}, subgraph label:{}'.format(len(GT_labels),
                                                                                                      len(self.labels))
            assert len(self.labels) == len(labels_vote)
            self.GT_labels = GT_labels

        else:
            self.subgraphs, self.labels, self.sentences = self.__build_subgraph_set_with_sentence__(sentences_eval,
                                                                                                    labels_eval)
            self.GT_labels = self.labels
            self.logger.info('eval sentences origin:{}, hit:{}'.format(len(labels_eval), len(self.labels)))

    def build_large_graph(self, sentences, labels):

        self.logger.info('build graphs, total number keywords:{}'.format(len(self.keywords)))
        # self.logger.info(str(self.keywords))
        self.edges = Edges(self.cfg, self.keywords)

        for cur_sentence, cur_label in zip(sentences, labels):
            hit_words = get_sentence_hit_keywords(cur_sentence, keywords = self.keywords, return_labels = False)
            for i in range(len(hit_words)):
                start_index = i if self.self_loop else i + 1
                for j in range(start_index, len(hit_words)):  # self loop
                    self.edges.add_edge(hit_words[i], hit_words[j], edge_cat = cur_label)

        src = torch.tensor(self.edges.edge_src).long()
        dst = torch.tensor(self.edges.edge_dst).long()

        Large_G = dgl.graph((src, dst))
        Large_G.ndata['nf'] = self.keywords.feature
        # Large_G.edata['ef'] = self.edges.feature

        self.Large_G = Large_G

    def pretrain_large_graph(self):
        pass

    def __build_subgraph_set_with_sentence__(self, sentences, labels_all):
        subgraphs = []
        labels = []
        res_sentences = []
        subgraph_nodeID_to_keywords = []
        for index_s, (cur_sentence, cur_label) in enumerate(zip(sentences, labels_all)):
            words, origin_idx = get_sentence_hit_keywords(cur_sentence, self.keywords, return_origin_index = True)
            if (len(words) == 0):
                continue

            set_words = list(set(words))

            word_to_index = {}
            for one_word, one_index in zip(words, origin_idx):
                word_to_index[one_word] = one_index

            node_IDs = []
            subgraph_nodeID_to_word = {}
            cur_subgraph_word_to_index = {}
            for i, word in enumerate(set_words):
                one_node_ID = self.keywords.keywords_to_index[word]
                node_IDs.append(one_node_ID)
                subgraph_nodeID_to_word[i] = word
                cur_subgraph_word_to_index[word] = i

            feature_node_IDs_count = [0] * len(set_words)
            for word in words:
                index = cur_subgraph_word_to_index[word]
                feature_node_IDs_count[index] += 1
            feature_node_IDs_count = torch.tensor(feature_node_IDs_count).float().reshape(-1, 1)  # [len,1]

            subgraph = dgl.node_subgraph(self.Large_G, node_IDs)

            origin_nf = subgraph.ndata['nf']
            # print('origin_nf shape:{}'.format(origin_nf.shape))
            cat_feature = torch.cat((origin_nf, feature_node_IDs_count), dim = 1)
            # self.logger.info('node feature dim:{}'.format(cat_feature.shape))
            subgraph.ndata['nf'] = cat_feature

            subgraphs.append(subgraph)
            labels.append(cur_label)
            res_sentences.append(cur_sentence)
            subgraph_nodeID_to_keywords.append(subgraph_nodeID_to_word)
        # return subgraphs, labels, res_sentences, subgraph_nodeID_to_keywords
        return subgraphs, labels, res_sentences

    def __getitem__(self, index):
        return self.subgraphs[index], self.labels[index], self.sentences[index], self.GT_labels[index]

    def __len__(self):
        return len(self.subgraphs)


class Collect_FN_Combine():
    def __init__(self):
        super(Collect_FN_Combine, self).__init__()
        self.tokenizer = LongformerTokenizer.from_pretrained('allenai/longformer-base-4096')

    def __call__(self, batchs):
        # print(batchs)

        graphs, labels, sentences, GT_labels = map(list, zip(*batchs))
        graphs = dgl.batch(graphs)
        encoding = self.tokenizer(sentences, return_tensors = 'pt', padding = True, truncation = True)
        input_ids = encoding['input_ids']
        attention_mask = encoding['attention_mask']
        ans = {'input_ids': input_ids, 'attention_mask': attention_mask}
        labels = torch.tensor(labels).long()
        GT_labels = torch.tensor(GT_labels).long()
        ans['labels'] = labels
        ans['graphs'] = graphs
        ans['GT_labels'] = GT_labels
        return ans

#
# def collate_fn(samples):
#     # The input `samples` is a list of pairs
#     #  (graphs, labels).
#     graphs, labels = map(list, zip(*samples))
#     batched_graph = dgl.batch(graphs)
#     return {'graphs': batched_graph, 'labels': torch.tensor(labels).long()}
