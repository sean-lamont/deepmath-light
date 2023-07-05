import logging
import torch
import random
from torch.utils.data import DataLoader
from lightning.pytorch import LightningDataModule
from tqdm import tqdm
from torch_geometric.data.batch import Batch
from deepmath.deephol.train.graph_data_utils import DirectedData


# todo sequence data for Transformer
class HOListGraphModule(LightningDataModule):
    def __init__(self, dir, batch_size):
        super().__init__()
        self.dir = dir
        self.batch_size = batch_size

    def setup(self, stage: str) -> None:
        if stage == "fit":
            logging.info("Loading data..")
            data = torch.load(self.dir)
            logging.info("Filtering data..")

            self.torch_dict = {k: DirectedData(x=torch.LongTensor(v['onehot']),
                                               edge_index=torch.LongTensor(v['edge_index']),
                                               edge_attr=v['edge_attr'],
                                               abs_pe=v['depth'],
                                               attention_edge_index=v['attention_edge_index'])
                               for (k, v) in data['torch_dict'].items() if 'attention_edge_index' in v}

            self.train_data = self.filter(data['train_data'])
            self.val_data = self.filter(data['val_data'])
            self.vocab = data['vocab']
            self.thms_ls = [d for d in data['train_thm_ls'] if d in self.torch_dict]

    def filter(self, data):
        def process(d):
            if d['goal'] in self.torch_dict:
                if len(d['thms']) == 0:
                    return d
                thms = [th for th in d['thms'] if th in self.torch_dict]
                if len(thms) > 0:
                    d['thms'] = thms
                    return d
                else:
                    return None
            else:
                return None

        return [process(d) for d in tqdm(data) if process(d) is not None]

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.batch_size, collate_fn=self.gen_batch)

    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=self.batch_size, collate_fn=self.gen_batch)

    def gen_batch(self, batch):
        # todo filter negative sampling to be disjoint from positive samples

        # batch will be a list of proof step dictionaries with goal, thms, tactic_id
        goals = [self.torch_dict[x['goal']] for x in batch]

        # select random positive sample
        # if no parameters set it as a single element with '1' mapping to special token for no parameters
        pos_thms = [self.torch_dict[random.choice(x['thms'])] if len(x['thms']) > 0
                    else DirectedData(x=torch.LongTensor([1]), edge_index=torch.LongTensor([[], []]),
                                      edge_attr=torch.LongTensor([]), attention_edge_index=torch.LongTensor([[], []]),
                                      abs_pe=torch.LongTensor([0]))
                    for x in batch]

        tacs = torch.LongTensor([x['tac_id'] for x in batch])

        # 15 random negative samples per goal
        neg_thms = [[self.torch_dict[a] for a in random.sample(self.thms_ls, 15)] for _ in goals]

        goals = Batch.from_data_list(goals)
        pos_thms = Batch.from_data_list(pos_thms)
        neg_thms = [Batch.from_data_list(th) for th in neg_thms]

        return goals, tacs, pos_thms, neg_thms


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    module = HOListGraphModule(dir='/home/sean/Documents/phd/deepmath-light/deepmath/new_data_/data.pt',batch_size=16)
    # module.setup("fit")
    #
    # loader = module.train_dataloader()
    # i = 0
    # for b in tqdm(loader):
    #     i += 1
    #
    # print (i)
    #
    #





    # def gen_batch_relation(self, batch):
    #     # todo filter negative sampling to be disjoint from positive samples
    #     # batch will be a list of proof step dictionaries with goal, thms, tactic_id
    #     goals = [self.torch_dict[x['goal']] for x in batch]
    #
    #     # select random positive sample
    #     # if no parameters set it as a single element with '1' mapping to special token for no parameters
    #     pos_thms = [self.torch_dict[random.choice(x['thms'])] if len(x['thms']) > 0
    #                 else Data(x=torch.LongTensor([1]), edge_index=torch.LongTensor([[], []]),
    #                           edge_attr=torch.LongTensor([]))
    #                 for x in batch]
    #     tacs = torch.LongTensor([x['tac_id'] for x in batch])
    #
    #     # 15 random negative samples per goal
    #     neg_thms = [[self.torch_dict[a] for a in random.sample(self.thms_ls, 15)] for _ in goals]
    #
    #     def data_to_relation(data_list):
    #         xi = [torch.LongTensor([data.x[i] for i in data.edge_index[0]]) + 1 for data in data_list]
    #         xj = [torch.LongTensor([data.x[i] for i in data.edge_index[1]]) + 1 for data in data_list]
    #         edge_attr = [data.edge_attr for data in data_list]
    #
    #         xi = torch.nn.utils.rnn.pad_sequence(xi)
    #         xj = torch.nn.utils.rnn.pad_sequence(xj)
    #         edge_attr = torch.nn.utils.rnn.pad_sequence(edge_attr)
    #
    #         mask = (xi == 0).T
    #         mask = torch.cat([mask, torch.zeros(mask.shape[0]).bool().unsqueeze(1)], dim=1)
    #
    #         return Data(xi=xi, xj=xj, edge_attr_=edge_attr, mask=mask)
    #
    #     goals = data_to_relation(goals)
    #     pos_thms = data_to_relation(pos_thms)
    #     neg_thms = [data_to_relation(th) for th in neg_thms]
    #
    #     return goals, tacs, pos_thms, neg_thms
    #
