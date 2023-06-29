import logging
import torch
import random
from torch.utils.data import DataLoader
from lightning.pytorch import LightningDataModule
import os
from deepmath.deephol.deephol_loop import prooflog_to_torch
from deepmath.deephol.deephol_loop import options_pb2
from deepmath.deephol import io_util
from deepmath.deephol import deephol_pb2
from tqdm import tqdm
from deepmath.deephol.utilities.sexpression_to_torch import sexpression_to_pyg
from torch_geometric.data.batch import Batch
from torch_geometric.data import Data

def ptr_to_complete_edge_index(ptr):
    # print (ptr)
    from_lists = [torch.arange(ptr[i], ptr[i + 1]).repeat_interleave(ptr[i + 1] - ptr[i]) for i in range(len(ptr) - 1)]
    to_lists = [torch.arange(ptr[i], ptr[i + 1]).repeat(ptr[i + 1] - ptr[i]) for i in range(len(ptr) - 1)]
    combined_complete_edge_index = torch.vstack((torch.cat(from_lists, dim=0), torch.cat(to_lists, dim=0)))
    return combined_complete_edge_index

class HOListTrainingModule(LightningDataModule):
    def __init__(self, dir, batch_size=16):
        super().__init__()
        self.dir = dir
        self.batch_size = batch_size

    # gen vocab dictionary from file
    def gen_vocab_dict(self, vocab_file):
        with open(vocab_file) as f:
            x = f.readlines()
        vocab = {}
        for a, b in enumerate(x):
            vocab[b.replace("\n", "")] = a
        return vocab

    def prepare_data(self) -> None:
        if not os.path.exists(self.dir):
            logging.info('Generating data..')
            os.mkdir(self.dir)

            scrub_parameters = options_pb2.ConvertorOptions.NOTHING
            tactics_filename = '/home/sean/Documents/phd/deepmath-light/deepmath/deephol/data/hollight_tactics.textpb'

            logging.info('Loading theorem database..')
            theorem_db = io_util.load_theorem_database_from_file(
                '/home/sean/Documents/phd/deepmath-light/deepmath/deephol/data/theorem_database_v1.1.textpb')

            train_logs = io_util.read_protos(
                '/home/sean/Documents/phd/hol-light/holist/hollightdata/final/proofs/human/train/prooflogs*',
                deephol_pb2.ProofLog)
            val_logs = io_util.read_protos(
                '/home/sean/Documents/phd/hol-light/holist/hollightdata/final/proofs/human/valid/prooflogs*',
                deephol_pb2.ProofLog)

            # validation set:
            # len([a for a in theorem_db.theorems if a.training_split == 2 and  'complex' in a.library_tag])

            options = options_pb2.ConvertorOptions(tactics_path=tactics_filename, scrub_parameters=scrub_parameters)
            converter = prooflog_to_torch.create_processor(options=options, theorem_database=theorem_db)

            logging.info('Generating vocab..')
            vocab = self.gen_vocab_dict(
                '/home/sean/Documents/phd/hol-light/holist/hollightdata/final/proofs/human/vocab_ls.txt')
            vocab['UNK'] = len(vocab)

            logging.info('Loading proof logs..')
            train_proof_logs = []
            for j,i in tqdm(enumerate(converter.process_proof_logs(train_logs))):
                train_proof_logs.append(i)

            val_proof_logs = []
            for j,i in tqdm(enumerate(converter.process_proof_logs(val_logs))):
                val_proof_logs.append(i)

            train_params = []
            val_params = []
            for a in train_proof_logs:
                train_params.extend(a['thms'])

            for a in val_proof_logs:
                val_params.extend(a['thms'])

            all_params = train_params + val_params

            all_exprs = list(
                set([a['goal'] for a in train_proof_logs] + [a['goal'] for a in val_proof_logs] + all_params))

            logging.info(f'{len(all_exprs)} unique expressions')
            logging.info('Generating torch data dictionary from expressions..')

            torch_dict = {}
            for expr in tqdm(all_exprs):
                torch_dict[expr] = sexpression_to_pyg(expr, vocab)

            train_proof_logs = [{'goal': a['goal'], 'thms': a['thms'], 'tac_id': a['tac_id']} for a in train_proof_logs]
            val_proof_logs = [{'goal': a['goal'], 'thms': a['thms'], 'tac_id': a['tac_id']} for a in val_proof_logs]

            data = {'train_data': train_proof_logs, 'val_data': val_proof_logs,
                    'torch_dict': torch_dict, 'train_thm_ls': list(set(train_params)), 'vocab': vocab}

            torch.save(data, self.dir + '/data.pt')
            logging.info('Done')

    def setup(self, stage: str) -> None:
        if stage == "fit":
            data = torch.load(self.dir + '/data.pt')
            self.train_data = data['train_data']
            self.val_data = data['val_data']
            self.torch_dict = data['torch_dict']
            self.vocab = data['vocab']
            self.thms_ls = data['train_thm_ls']

    def train_dataloader(self):
        # return DataLoader(self.train_data, batch_size=self.batch_size, collate_fn=self.gen_batch)
        return DataLoader(self.train_data, batch_size=self.batch_size, collate_fn=self.gen_batch_relation, num_workers=2)

    def val_dataloader(self):
        # return DataLoader(self.val_data, batch_size=self.batch_size, collate_fn=self.gen_batch)
        return DataLoader(self.val_data, batch_size=self.batch_size, collate_fn=self.gen_batch_relation, num_workers=2)

    def gen_batch(self, batch):
        # todo filter negative sampling to be disjoint from positive samples

        # batch will be a list of proof step dictionaries with goal, thms, tactic_id
        goals = [self.torch_dict[x['goal']] for x in batch]

        # select random positive sample
        # if no parameters set it as a single element with '1' mapping to special token for no parameters
        pos_thms = [self.torch_dict[random.choice(x['thms'])] if len(x['thms']) > 0
                    else Data(x=torch.LongTensor([1]), edge_index=torch.LongTensor([[], []]), edge_attr=torch.LongTensor([]))
                    for x in batch]
        tacs = torch.LongTensor([x['tac_id'] for x in batch])

        # 15 random negative samples per goal
        neg_thms = [[self.torch_dict[a] for a in random.sample(self.thms_ls, 15)] for _ in goals]

        goals = Batch.from_data_list(goals)
        pos_thms = Batch.from_data_list(pos_thms)
        neg_thms = [Batch.from_data_list(th) for th in neg_thms]

        goals.attention_edge_index = ptr_to_complete_edge_index(goals.ptr)
        pos_thms.attention_edge_index = ptr_to_complete_edge_index(pos_thms.ptr)

        for i,th in enumerate(neg_thms):
            neg_thms[i].attention_edge_index = ptr_to_complete_edge_index(th.ptr)

        return goals, tacs, pos_thms, neg_thms

    def gen_batch_relation(self, batch):
        # todo filter negative sampling to be disjoint from positive samples
        # batch will be a list of proof step dictionaries with goal, thms, tactic_id
        goals = [self.torch_dict[x['goal']] for x in batch]

        # select random positive sample
        # if no parameters set it as a single element with '1' mapping to special token for no parameters
        pos_thms = [self.torch_dict[random.choice(x['thms'])] if len(x['thms']) > 0
                    else Data(x=torch.LongTensor([1]), edge_index=torch.LongTensor([[], []]), edge_attr=torch.LongTensor([]))
                    for x in batch]
        tacs = torch.LongTensor([x['tac_id'] for x in batch])

        # 15 random negative samples per goal
        neg_thms = [[self.torch_dict[a] for a in random.sample(self.thms_ls, 15)] for _ in goals]

        def data_to_relation(data_list):
            xi = [torch.LongTensor([data.x[i] for i in data.edge_index[0]]) + 1 for data in data_list]
            xj = [torch.LongTensor([data.x[i] for i in data.edge_index[1]]) + 1 for data in data_list]
            edge_attr = [data.edge_attr for data in data_list]

            xi = torch.nn.utils.rnn.pad_sequence(xi)
            xj = torch.nn.utils.rnn.pad_sequence(xj)
            edge_attr = torch.nn.utils.rnn.pad_sequence(edge_attr)

            mask = (xi == 0).T
            mask = torch.cat([mask, torch.zeros(mask.shape[0]).bool().unsqueeze(1)], dim=1)

            return Data(xi=xi, xj=xj, edge_attr_=edge_attr, mask=mask)

        goals = data_to_relation(goals)
        pos_thms = data_to_relation(pos_thms)
        neg_thms = [data_to_relation(th) for th in neg_thms]

        return goals, tacs, pos_thms, neg_thms


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    module = HOListTrainingModule('/home/sean/Documents/phd/deepmath-light/deepmath/train_data_new/')
    module.prepare_data()
