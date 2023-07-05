import logging
import os

import torch
from tqdm import tqdm

from deepmath.deephol import io_util, deephol_pb2
from deepmath.deephol.deephol_loop import options_pb2
from deepmath.deephol.utilities.sexpression_to_graph import sexpression_to_graph
from deepmath.deephol.utilities import prooflog_to_examples


def prepare_data(save_dir, tac_dir, theorem_dir, train_logs, val_logs, test_logs=None, vocab_file=None):
    # gen vocab dictionary from file
    def gen_vocab_dict(self, vocab_file):
        with open(vocab_file) as f:
            x = f.readlines()
        vocab = {}
        for a, b in enumerate(x):
            vocab[b.replace("\n", "")] = a
        return vocab

    if not os.path.exists(save_dir):
        logging.info('Generating data..')
        os.mkdir(save_dir)

        scrub_parameters = options_pb2.ConvertorOptions.NOTHING

        logging.info('Loading theorem database..')
        theorem_db = io_util.load_theorem_database_from_file(theorem_dir)

        train_logs = io_util.read_protos(train_logs, deephol_pb2.ProofLog)
        val_logs = io_util.read_protos(val_logs, deephol_pb2.ProofLog)

        options = options_pb2.ConvertorOptions(tactics_path=tac_dir, scrub_parameters=scrub_parameters)
        converter = prooflog_to_examples.create_processor(options=options, theorem_database=theorem_db)

        logging.info('Loading proof logs..')
        train_proof_logs = []
        for j, i in tqdm(enumerate(converter.process_proof_logs(train_logs))):
            train_proof_logs.append(i)

        val_proof_logs = []
        for j, i in tqdm(enumerate(converter.process_proof_logs(val_logs))):
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
        logging.info('Generating data dictionary from expressions..')

        expr_dict = {expr: sexpression_to_graph(expr) for expr in tqdm(all_exprs)}

        train_proof_logs = [{'goal': a['goal'], 'thms': a['thms'], 'tac_id': a['tac_id']} for a in train_proof_logs]
        val_proof_logs = [{'goal': a['goal'], 'thms': a['thms'], 'tac_id': a['tac_id']} for a in val_proof_logs]

        if vocab_file:
            logging.info(f'Generating vocab from file {vocab_file}..')
            vocab = gen_vocab_dict(vocab_file)
            vocab['UNK'] = len(vocab)

        if not vocab_file:
            logging.info(f'Generating vocab from proof logs..')
            vocab = set([token for expr in expr_dict.values() for token in expr['tokens']])
            vocab["UNK"] = len(vocab)

        data = {'train_data': train_proof_logs, 'val_data': val_proof_logs,
                'expr_dict': expr_dict, 'train_thm_ls': list(set(train_params)), 'vocab': vocab}

        torch.save(data, save_dir + '/data.pt')
        logging.info('Done')


if __name__ == '__main__':
    save_dir = '/home/sean/Documents/phd/deepmath-light/deepmath/processed_train_data'
    vocab_file = '/home/sean/Documents/phd/hol-light/holist/hollightdata/final/proofs/human/vocab_ls.txt'
    tac_dir = '/home/sean/Documents/phd/deepmath-light/deepmath/deephol/data/hollight_tactics.textpb'
    theorem_dir = '/home/sean/Documents/phd/deepmath-light/deepmath/deephol/data/theorem_database_v1.1.textpb'
    train_logs = '/home/sean/Documents/phd/hol-light/holist/hollightdata/final/proofs/human/train/prooflogs*'
    val_logs = '/home/sean/Documents/phd/hol-light/holist/hollightdata/final/proofs/human/valid/prooflogs*'

    prepare_data(save_dir=save_dir, tac_dir=tac_dir, vocab_file=vocab_file, theorem_dir=theorem_dir,
                 train_logs=train_logs, val_logs=val_logs)
