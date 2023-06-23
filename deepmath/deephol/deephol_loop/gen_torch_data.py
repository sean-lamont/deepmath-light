from deepmath.deephol.deephol_loop import prooflog_to_torch
from deepmath.deephol.deephol_loop import options_pb2
from deepmath.deephol import io_util
from deepmath.deephol import deephol_pb2
from tqdm import tqdm
from deepmath.deephol.utilities.sexpression_to_torch import sexpression_to_pyg


if __name__ == '__main__':
    # todo is this correct scrub level?
    scrub_parameters = options_pb2.ConvertorOptions.VALIDATION_AND_TESTING

    tactics_filename = '/home/sean/Documents/phd/deepmath-light/deepmath/deephol/data/hollight_tactics.textpb'
    theorem_db = io_util.load_theorem_database_from_file('/home/sean/Documents/phd/deepmath-light/deepmath/deephol/data/theorem_database_v1.1.textpb')

    train_logs = io_util.read_protos('/home/sean/Documents/phd/hol-light/holist/hollightdata/final/proofs/human/train/prooflogs*',deephol_pb2.ProofLog)
    val_logs = io_util.read_protos('/home/sean/Documents/phd/hol-light/holist/hollightdata/final/proofs/human/valid/prooflogs*',deephol_pb2.ProofLog)
    # test_logs = io_util.read_protos('/home/sean/Documents/phd/hol-light/holist/hollightdata/final/proofs/human/test/prooflogs*',deephol_pb2.ProofLog)

    #validation set:
    #len([a for a in theorem_db.theorems if a.training_split == 2 and  'complex' in a.library_tag])

    options = options_pb2.ConvertorOptions(tactics_path=tactics_filename, scrub_parameters=scrub_parameters)
    converter = prooflog_to_torch.create_processor(options=options, theorem_database=theorem_db)

    ll = []
    for i in tqdm(converter.process_proof_logs(train_logs)):
        ll.append(i)

    vocab = gen_vocab_dict('/home/sean/Documents/phd/hol-light/holist/hollightdata/final/proofs/human/vocab_ls.txt')

    data_list = [sexpression_to_pyg(y['goal'], vocab) for y in tqdm(ll)]

    # test case from paper
    # data = sexpression_to_pyg('(a (c (fun (fun A bool) bool) !) (l (v A x) (a (a (c (fun (fun A bool) A) =) (v A x)) (v A x))))', vocab=vocab)
