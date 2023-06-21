from __future__ import absolute_import
from __future__ import division
# Import Type Annotations
from __future__ import print_function

import logging
import sys

""""
DeepHOL non-distributed prover.
"""

from deepmath.deephol import prover_flags
from deepmath.deephol import prover_runner
from absl import flags


def main():
  prover_runner.program_started()
  prover_runner.run_pipeline(*prover_flags.process_prover_flags())


if __name__ == '__main__':
  logging.basicConfig(level=logging.DEBUG)


  FLAGS = flags.FLAGS
  FLAGS(sys.argv)

  main()

# flags.DEFINE_string(
#   'config', '/home/sean/Documents/phd/deepmath-light/deepmath/deephol/deephol_loop/data/loop1.pbtxt',
#   'Config file'
# )

# config = io_util.load_text_proto(FLAGS.config, loop_pb2.LoopConfig)

# prover_options = deephol_pb2.ProverOptions()
# prover_options.CopyFrom(config.prover_options)

# output_dir = '/home/sean/Documents/phd/deepmath-light/deepmath/deephol/data/'

# print (config)

# io_util.write_text_proto(
#   str(os.path.join(output_dir, 'prover_options.pbtxt')), prover_options)

