from deepmath.models.tactic_predictor import TacticPrecdictor, CombinerNetwork
from lightning.pytorch.loggers import WandbLogger
import logging
from deepmath.deephol.deephol_loop import torch_training_module
from deepmath.models.gnn.formula_net.gnn_encoder import GNNEncoder
import torch
from deepmath.deephol.deephol_loop.torch_data_module import HOListTrainingModule
import lightning.pytorch as pl



if __name__ == "__main__":
    NUM_TOKENS = 2044 + 5
    logging.basicConfig(level=logging.DEBUG)
    embedding_model_goal = GNNEncoder(input_shape=NUM_TOKENS, embedding_dim=128, num_iterations=12)
    embedding_model_premise = GNNEncoder(input_shape=NUM_TOKENS, embedding_dim=128, num_iterations=12)
    tac_model = TacticPrecdictor(embedding_dim=1024, num_tactics=41)
    combiner_model = CombinerNetwork(embedding_dim=1024, num_tactics=41, tac_embed_dim=128)

    module = HOListTrainingModule('/home/sean/Documents/phd/deepmath-light/deepmath/train_data_new/', batch_size=16)

    torch.set_float32_matmul_precision('medium')
    experiment = torch_training_module.HOListTraining_(embedding_model_goal=embedding_model_goal,
                                embedding_model_premise=embedding_model_premise,
                                tac_model=tac_model,
                                combiner_model=combiner_model)


    logger = WandbLogger(project='HOList Pretrain',
                         name='GNN',
                         # config=self.config,
                         # notes=self.config['notes'],
                         # log_model="all",
                         # offline=True,
                         )
    trainer = pl.Trainer(enable_progress_bar=True,devices=[0], val_check_interval=2048, limit_val_batches=512, logger=logger)
    trainer.fit(model=experiment, datamodule=module)
