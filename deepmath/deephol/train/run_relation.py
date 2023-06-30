import logging
import lightning.pytorch as pl
import torch
from lightning.pytorch.loggers import WandbLogger
from deepmath.deephol.deephol_loop import torch_training_module
from deepmath.deephol.deephol_loop.torch_data_module import HOListTrainingModule
from deepmath.models.get_model import get_model
from deepmath.models.tactic_predictor import TacticPrecdictor, CombinerNetwork

if __name__ == "__main__":
    NUM_TOKENS = 2044 + 5
    # logging.basicConfig(level=logging.WARNING)
    logging.basicConfig(level=logging.DEBUG)

    model_config = {
        # "model_type": "transformer_relation",
        "model_type": "transformer_relation_small",
        "vocab_size": NUM_TOKENS,
        # "vocab_size": VOCAB_SIZE + 1,
        "embedding_dim": 256,
        "dim_feedforward": 512,
        "num_heads": 8,
        "num_layers": 4,
        "dropout": 0.5
    }

    embedding_model_goal = get_model(model_config)
    embedding_model_premise = get_model(model_config)

    tac_model = TacticPrecdictor(embedding_dim=1024, num_tactics=41)
    combiner_model = CombinerNetwork(embedding_dim=1024, num_tactics=41, tac_embed_dim=128)

    module = HOListTrainingModule('/home/sean/Documents/phd/deepmath-light/deepmath/train_data_new/', batch_size=16)

    torch.set_float32_matmul_precision('medium')
    experiment = torch_training_module.HOListTraining_(embedding_model_goal=embedding_model_goal,
                                                       embedding_model_premise=embedding_model_premise,
                                                       tac_model=tac_model,
                                                       combiner_model=combiner_model)


    logger = WandbLogger(project='HOList Pretrain',
                         name='Relation',
                         config=model_config,
                         # notes=self.config['notes'],
                         # log_model="all",
                         offline=True,
                         )
    trainer = pl.Trainer(enable_progress_bar=True,devices=[0], val_check_interval=2048, limit_val_batches=512, logger=logger)
    trainer.fit(model=experiment, datamodule=module)
