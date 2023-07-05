from deepmath.models.tactic_predictor import TacticPrecdictor, CombinerNetwork
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
import logging
from deepmath.deephol.train import torch_training_module
from deepmath.models.gnn.formula_net.gnn_encoder import GNNEncoder
from deepmath.deephol.train.torch_data_module import HOListGraphModule
import lightning.pytorch as pl



if __name__ == "__main__":
    NUM_TOKENS = 2044 + 5
    logging.basicConfig(level=logging.DEBUG)


    exp_config = {
        "learning_rate": 1e-4,
        "epochs": 20,
        "weight_decay": 1e-6,
        "batch_size": 16,
        "model_save": False,
        "val_size": 512,
        "logging": False,
        "checkpoint_dir": "/home/sean/Documents/phd/deepmath-light/deepmath/train/",
        "device": [0],
        "val_frequency": 2048,
        "num_tactics": 41,
        "tac_embed_dim": 128,
        "final_embed_dim": 1024
    }

    data_config = {"source": "graph", "dir": '/home/sean/Documents/phd/deepmath-light/deepmath/attention_train_data/',
                   'batch_size': exp_config['batch_size']}

    data_module = HOListGraphModule(config=data_config)

    embedding_model_goal = GNNEncoder(input_shape=NUM_TOKENS, embedding_dim=128, num_iterations=12)
    embedding_model_premise = GNNEncoder(input_shape=NUM_TOKENS, embedding_dim=128, num_iterations=12)

    experiment = torch_training_module.HOListTraining_(embedding_model_goal=embedding_model_goal,
                                                       embedding_model_premise=embedding_model_premise,
                                                       tac_model=TacticPrecdictor(
                                                           embedding_dim=exp_config['final_embed_dim'],
                                                           num_tactics=exp_config['num_tactics']),
                                                       combiner_model=CombinerNetwork(
                                                           embedding_dim=exp_config['final_embed_dim'],
                                                           num_tactics=exp_config['num_tactics'],
                                                           tac_embed_dim=exp_config['tac_embed_dim']),
                                                       lr=exp_config['learning_rate'],
                                                       batch_size=exp_config['batch_size'])

    logger = WandbLogger(project='HOList Pretrain',
                         name='GNN Filtered Dataset',
                         config={'model_config': {'embedding_dim': 128, 'num_iterations': 12}, 'exp_config': exp_config},
                         # offline=True,
                         )

    callbacks = []

    # todo update model artifacts manually
    checkpoint_callback = ModelCheckpoint(monitor="rel_param_acc", mode="max",
                                          auto_insert_metric_name=True,
                                          save_top_k=3,
                                          filename="{epoch}-{rel_param_acc}-{topk_acc}",
                                          save_on_train_epoch_end=True,
                                          save_last=True,
                                          save_weights_only=True,
                                          dirpath=exp_config['checkpoint_dir'])

    callbacks.append(checkpoint_callback)

    trainer = pl.Trainer(enable_progress_bar=True,
                         devices=exp_config['device'],
                         val_check_interval=exp_config['val_frequency'],
                         limit_val_batches=exp_config['val_size'],
                         logger=logger,
                         callbacks=callbacks)

    trainer.fit(model=experiment, datamodule=data_module)



