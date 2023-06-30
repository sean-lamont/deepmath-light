import logging
from lightning.pytorch.callbacks import ModelCheckpoint
import lightning.pytorch as pl
import torch
from lightning.pytorch.loggers import WandbLogger
from deepmath.deephol.deephol_loop import torch_training_module
from deepmath.deephol.deephol_loop.torch_data_module import HOListTrainingModule
from deepmath.models.get_model import get_model
from deepmath.models.tactic_predictor import TacticPrecdictor, CombinerNetwork

if __name__ == "__main__":
    NUM_TOKENS = 2044 + 5
    logging.basicConfig(level=logging.DEBUG)

    model_config = {
        "model_type": "sat",
        # 'gnn_type': 'di_gcn',
        "num_edge_features": 3,
        "vocab_size": NUM_TOKENS,
        "embedding_dim": 256,
        "dim_feedforward": 256,
        "num_heads": 4,
        "num_layers": 2,
        "in_embed": True,
        "se": "gnn-encoder",
        "abs_pe": False,
        "abs_pe_dim": 256,
        "use_edge_attr": True,
        "dropout": 0.5,
        "gnn_layers": 12,
        "directed_attention": False,
        'small_inner': True
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
                         name='SAT',
                         config=model_config,
                         # notes=self.config['notes'],
                         # log_model="all",
                         # offline=True,
                         )
    trainer = pl.Trainer(enable_progress_bar=True, devices=[1], val_check_interval=2048, limit_val_batches=512,
                         logger=logger)
    trainer.fit(model=experiment, datamodule=module)




    def get_experiment(exp_config, model_config):
        return torch_training_module.HOListTraining_(embedding_model_goal=get_model(model_config),
                                                     embedding_model_premise=get_model(model_config),
                                                     tac_model=TacticPrecdictor(embedding_dim=1024, num_tactics=41),
                                                     combiner_model=CombinerNetwork(embedding_dim=1024,
                                                                                    num_tactics=41,
                                                                                    tac_embed_dim=128),
                                                     lr=exp_config['learning_rate'],
                                                     batch_size=exp_config['batch_size'])

    def get_data(data_config):
        return HOListTrainingModule(config=data_config)

    experiment = get_experiment(self.exp_config, self.model_config)

    data_module = get_data(self.data_config)

    logger = WandbLogger(project=self.config['project'],
                         name=self.config['name'],
                         config=self.config,
                         notes=self.config['notes'],
                         # offline=True,
                         )

    callbacks = []

    # todo update model artifacts manually

    checkpoint_callback = ModelCheckpoint(monitor="rel_param_acc", mode="max",
                                          auto_insert_metric_name=True,
                                          save_top_k=3,
                                          filename="{epoch}-{rel_param_acc}-{topk_acc}-{tac_acc}-{pos_acc}-{neg_acc}",
                                          save_on_train_epoch_end=True,
                                          save_last=True,
                                          save_weights_only=True,
                                          dirpath=self.exp_config['checkpoint_dir'])

    callbacks.append(checkpoint_callback)

    trainer = pl.Trainer(
        max_epochs=self.exp_config['epochs'],
        val_check_interval=self.exp_config['val_frequency'],
        limit_val_batches=self.exp_config['val_size'] // self.exp_config['batch_size'],
        logger=logger,
        enable_progress_bar=True,
        log_every_n_steps=500,
        # accelerator='gpu',
        devices=self.exp_config['device'],
        enable_checkpointing=True,
        callbacks=callbacks,
    )

    trainer.fit(model=experiment, datamodule=data_module)
