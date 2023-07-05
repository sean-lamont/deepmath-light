from deepmath.models.transformer.transformer_encoder_model import TransformerWrapper
from deepmath.models.gnn.formula_net.gnn_encoder import GNNEncoder
from deepmath.models.sat.models import GraphTransformer
from deepmath.models.gnn.pna import GCNGNN, DiGCNGNN

'''
Utility function to fetch model given a configuration dict
'''


def get_model(model_config):
    if model_config['model_type'] == 'sat':
        return GraphTransformer(in_size=model_config['vocab_size'],
                                num_class=2,
                                d_model=model_config['embedding_dim'],
                                dim_feedforward=model_config['dim_feedforward'],
                                num_heads=model_config['num_heads'],
                                num_layers=model_config['num_layers'],
                                in_embed=model_config['in_embed'],
                                se=model_config['se'],
                                gnn_type=model_config['gnn_type'] if 'gnn_type' in model_config else 'gcn',
                                abs_pe=model_config['abs_pe'],
                                abs_pe_dim=model_config['abs_pe_dim'],
                                use_edge_attr=model_config['use_edge_attr'],
                                num_edge_features=model_config['num_edge_features'],
                                dropout=model_config['dropout'],
                                k_hop=model_config['gnn_layers'],
                                small_inner=model_config['small_inner'] if 'small_inner' in model_config else False)

    elif model_config['model_type'] == 'gnn-encoder':
        return GNNEncoder(input_shape=model_config['vocab_size'],
                          embedding_dim=model_config['embedding_dim'],
                          num_iterations=model_config['gnn_layers'],
                          dropout=model_config['dropout'])

    elif model_config['model_type'] == 'transformer':
        return TransformerWrapper(ntoken=model_config['vocab_size'],
                                  d_model=model_config['embedding_dim'],
                                  nhead=model_config['num_heads'],
                                  nlayers=model_config['num_layers'],
                                  dropout=model_config['dropout'],
                                  d_hid=model_config['dim_feedforward'],
                                  small_inner=model_config['small_inner'] if 'small_inner' in model_config else False)

    else:
        raise NotImplementedError
