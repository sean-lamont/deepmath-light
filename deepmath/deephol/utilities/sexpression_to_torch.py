from deepmath.deephol.utilities.sexpression_graphs import SExpressionGraph
from deepmath.deephol.utilities import sexpression_graphs

from typing import Dict, Iterable, List, NewType, Optional, Set, Text
from typing import Union
import torch
from torch_geometric.data import Data


def sexpression_to_pyg(sexpression_txt: Text, vocab: Dict) -> Data:

    sexpression = SExpressionGraph(sexpression_txt)

    edges = []
    node_to_tok = {}

    def process_sexpression_graph(node, depth):
        node_id = sexpression_graphs.to_node_id(node)

        # or check with is_leaf_node?..
        if len(sexpression.get_children(node)) == 0:
            if node_id not in node_to_tok:
                node_to_tok[node_id] = node
            assert sexpression.is_leaf_node(node_id)

        for i,child in enumerate(sexpression.get_children(node)):
            if i == 0:
                node_to_tok[node_id] = sexpression.to_text(child)
                # print("----" * depth + sexpression.to_text(child))
                continue

            edges.append((node_id, child, i))
            # order of children

            process_sexpression_graph(sexpression.to_text(child), depth + 1)


    process_sexpression_graph(sexpression.to_text(sexpression.roots()[0]), 0)

    edges = set(edges)
    senders = [a[0] for a in edges]
    receivers = [a[1] for a in edges]
    edge_attr = [a[2] for a in edges]

    all_nodes = list(set(senders + receivers))
    senders = [all_nodes.index(i) for i in senders]
    receivers = [all_nodes.index(i) for i in receivers]

    node_to_tok_ = {}
    for k,v in node_to_tok.items():
        node_to_tok_[all_nodes.index(k)] = v

    assert len(node_to_tok_) == len(all_nodes)

    tok_list = [0 for _ in range(len(all_nodes))]

    for k,v in node_to_tok_.items():
        tok_list[k] = v


    edge_index = torch.LongTensor([senders,receivers])
    edge_attr = torch.LongTensor(edge_attr)

    x = torch.LongTensor([vocab[tok] for tok in tok_list])

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


