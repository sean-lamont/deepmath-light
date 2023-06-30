import torch
import torch_geometric

def get_directed_edge_index(num_nodes, edge_idx):
    from_idx = []
    to_idx = []

    for i in range(0,num_nodes-1):
        # to_idx = [i]
        try:
            ancestor_nodes, _, self_idx, _ = torch_geometric.utils.k_hop_subgraph(i, num_hops=num_nodes,edge_index=edge_idx)
            # print (f"ancestor nodes for {i}: {ancestor_nodes}")
        except:
            print (f"exception {i, num_nodes, edge_idx}")

        # ancestor_nodes = ancestor_nodes.item()
        found_nodes = list(ancestor_nodes.numpy())
        found_nodes.remove(i)


        if found_nodes is not None:
            for node in found_nodes:
                to_idx.append(i)
                from_idx.append(node)

        children_nodes, _, self_idx, _ = torch_geometric.utils.k_hop_subgraph(i, num_hops=num_nodes,edge_index=edge_idx, flow='target_to_source')

        found_nodes = list(children_nodes.numpy())
        found_nodes.remove(i)
        if found_nodes is not None:
            for node in found_nodes:
                to_idx.append(i)
                from_idx.append(node)

    return torch.tensor([from_idx, to_idx], dtype=torch.long)


# def get_directed_edge_index(num_nodes, edge_idx):
#     from_idx = []
#     to_idx = []
#
#     for i in range(0,num_nodes-1):
#         # to_idx = [i]
#         try:
#             ancestor_nodes, _, self_idx, _ = torch_geometric.utils.k_hop_subgraph(i, num_hops=num_nodes,edge_index=edge_idx)
#         except:
#             print (f"exception {i, num_nodes, edge_idx}")
#
#         # ancestor_nodes = ancestor_nodes.item()
#         found_nodes = list(ancestor_nodes).remove(i)
#
#         if found_nodes is not none:
#             for node in found_nodes:
#                 to_idx.append(i)
#                 from_idx.append(node)
#
#         children_nodes, _, self_idx, _ = torch_geometric.utils.k_hop_subgraph(i, num_hops=num_nodes,edge_index=edge_idx, flow='target_to_source')
#         # children_nodes = children_nodes.item()
#         # print (found_nodes, children_nodes, i, self_idx.item(), edge_idx)
#         found_nodes = list(children_nodes).remove(i)
#
#         if found_nodes is not none:
#             for node in found_nodes:
#                 to_idx.append(i)
#                 from_idx.append(node)
#
#     return torch.tensor([from_idx, to_idx], dtype=torch.long)

# probably slow, could recursively do k-hop subgraph with k = 1 instead
def get_depth_from_graph(num_nodes, edge_index):
    to_idx = edge_index[1]


    # find source node
    all_nodes = torch.arange(num_nodes)
    source_node = [x for x in all_nodes if x not in to_idx]

    assert len(source_node) == 1

    source_node = source_node[0]

    depths = torch.zeros(num_nodes, dtype=torch.long)

    prev_depth_nodes = [source_node]

    for i in range(1, num_nodes):
        all_i_depth_nodes , _, _, _ = torch_geometric.utils.k_hop_subgraph(source_node.item(), num_hops=i, edge_index=edge_index, flow='target_to_source')
        i_depth_nodes = [j for j in all_i_depth_nodes if j not in prev_depth_nodes]

        for node_idx in i_depth_nodes:
            depths[node_idx] = i

        prev_depth_nodes = all_i_depth_nodes


    return depths

