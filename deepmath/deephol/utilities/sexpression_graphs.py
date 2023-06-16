"""Turn SExpressions into graphs and then into a tensorflow-digestible format."""

from __future__ import absolute_import
from __future__ import division
# Import Type Annotations
from __future__ import print_function

import farmhash
import six
# import tensorflow as tf
import logging
from typing import Dict, Iterable, List, NewType, Optional, Set, Text
from typing import Union
from deepmath.deephol.utilities import sexpression_parser
from deepmath.proof_assistant import proof_assistant_pb2

NodeID = NewType('NodeID', int)  # for nodes in the S-expression graph


def to_node_id(sexp: Text) -> NodeID:
  return NodeID(farmhash.fingerprint64(sexp))


def theorem_sexpression(theorem: proof_assistant_pb2.Theorem) -> Text:
  """Converts theorem object to an S-expression."""
  if theorem.tag == proof_assistant_pb2.Theorem.GOAL:
    return '(g (%s) %s)' % (' '.join(theorem.hypotheses), theorem.conclusion)
  if theorem.tag == proof_assistant_pb2.Theorem.THEOREM:
    return '(h (%s) %s)' % (' '.join(theorem.hypotheses), theorem.conclusion)
  if theorem.tag == proof_assistant_pb2.Theorem.DEFINITION:
    if theorem.hypotheses:
      raise ValueError('Detected definition with hypotheses.')
    return '(d (%s) %s)' % (' '.join(
        theorem.definition.constants), theorem.conclusion)
  if theorem.tag == proof_assistant_pb2.Theorem.TYPE_DEFINITION:
    if theorem.hypotheses:
      raise ValueError('Detected type definition with hypotheses.')
    return '(t %s %s)' % (theorem.type_definition.type_name, theorem.conclusion)
  raise ValueError('Unknown theorem tag.')


class SExpressionGraph(object):
  """Minimal graph representation of SExpressions generated by HOL Light.

  Nodes in this graph represent subexpressions.

  There are two types of edges: parent edges and child edges. The order of
  children is important; the order of parents is not.
  """

  def __init__(self, sexp=None):
    self.parents = {}  # type: Dict[NodeID, Set[NodeID]]
    self.children = {}  # type: Dict[NodeID, List[NodeID]]
    # Internal nodes have None label
    self.labels = {}  # type: Dict[NodeID, Optional[Text]]
    if sexp is not None:
      self.add_sexp(sexp)

  @property
  def nodes(self) -> Iterable[NodeID]:
    return self.children.keys()

  def __contains__(self, sexp: Text):
    return to_node_id(sexp) in self.labels

  def __len__(self):
    """Returns the number of nodes in the DAG representation of the expr."""
    return len(self.labels)

  def get_parents(self, sexp: Text) -> Set[NodeID]:
    return self.parents[to_node_id(sexp)]

  def get_children(self, sexp: Text) -> List[NodeID]:
    return self.children[to_node_id(sexp)]

  def get_label(self, sexp: Text) -> Optional[Text]:
    return self.labels[to_node_id(sexp)]

  def add_sexp(self,
               sexp_source: Union[Text, proof_assistant_pb2.Theorem, Iterable[
                   Union[Text, proof_assistant_pb2.Theorem]]]):
    """Adds new S-expressions; can be Text, Theorems, or lists thereof."""
    if (not isinstance(sexp_source, six.string_types) and
        not isinstance(sexp_source, proof_assistant_pb2.Theorem)):
      for s in sexp_source:
        self.add_sexp(s)
      return
    if isinstance(sexp_source, proof_assistant_pb2.Theorem):
      sexp_source = theorem_sexpression(sexp_source)
    self._add_text_sexp(sexp_source)

  def _add_text_sexp(self, sexp: Text):
    """Add new nodes to the S-expression graph."""
    if sexp in self:
      if self.to_text(to_node_id(sexp)) != sexp:
        logging.fatal('Fingerprint collision in S-expression graph parser.')
      return
    children = sexpression_parser.children(sexp)
    node_id = to_node_id(sexp)
    self.children[node_id] = []
    self.parents[node_id] = set()
    self.labels[node_id] = None if children else sexp
    for c in children:
      self.add_sexp(c)
      child_id = to_node_id(c)
      self.children[node_id].append(child_id)
      self.parents[child_id].add(node_id)

  def is_empty_string(self) -> bool:
    """Checks if the graph represents the empty string."""
    return len(self.parents) == 1 and '' in self

  def is_leaf_node(self, node: NodeID) -> bool:
    return node in self.labels and self.labels[node] is not None

  def global_post_order(self, skip_first_child=False) -> Dict[NodeID, int]:
    order = {}
    for n in self.roots():  # roots is sorted by hash, hence output is unique
      self.post_order(n, order=order, skip_first_child=skip_first_child)
    return order

  def post_order(self, node: NodeID, order=None,
                 skip_first_child=False) -> Dict[NodeID, int]:
    """Compute the unique post order for the given node.

    Non-recursive implementation of the following code:
      if node in order:
        return order
      for c in self.children[node]:
        post_order(c, order)
      order[node] = len(order)+1

    Using a non-recursive implementation as the expressions can become quite
    large, so there is the risk to run into stack overflows. As we use this in
    large pipelines, such errors could be quite costly.

    Args:
      node: node indicating the subexpression to start from.
      order: For internal use only; used for partial mappings.
      skip_first_child: Ignore the first child of each node. This helps us to
        skip nodes that do not show up in certain tree representations of HOL
        Light terms.

    Returns:
      Mapping from NodeIDs to unique consecutive integers starting from 0.
    """
    if order is None:
      order = {}
    order_id = len(order)
    stack = [node]
    while stack:
      node = stack[-1]  # node will only be popped when children are processed
      if node in order:
        stack.pop()
        continue
      all_children_in_order = True
      children = self.children[node]
      if skip_first_child:
        children = children[1:]
      for c in children[::-1]:  # reversed, so that stack.pop is in order
        if c not in order:
          # process all children before node, which is still on the stack
          stack.append(c)
          all_children_in_order = False
      if all_children_in_order:
        stack.pop()
        order[node] = order_id
        order_id += 1
    return order

  def to_text(self, node_id: NodeID) -> Text:
    """Return the string of the S-expression represented by node_id."""

    def space_needed(worklist):
      """Slightly hacky but works for s-expressions."""
      return worklist and worklist[-1] != ')'

    # Traverse the DAG as a tree; respect the order of the children and insert
    # parentheses where needed.
    tokens = []
    worklist = [node_id]  # contains node_ids and closing parens
    while worklist:
      item = worklist.pop()  # pops the last item
      if item == ')':
        tokens.append(item)
        if space_needed(worklist):
          tokens.append(' ')
        continue

      node_id = item
      if self.is_leaf_node(node_id):
        tokens.append(self.labels[node_id])
        if space_needed(worklist):
          tokens.append(' ')
      else:
        tokens.append('(')
        worklist.append(')')
        for c in self.children[node_id][::-1]:
          worklist.append(c)
    return ''.join(tokens)

  # TODO(mrabe): maybe introduce field maintaining the list of roots?
  def roots(self):
    """Returns all nodes without parents; sorted by hash value."""
    roots = [n for n in self.nodes if not self.parents[n]]
    roots.sort()
    return roots

  def is_abstraction(self, node):
    """Relies on the S-expression originating from HOL Light."""
    if self.is_leaf_node(node):
      return False
    return self.labels[self.children[node][0]] == 'l'

  def get_bound_variable(self, node) -> Optional[Text]:
    """Relies on the S-expression originating from HOL Light."""
    if not self.is_abstraction(node):
      return None
    variable_node = self.children[node][1]
    if self.labels[self.children[variable_node][0]] != 'v':
      raise ValueError(
          'Expected a variable node (v <type> <var_name>). Expression was %s' %
          self.to_text(node))
    if not self.is_leaf_node(self.children[variable_node][2]):
      raise ValueError(
          'Expected a variable node (v <type> <var_name>). Expression was: %s' %
          self.to_text(node))
    return self.labels[self.children[variable_node][2]]

  def is_variable(self, node):
    """Relies on the s-expression originating from HOL Light."""
    if self.is_leaf_node(node):
      return False
    return self.labels[self.children[node][0]] == 'v'

  def is_variable_name(self, node):
    """Relies on the S-expression originating from HOL Light."""
    if not self.is_leaf_node(node):
      return False
    return any([
        self.is_variable(p) and self.children[p][2] == node
        for p in self.parents[node]
    ])
