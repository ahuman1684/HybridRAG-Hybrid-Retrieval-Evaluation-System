# Binary Search Trees

A binary search tree (BST) is a binary tree data structure where each node
has at most two children, referred to as the left child and the right child.
The defining invariant is the BST property: for any node N, every key in
N's left subtree is less than N's key, and every key in N's right subtree
is greater than N's key. This ordering is what makes search efficient.

## Core Operations

Search, insertion, and deletion in a BST all follow the same basic strategy:
compare the target key against the current node and recurse into the left or
right subtree depending on the comparison. In a balanced tree, this gives
O(log n) time complexity for all three operations, since each comparison
eliminates roughly half of the remaining search space. In the worst case,
however, a BST built by inserting already-sorted data degenerates into a
linked list, giving O(n) time for search, insert, and delete.

Deletion is the trickiest of the three operations. There are three cases to
handle: deleting a leaf node (trivial, just remove it), deleting a node with
one child (replace the node with its child), and deleting a node with two
children (find the in-order successor - the smallest node in the right
subtree - copy its value into the node being deleted, then delete the
successor, which is guaranteed to have at most one child).

## Self-Balancing Variants

Because an unbalanced BST loses its O(log n) guarantee, self-balancing
variants exist to keep the tree height at O(log n) regardless of insertion
order. AVL trees enforce a strict balance factor (the height difference
between left and right subtrees of any node is at most 1) and rebalance via
rotations after every insertion or deletion. Red-black trees use a looser
balancing rule based on node coloring, which requires fewer rotations on
average than AVL trees, trading a slightly taller tree for cheaper
maintenance. Red-black trees are the backbone of many standard library
ordered containers, including C++'s `std::map` and Java's `TreeMap`.

## In-Order Traversal

An in-order traversal of a BST (visit left subtree, visit node, visit right
subtree) yields the keys in sorted order. This is a direct consequence of
the BST property and is frequently used to validate that a tree satisfies
the BST invariant, or to serialize a BST into a sorted sequence without a
separate sort step.

## When to Use a BST

BSTs are a good fit when you need an ordered collection that supports fast
search, insertion, deletion, and range queries (e.g. "give me all keys
between 10 and 50"), which a hash table cannot do efficiently since it has
no notion of key ordering. If you don't need ordering or range queries, a
hash table's average O(1) operations will usually outperform a BST's
O(log n) operations for pure lookups.
