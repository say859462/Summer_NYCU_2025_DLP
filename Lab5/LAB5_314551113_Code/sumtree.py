import numpy as np

"""
    SumTree is a data structure that allows for efficient range sum queries and updates.
    Each leaf node contains the priority of a sample, and each internal node contains the sum of its children.
"""


class SumTree:

    def __init__(self, capacity):
        self.capacity = capacity

        """
            Assume that capacity is k to the power of 2.
            the number of internal node is 2^0 + 2^1 + ... + 2^(k-1) = 2^k - 1 = capacity - 1
            The number of leaf nodes is 2^k = capacity.
            So there are total 2 * capacity - 1 nodes in the tree.
        """
        self.tree = np.zeros(2 * capacity - 1)

        # data: (state, action, reward, next_state, done)
        self.data = np.zeros(capacity, dtype=object)
        # data_pointer: pointer to the next position to insert data
        self.data_pointer = 0
        self.n_entries = 0

    def add(self, priority, data):

        # 2^k - 1 = capacity - 1 is the last internal node index
        tree_idx = self.data_pointer + self.capacity - 1
        self.data[self.data_pointer] = data

        # Update the priority in the tree
        self.update(tree_idx, priority)

        self.data_pointer = (self.data_pointer + 1) % self.capacity
        
        if self.n_entries < self.capacity:
            self.n_entries += 1
            


    def update(self, tree_idx, priority):

        # Calculate the change in priority
        change = priority - self.tree[tree_idx]

        self.tree[tree_idx] = priority

        # Propagate the change up the tree
        while tree_idx != 0:
            # we minus 1 because the tree is 0-indexed
            tree_idx = (tree_idx - 1) // 2
            self.tree[tree_idx] += change

    def get_leaf(self,v):

        parent_idx = 0

        while True:
            left_child_idx = 2 * parent_idx + 1
            right_child_idx = left_child_idx + 1


            if left_child_idx >= len(self.tree):
                leaf_tree_idx = parent_idx
                break
            else:
                if v <= self.tree[left_child_idx]:
                    parent_idx = left_child_idx
                else:
                    v -= self.tree[left_child_idx]
                    parent_idx = right_child_idx
                    
        data_idx = leaf_tree_idx - self.capacity + 1
        
        return leaf_tree_idx, self.tree[leaf_tree_idx], self.data[data_idx]

    def total_priority(self):
        return self.tree[0]