# MaxUnionFind {{{
class MaxUnionFind:
    def __init__(self, size, nums):
        self.parent = [i for i in range(size)]
        self.max = nums[:]
        self.nums = nums
    
    def find(self, v):
        if v == self.parent[v]:
            return v
        self.parent[v] = self.find(self.parent[v])
        return self.parent[v]
    
    def union(self, u, v):
        u = self.find(u)
        v = self.find(v)
        if u != v:
            self.parent[v] = u
            self.max[u] = max(self.max[u], self.max[v])

# }}}

# UnionFind {{{ 
class UnionFind:
    def __init__(self, size):
        self.parent = [i for i in range(size)]
    
    def find(self, v):
        if v == self.parent[v]:
            return v
        self.parent[v] = self.find(self.parent[v])
        return self.parent[v]
    
    def union(self, u, v):
        u = self.find(u)
        v = self.find(v)
        if u != v:
            self.parent[v] = u
# }}}

# Trie {{{
class Trie:
    def __init__(self):
        self.children = dict()
        self.is_word = False

    def add_word(self, word):
        if not word:
            self.is_word = True
            return
        
        self.children[word[0]] = self.children.get(word[0], Trie())
        self.children[word[0]].add_word(word[1:])
# }}}

# Segment Tree {{{

class SegmentTree:
    def __init__(self, nums):
        self.l = [-1]*(4*len(nums))
        self.nums = nums
        self.build(1, 0, len(nums) - 1)

    def build(self, v, tl, tr):
        if (tl==tr):
            self.l[v] = self.nums[tl]
        else:
            tm = (tl+tr)//2
            self.build(v*2,tl,tm)
            self.build(v*2+1,tm+1,tr)
            self.l[v]=self.l[v*2]+self.l[v*2+1]
    
    def update_full(self, v,tl, tr, pos, new_val):
        if (tl==tr):
            self.l[v] = new_val
        else:
            tm = (tr+tl)//2
            if pos<=tm:
                self.update_full(v*2, tl, tm, pos, new_val)
            else:
                self.update_full(v*2+1, tm+1, tr, pos, new_val)
            self.l[v] = self.l[v*2]+self.l[v*2+1]

    def query_full(self, v, tl, tr, l, r):
        if l > r:
            return 0
        elif l == tl and r == tr:
            return self.l[v]
        tm = (tl+tr)//2
        return self.query_full(v*2,tl, tm,l,min(r,tm))+self.query_full(v*2+1,tm+1,tr,max(l,tm+1),r)

    def query(self,left,right):
        return self.query_full(1,0,len(self.nums)-1,left,right)

    def update(self, index, val):
        self.update_full(1, 0, len(self.nums)-1, index, val)

# }}}

# Topological Sort, Khans {{{

# g is an adjancency map
def topo_sort(g):
    d = dict()
    q = []
    t = []

    for i in g:
        for j in g[i]:
            d[j] = d.get(j, 0) + 1

    for i in g:
        if d.get(i, 0) == 0:
            q.append(i)

    while q:
        n = q.pop(0)
        t.append(n)

        for i in g[n]:
            d[i] -= 1

            if d[i] == 0:
                q.append(i)

    if len(t) != len(g):
        return None
    return t

# }}}

# Systems of Linear Equations, Gauss Jordan Elimination {{{
# example input:
# a = [[4, 1], [1, -1]]
# b = [5, 10]
# example output: [3.0, -7.0]

def gauss_jordan_elimination(a, b):
    aug_matrix = [a[i] + [b[i]] for i in range(len(a))]

    N = len(aug_matrix)
    M = len(aug_matrix[0])

    for i in range(N):
        if aug_matrix[i][i] == 0:
            for j in range(i + 1, N):
                if aug_matrix[j][i] != 0:
                    aug_matrix[i], aug_matrix[j] = aug_matrix[j], aug_matrix[i]
                    break
            else:
                return None

        for j in range(N):
            if i != j:
                ratio = aug_matrix[j][i] / aug_matrix[i][i]
                for k in range(M):
                    aug_matrix[j][k] -= ratio * aug_matrix[i][k]

    for i in range(N):
        divisor = aug_matrix[i][i]
        for j in range(M):
            aug_matrix[i][j] /= divisor

    return [round(aug_matrix[i][-1], 6) for i in range(len(aug_matrix))]
# }}}

# Sorted List {{{
# https://github.com/cheran-senthil/PyRival/blob/master/pyrival/data_structures/SortedList.py

from bisect import bisect_left as lower_bound
from bisect import bisect_right as upper_bound


class FenwickTree:
    def __init__(self, x):
        bit = self.bit = list(x)
        size = self.size = len(bit)
        for i in range(size):
            j = i | (i + 1)
            if j < size:
                bit[j] += bit[i]

    def update(self, idx, x):
        """updates bit[idx] += x"""
        while idx < self.size:
            self.bit[idx] += x
            idx |= idx + 1

    def __call__(self, end):
        """calc sum(bit[:end])"""
        x = 0
        while end:
            x += self.bit[end - 1]
            end &= end - 1
        return x

    def find_kth(self, k):
        """Find largest idx such that sum(bit[:idx]) <= k"""
        idx = -1
        for d in reversed(range(self.size.bit_length())):
            right_idx = idx + (1 << d)
            if right_idx < self.size and self.bit[right_idx] <= k:
                idx = right_idx
                k -= self.bit[idx]
        return idx + 1, k


class SortedList:
    block_size = 700

    def __init__(self, iterable=()):
        iterable = sorted(iterable)
        self.micros = [iterable[i:i + self.block_size - 1] for i in range(0, len(iterable), self.block_size - 1)] or [[]]
        self.macro = [i[0] for i in self.micros[1:]]
        self.micro_size = [len(i) for i in self.micros]
        self.fenwick = FenwickTree(self.micro_size)
        self.size = len(iterable)

    def insert(self, x):
        i = lower_bound(self.macro, x)
        j = upper_bound(self.micros[i], x)
        self.micros[i].insert(j, x)
        self.size += 1
        self.micro_size[i] += 1
        self.fenwick.update(i, 1)
        if len(self.micros[i]) >= self.block_size:
            self.micros[i:i + 1] = self.micros[i][:self.block_size >> 1], self.micros[i][self.block_size >> 1:]
            self.micro_size[i:i + 1] = self.block_size >> 1, self.block_size >> 1
            self.fenwick = FenwickTree(self.micro_size)
            self.macro.insert(i, self.micros[i + 1][0])

    def pop(self, k=-1):
        i, j = self._find_kth(k)
        self.size -= 1
        self.micro_size[i] -= 1
        self.fenwick.update(i, -1)
        return self.micros[i].pop(j)

    def __getitem__(self, k):
        i, j = self._find_kth(k)
        return self.micros[i][j]

    def count(self, x):
        return self.upper_bound(x) - self.lower_bound(x)

    def __contains__(self, x):
        return self.count(x) > 0

    def lower_bound(self, x):
        i = lower_bound(self.macro, x)
        return self.fenwick(i) + lower_bound(self.micros[i], x)

    def upper_bound(self, x):
        i = upper_bound(self.macro, x)
        return self.fenwick(i) + upper_bound(self.micros[i], x)

    def _find_kth(self, k):
        return self.fenwick.find_kth(k + self.size if k < 0 else k)

    def __len__(self):
        return self.size

    def __iter__(self):
        return (x for micro in self.micros for x in micro)

    def __repr__(self):
        return str(list(self))
# }}}
