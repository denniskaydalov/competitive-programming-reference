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
