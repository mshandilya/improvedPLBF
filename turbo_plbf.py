import numpy as np
import copy
import sys
import turbo_structs
from time import time

def optFPR(ts):
    fs = []
    for i in range(len(ts)):
        g_reg = pg[ts[i]-1]
        h_reg = ph[ts[i]-1]
        if i > 0:
            g_reg -= pg[ts[i-1]-1]
        if i > 0:
            h_reg -= ph[ts[i-1]-1]
        fs.append((F*g_reg)/h_reg)
    while max(fs) > 1:
        for i in range(len(fs)):
            if fs[i] > 1:
                fs[i] = 1
        gs,hs = 0,0
        for i in range(len(fs)):
            g_reg = pg[ts[i]-1]
            h_reg = ph[ts[i]-1]
            if i > 0:
                g_reg -= pg[ts[i-1]-1]
            if i > 0:
                h_reg -= ph[ts[i-1]-1]
            if fs[i] == 1:
                gs += g_reg
                hs += h_reg
            elif fs[i] < 1:
                fs[i] = (g_reg*(F-hs))/(h_reg*(1-gs))
    return fs

def normBloom():
    return S*np.log2(np.e)*np.log2(1/F)

def spaceUsed(ts):
    ts.sort()
    fs = optFPR(ts)
    sz = 0
    for i in range(len(fs)):
        fi = fs[i]
        g_reg = pg[ts[i]-1]
        h_reg = ph[ts[i]-1]
        if i > 0:
            g_reg -= pg[ts[i-1]-1]
        if i > 0:
            h_reg -= ph[ts[i-1]-1]
        sz += S*((g_reg)/(pg[-2]))*np.log2(1/fi)*np.log2(np.e)
    return sz

sys.setrecursionlimit(10000)  # Setting high recursion limit to avail deepcopy

"""
N: Number of equally spaced score buckets
k: Target number of partitions
S: Size of the dataset being tested on
F: Target False Positive Rate
"""
N = 1000
F = 1e-2
S = 200000
k = 5

st_time = time()

# Loading sample Probability Mass Functions
dl = turbo_structs.DataLoader('default')
g, h, pg, ph = dl.load()


def dkl(x, y):
    if x > y:
        return -np.infty
    return (pg[y] - pg[x - 1]) * np.log2((pg[y] - pg[x - 1]) / (ph[y] - ph[x - 1]))


# the reduce function must take a linked list and return a modified linked list representing the columns present in the
# new matrix
def reduce(mat_a: turbo_structs.ImplicitMatrix):
    diag_point = 0  # Represents the index currently being pointed to
    cur_val = mat_a.fetch()  # Initializes the matrix pointers to (0, 0)
    while mat_a.rows.size < mat_a.columns.size:
        comp_val = mat_a.fetch(0, 1)
        if cur_val >= comp_val and diag_point < mat_a.rows.size - 1:
            cur_val = mat_a.fetch(1, 0)
            diag_point += 1
        elif cur_val >= comp_val and diag_point == mat_a.rows.size - 1:
            mat_a.delete()
            if mat_a.columns.size != diag_point + 1:
                cur_val = mat_a.fetch(0, -1)
        else:
            mat_a.fetch(0, -1)
            mat_a.delete()
            if diag_point > 0:
                mat_a.fetch(-1, -1)
                diag_point -= 1
    return


def max_compute(mat_a: turbo_structs.ImplicitMatrix):
    lower_bound = 0
    upper_bound = -1
    reduce(mat_a)
    mat_a.rows.pos_reset()
    mat_a.columns.pos_reset()
    true_upper_bound = mat_a.columns.end.val
    if mat_a.rows.size == 1:
        return [mat_a.columns.fetch_first().val]
    mat_b = copy.deepcopy(mat_a)
    mat_b.fetch()
    cur_size = mat_b.rows.size
    for row_ind in range(cur_size):
        if row_ind % 2 == 0:
            mat_b.delete('row')
        else:
            mat_b.fetch(1, 0)
    mat_b.rows.pos_reset()
    mat_b.columns.pos_reset()
    maxes = max_compute(mat_b)
    max_ind = 0
    ans = []
    mat_a.fetch()
    for row_ind in range(mat_a.rows.size):
        if row_ind % 2 == 0:
            upper_bound = maxes[max_ind] if (max_ind < len(maxes)) else true_upper_bound
            lower_bound = 0 if len(ans) == 0 else ans[-1]
            candidate_max = -np.infty
            candidate_max_ind = -1
            while mat_a.columns.fetch().val != upper_bound:
                candidate_val = mat_a.fetch()
                if candidate_val > candidate_max:
                    candidate_max = candidate_val
                    candidate_max_ind = mat_a.columns.fetch().val
                mat_a.fetch(0, 1)
            candidate_val = mat_a.fetch()
            if candidate_val > candidate_max:
                candidate_max = candidate_val
                candidate_max_ind = mat_a.columns.fetch().val
            ans.append(candidate_max_ind)
            mat_a.fetch(1, 0)
        else:
            while mat_a.columns.fetch().val != maxes[max_ind]:
                mat_a.fetch(0, 1)
            ans.append(maxes[max_ind])
            lower_bound = max(maxes[max_ind], lower_bound)
            max_ind += 1
            mat_a.fetch(1, 0)
    return ans


# Precalculate for just 1 partition
dp = np.array([[[0, -1] for _ in range(k)] for __ in range(N)], dtype=np.longdouble)
for ind in range(N):
    dp[ind, 0, 0] = dkl(0, ind)
for partitions in range(1, k):
    rows = turbo_structs.LinkedList()
    for ind in range(N):
        rows.insert(ind)
    rows.pos_reset()
    columns = copy.deepcopy(rows)
    a = turbo_structs.ImplicitMatrix(lambda x, y: (
        ((dp[y - 1, partitions - 1, 0] + dkl(y, x)) if (y > 0) else (-turbo_structs.INF)) if (y <= x) else (
            -turbo_structs.INF)), rows, columns)
    col_indices = max_compute(a)
    for ind in range(N):
        dp[ind, partitions, 0] = (
            (dp[col_indices[ind] - 1][partitions - 1][0] + dkl(col_indices[ind], ind)) if (col_indices[ind] > 0) else (
                -turbo_structs.INF)) if (col_indices[ind] <= ind) else (-turbo_structs.INF)
        dp[ind][partitions][1] = col_indices[ind]

en_time = time()
print("Construction Time: "+str(round((en_time-st_time)*1000))+"ms")

cur_x = N - 1
cur_y = k - 1
print(f'Score of partition: {dp[cur_x, cur_y, 0]}')
ts = []
while cur_x != -1:
    ts.append(cur_x+1)
    cur_x = int(dp[cur_x, cur_y, 1])
    cur_y -= 1

print("Space Used (Bloom Filter):",normBloom()/(1024*1024),"Mb")
print("Space Used (TurboPLBF):",spaceUsed(ts)/(1024*1024),"Mb")