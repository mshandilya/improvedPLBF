import numpy as np
from time import time
import turbo_structs
import random

def optFPR(ts):
    fs = []
    for i in range(len(ts)):
        g_reg = pg[ts[i]]
        h_reg = ph[ts[i]]
        if i > 0:
            g_reg -= pg[ts[i-1]]
        if i > 0:
            h_reg -= ph[ts[i-1]]
        fs.append((F*g_reg)/h_reg)
    while max(fs) > 1:
        for i in range(len(fs)):
            if fs[i] > 1:
                fs[i] = 1
        gs,hs = 0,0
        for i in range(len(fs)):
            g_reg = pg[ts[i]]
            h_reg = ph[ts[i]]
            if i > 0:
                g_reg -= pg[ts[i-1]]
            if i > 0:
                h_reg -= ph[ts[i-1]]
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
        g_reg = pg[ts[i]]
        h_reg = ph[ts[i]]
        if i > 0:
            g_reg -= pg[ts[i-1]]
        if i > 0:
            h_reg -= ph[ts[i-1]]
        sz += S*((g_reg)/(pg[-2]))*np.log2(1/fi)*np.log2(np.e)
    return sz

INF = 1e18

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
dl = turbo_structs.DataLoader('default')
g, h, pg, ph = dl.load()
pg = [0]+pg
ph = [0]+ph

def dkl(l, r):
    if r < l:
        return l-r+1
    elif r < 1:
        return r-l+1
    elif r >= 1 and l < 1:
        if r <= N:
            return (pg[r]-pg[0])*np.log2((ph[r]-ph[0])/(pg[r]-pg[0])) + 1-l
        else:
            return (pg[N]-pg[0])*np.log2((ph[N]-ph[0])/(pg[N]-pg[0])) + 1-l + r-N
    elif r <= N:
        return (pg[r]-pg[l-1])*np.log2((ph[r]-ph[l-1])/(pg[r]-pg[l-1]))
    elif l <= N and r > N:
        return (pg[N]-pg[l-1])*np.log2((ph[N]-ph[l-1])/(pg[N]-pg[l-1])) + r-N
    else:
        return r-l+1

def F_func(x):
    cur_pos = 1
    val = 0
    for i in range(len(x)):
        xi = abs(round(x[i]*N))
        val += dkl(cur_pos, cur_pos+xi-1)
        cur_pos += xi
    return val

def G(x):
    tot = 0
    pos = 0
    for i in range(len(x)):
        tot += abs(round(x[i]*N))
        if x[i] >= 0:
            pos += 1
    return ((tot-N)/N)

def get_rand_pt(k):
    pt = []
    for _ in range(k):
        pt.append(random.random())
    return pt

def grad(func, pt):
    eps = 1/N
    gra = []
    for i in range(len(pt)):
        pt[i] -= eps
        y1 = func(pt)
        pt[i] += 2*eps
        y2 = func(pt)
        pt[i] -= eps
        gra.append((y2-y1)/(2*eps))
    return gra

random.seed(42)
pt = get_rand_pt(k)
lamb = random.random()
c = 1e4
pathx = []
pathy = []
best_pt = None
best_f = 0

for iter in range(N):
    if G(pt) == 0:
        if F_func(pt) < best_f:
            best_f = F_func(pt)
            best_pt = pt.copy()
    gradf = grad(F_func, pt)
    gradg = grad(G, pt)
    old_lamb = lamb
    Gval = G(pt)
    lamb += Gval
    for i in range(k):
        pt[i] += 1e-5*(-gradf[i]-old_lamb*gradg[i]-c*Gval*gradg[i])

en_time = time()
print("Construction Time: "+str(round((en_time-st_time)*1000))+"ms")

print("Score of partition:",-best_f)
cur = sum([round(x*N) for x in best_pt])
ts = []
for x in best_pt:
    ts.append(cur)
    cur -= round(x*N)

print("Space Used (Bloom Filter):",normBloom()/(1024*1024),"Mb")
print("Space Used (DiffPLBF):",spaceUsed(ts)/(1024*1024),"Mb")