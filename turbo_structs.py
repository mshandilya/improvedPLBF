import math
import random
import numpy as np

# Defining Constants
INF = 1e18


def create_ideal_g_h(N):
    g_h_dif = [random.random() for _ in range(N)]
    g_h_list = [g_h_dif[0]]
    for i in range(1, N):
        g_h_list.append(g_h_list[i - 1] + g_h_dif[i])

    unnormed_h = [random.random() for _ in range(N)]
    sum_unnormed_h = sum(unnormed_h)
    h = [uh / sum_unnormed_h for uh in unnormed_h]

    unnormed_g = [h * g_h for h, g_h in zip(h, g_h_list)]
    sum_unnormed_g = sum(unnormed_g)
    g = [ug / sum_unnormed_g for ug in unnormed_g]

    return g, h

def simple_g_h(N):
    g = []
    for i in range(1,N+1):
        g.append(i/N)
    h = g[::-1]
    return g, h


class DataLoader:
    def __init__(self, dist='default', n=1000, lamb=3):
        # Provide custom distribution here, or generate it from the model's outputs
        if dist == 'default':
            self.g = [((lamb ** ((k * 15) // n) * np.exp(-lamb)) * 5) / math.factorial((k * 15) // n) for k in
                      range(1, n + 1)]
            self.h = self.g.copy()[::-1]
            self.g = np.array(self.g)
            self.h = np.array(self.h)
        elif dist == 'ideal':
            self.g, self.h = create_ideal_g_h(n)
            self.g = np.array(self.g)
            self.h = np.array(self.h)
        elif dist == 'simple':
            self.g, self.h = simple_g_h(n)
            self.g = np.array(self.g)
            self.h = np.array(self.h)
        else:
            print('Error while loading data')
        pg = []
        ph = []
        pgc = 0
        phc = 0
        for i in range(n):
            pgc += self.g[i]
            phc += self.h[i]
            pg.append(pgc)
            ph.append(phc)
        pg.append(0)
        ph.append(0)
        self.pg = pg
        self.ph = ph

    def load(self):
        return self.g, self.h, self.pg, self.ph


class Node:
    def __init__(self, val=None, prev=None):
        self.val = val
        self.prev = prev
        self.next = None

    def reroute(self, nx_node):
        self.next = nx_node
        nx_node.prev = self

    def __repr__(self):
        return f'Val:{self.val}\nPrevious Value:{"empty" if self.prev is None else self.prev.val}\nNext Value:{"empty" if self.next is None else self.next.val}'


class LinkedList:
    def __init__(self):
        self.begin = None
        self.end = None
        self.size = 0
        self.pos = -1
        self.posNode = None

    def insert(self, val):
        if self.size == 0:
            self.begin = Node(val)
            self.end = self.begin
            self.pos = 0
            self.posNode = self.begin
            self.size += 1
        else:
            self.size += 1
            self.pos += 1
            my_next = self.posNode.next
            self.posNode.next = Node(val)
            self.posNode.next.prev = self.posNode
            self.posNode = self.posNode.next
            if my_next is None:
                self.end = self.posNode
            else:
                self.posNode.next = my_next
                my_next.prev = self.posNode

    def remove(self):
        if self.pos == -1:
            return
        self.size -= 1
        p = self.posNode.prev
        n = self.posNode.next
        if self.pos != 0:
            p.next = n
        else:
            self.begin = n
        if self.pos != self.size:
            n.prev = p
            self.posNode = n
        elif self.pos != 0:
            self.pos -= 1
            self.posNode = p
            self.end = self.posNode
        else:
            self.pos = -1
            self.posNode = None
            self.end = self.posNode

    def fetch(self):
        if self.pos != -1:
            return self.posNode
        else:
            return None

    def fetch_next(self):
        if self.pos == self.size - 1:
            return None
        if self.pos != -1:
            self.pos += 1
            self.posNode = self.posNode.next
            return self.posNode
        if self.size != 0:
            self.posNode = self.begin
            self.pos += 1
            return self.posNode
        return None

    def fetch_prev(self):
        if self.pos == 0:
            return None
        if self.pos != -1:
            self.pos -= 1
            self.posNode = self.posNode.prev
            return self.posNode
        if self.size != 0:
            self.pos = self.size - 1
            self.posNode = self.end
            return self.posNode
        return None

    def fetch_first(self):
        if self.size != 0:
            self.pos = 0
            self.posNode = self.begin
            return self.posNode
        return None

    def fetch_last(self):
        if self.size != 0:
            self.pos = self.size - 1
            self.posNode = self.end
            return self.posNode
        return None

    def pos_reset(self):
        self.pos = -1
        self.posNode = None


class ImplicitMatrix:
    # der_func is the function that derives the value of a matrix's cell value
    def __init__(self, der_func, rows: LinkedList, columns: LinkedList):
        self.derive = der_func
        self.rows = rows
        self.columns = columns

    # Initially the function would always fetch the 0th row and 0th column
    def fetch(self, del_row: int = 0, del_column: int = 0):
        if self.rows.pos == -1:
            r = self.rows.fetch_first()
        elif del_row == 1:
            r = self.rows.fetch_next()
        elif del_row == -1:
            r = self.rows.fetch_prev()
        else:
            r = self.rows.fetch()
        if self.columns.pos == -1:
            c = self.columns.fetch_first()
        elif del_column == 1:
            c = self.columns.fetch_next()
        elif del_column == -1:
            c = self.columns.fetch_prev()
        else:
            c = self.columns.fetch()
        if r is None or c is None:
            return None
        return self.derive(r.val, c.val)

    def delete(self, axis='column'):
        if axis == 'row':
            df = self.rows
        elif axis == 'column':
            df = self.columns
        else:
            return
        if df.pos == -1 and df.size > 0:
            df.fetch_first()
            df.remove()
        elif df.pos == -1:
            return
        else:
            df.remove()
