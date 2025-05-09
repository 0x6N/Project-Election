import csv, random, threading, math, copy, sys, tkinter as tk
from tkinter import ttk, messagebox, scrolledtext, filedialog
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

GAMMA          = 0.3
seed           = 42
rng            = random.Random(seed)
CANDIDATE_CNT  = 11

def read_csv_and_build_rankings(csv_path):
    rankings, k = [], None
    with open(csv_path, newline="", encoding="utf-8") as fh:
        reader = csv.reader(fh); next(reader, None)          # skip header
        for row in reader:
            if len(row) < 13: continue
            scores = [int(c) if c.isdigit() else None for c in row[1:12]]
            labeled = [(cid, sc) for cid, sc in enumerate(scores) if sc is not None]
            labeled.sort(key=lambda t: t[1], reverse=True)
            ranking = [cid for cid, _ in labeled]
            k = k or len(ranking)
            if len(ranking) == k:
                rankings.append(ranking)
    return rankings, k

def build_position_lookup(rankings):
    return [{item: idx for idx, item in enumerate(r)} for r in rankings]

def build_pref_matrices(rankings, n=11):
    """
    pref_global[i,j]      : #rankings where i is above j  (all voters)
    pref_by_root[c][i,j]  : same, restricted to voters whose first choice is c
    root_counts[c]        : #rankings starting with candidate c
    total_pairs_global    : R * (k choose 2)
    """
    k = len(rankings[0])
    R = len(rankings)
    pref_global   = np.zeros((n, n), dtype=np.int32)
    pref_by_root  = np.zeros((n, n, n), dtype=np.int32)   # root, i, j
    root_counts   = np.zeros(n, dtype=np.int32)

    for r in rankings:
        root = r[0]
        root_counts[root] += 1
        for i in range(len(r)):
            for j in range(i+1, len(r)):
                a, b = r[i], r[j]
                pref_global[a, b]        += 1
                pref_by_root[root, a, b] += 1

    total_pairs_global = R * (k * (k-1) // 2)
    return pref_global, pref_by_root, root_counts, total_pairs_global

def get_subtree_nodes(parent, cut_child):
    stack, sub = [cut_child], {cut_child}
    children = [[] for _ in parent]
    for v, p in enumerate(parent):
        if p is not None: children[p].append(v)
    while stack:
        u = stack.pop()
        for v in children[u]:
            if v not in sub:
                sub.add(v); stack.append(v)
    return sub

def build_forced_pairs(parent):
    children = [[] for _ in parent]
    for v, p in enumerate(parent):
        if p is not None: children[p].append(v)
    forced = set()
    def dfs(u, anc):
        forced.add((anc, u))
        for w in children[u]: dfs(w, anc)
    for a in range(len(parent)):
        for ch in children[a]:
            dfs(ch, a)
    return forced

def re_root_tree(parent, new_root):
    new_p = parent.copy()
    cur, prev = new_root, None
    while cur is not None:
        nxt = new_p[cur]
        new_p[cur] = prev
        prev, cur = cur, nxt
    return new_p

def violation_count(ordering, M, memo):
    if len(ordering) <= 1:
        return 0
    if ordering in memo:
        return memo[ordering]
    first = ordering[0]
    val = sum(M[first][x] for x in ordering[1:]) + violation_count(ordering[1:], M, memo)
    memo[ordering] = val
    return val

def missed_order_count(ordering, S, memo):
    if len(ordering) <= 1:
        return 0
    if ordering in memo:
        return memo[ordering]
    first = ordering[0]
    val = sum(S[first][x] for x in ordering[1:]) + missed_order_count(ordering[1:], S, memo)
    memo[ordering] = val
    return val

def violation_plus_missed_cost_of_tree(parent, rankings, position):
    n, R = len(parent), len(rankings)
    if R == 0:
        return 0.0
    k = len(rankings[0])
    forced = build_forced_pairs(parent)
    M = [[1 if (j, i) in forced else 0 for j in range(n)] for i in range(n)]
    S = [[1 if i != j and ((i, j) not in forced and (j, i) not in forced) else 0
          for j in range(n)] for i in range(n)]
    v_memo, m_memo = {}, {}
    total_v = sum(violation_count(tuple(r), M, v_memo) for r in rankings)
    total_m = sum(missed_order_count(tuple(r), S, m_memo) for r in rankings)
    return (total_v + GAMMA * total_m) / (R * (k*(k-1)/2))

def connex_subtree_plus_missed_cost_of_tree(parent, rankings, position):
    R = len(rankings)
    if R == 0:
        return 0.0
    k = len(rankings[0])
    rem = 0
    for r in rankings:
        integ, L = set(), 0
        for c in r:
            if L == 0:
                integ.add(c)
                L = 1
            else:
                ok = (parent[c] in integ) or any(parent[x] == c for x in integ)
                if ok:
                    integ.add(c)
                    L += 1
                else:
                    break
        rem += (k - L)
    base = rem / (R * k)
    forced = build_forced_pairs(parent)
    n = len(parent)
    S = [[1 if i != j and ((i, j) not in forced and (j, i) not in forced) else 0
          for j in range(n)] for i in range(n)]
    total_m = sum(missed_order_count(tuple(r), S, {}) for r in rankings)
    return base + GAMMA * total_m / (R * (k*(k-1)/2))

def fast_cost_of_tree_re_rooted(parent, pref_global, pref_by_root,
                                root_counts, total_pairs_global, gamma):
    """
    Exactly replicates old cost_of_tree_re_rooted but uses ONLY the
    pre-aggregated matrices (so complexity O(n²) instead of O(R·k²)).
    """
    n = len(parent)
    cost_numer = 0.0

    # forced pairs once per root
    forced_by_root = {}
    for c in np.nonzero(root_counts)[0]:
        forced_by_root[c] = build_forced_pairs(re_root_tree(parent, c))

    k_choose2 = total_pairs_global // root_counts.sum()   

    for c in np.nonzero(root_counts)[0]:
        forced = forced_by_root[c]
        pref_c = pref_by_root[c]

        viol   = int( pref_c.swapaxes(0,1)[tuple(zip(*forced))].sum() ) \
                 if forced else 0
        resolved = int( (pref_c + pref_c.swapaxes(0,1))[tuple(zip(*forced))].sum() ) \
                   if forced else 0
        missed = root_counts[c]*k_choose2 - resolved

        cost_numer += viol + gamma*missed

    return cost_numer / total_pairs_global

def cost_of_tree_re_rooted_fast_wrapper(parent, rankings, position):
    return fast_cost_of_tree_re_rooted(
        parent,
        PREF_GLOBAL,
        PREF_BY_ROOT,
        ROOT_COUNTS,
        TOTAL_PAIRS_GLOBAL,
        GAMMA
    )

def random_tree(n, root=0):
    prufer = [rng.randrange(n) for _ in range(n-2)]
    degree = [1]*n
    for x in prufer: degree[x]+=1
    leaves = {i for i,d in enumerate(degree) if d==1}
    edges=[]
    for x in prufer:
        leaf = min(leaves); leaves.remove(leaf)
        edges.append((leaf,x)); degree[x]-=1
        if degree[x]==1: leaves.add(x)
    u,v = sorted(leaves); edges.append((u,v))
    adj={i:[] for i in range(n)}
    for u,v in edges: adj[u].append(v); adj[v].append(u)
    parent=[None]*n; visited=[False]*n; q=[root]; visited[root]=True
    while q:
        u=q.pop(0)
        for v in adj[u]:
            if not visited[v]:
                visited[v]=True; parent[v]=u; q.append(v)
    return parent

COST_OF_TREE_FUNCTION = cost_of_tree_re_rooted_fast_wrapper

def local_search(parent, rankings, position,
                 update_iter_callback=None, max_iter=100):
    n = len(parent)
    cur_cost = COST_OF_TREE_FUNCTION(parent, rankings, position)
    if update_iter_callback:
        update_iter_callback(f"Iteration 0, cost: {cur_cost:.6f}")

    iters=0
    while iters < max_iter:
        best_move, best_cost = None, cur_cost
        for c, p in enumerate(parent):
            if p is None: continue            # skip root
            sub = get_subtree_nodes(parent, c)
            parent[c]=None                    # cut
            for x in range(n):
                if x in sub or x==p: continue
                parent[c]=x
                nc = COST_OF_TREE_FUNCTION(parent, rankings, position)
                if nc < best_cost:
                    best_cost, best_move = nc, (c,x)
                parent[c]=None
            parent[c]=p                       # restore
        if not best_move: break
        c,x = best_move; parent[c]=x
        cur_cost = best_cost
        iters += 1
        if update_iter_callback:
            update_iter_callback(f"Iteration {iters}, cost: {cur_cost:.6f}")
    return parent

def print_tree(parent):
    children=[[] for _ in parent]; root=None
    for v,p in enumerate(parent):
        if p is None: root=v
        else: children[p].append(v)
    def dfs(u,l=0):
        print('  '*l+str(u))
        for v in children[u]: dfs(v,l+1)
    print("Tree (root at top):"); dfs(root)

def draw_tree(parent):
    G = nx.DiGraph(); G.add_nodes_from(range(len(parent)))
    for v,p in enumerate(parent):
        if p is not None: G.add_edge(p,v)
    plt.figure()
    try: pos = nx.nx_agraph.graphviz_layout(G, prog='dot')
    except: pos = nx.spring_layout(G)
    nx.draw(G,pos,with_labels=True,node_size=800,node_color='lightblue',arrows=True)
    plt.title("Ranking tree"); plt.show()
    plt.figure()
    pos2 = nx.planar_layout(G)
    nx.draw(G, pos2, with_labels=True, node_size=800,
            node_color='lightblue', arrows=True)
    plt.title("Tree (planar layout)")
    plt.show()


def slow_cost_of_tree_re_rooted(parent, rankings, position):
    """Original slow version (for regression test only)."""
    R=len(rankings); n=len(parent)
    forced_by_root={c:build_forced_pairs(re_root_tree(parent,c)) for c in range(n)}
    ranking_pairs=[{(r[i],r[j]) for i in range(len(r)) for j in range(i+1,len(r))}
                   for r in rankings]
    total=0.0
    for idx,r in enumerate(rankings):
        forced=forced_by_root[r[0]]
        v = sum(1 for (i,j) in forced
                if i in position[idx] and j in position[idx]
                and position[idx][j] < position[idx][i])
        m = sum(1 for (i,j) in ranking_pairs[idx]
                if (i,j) not in forced and (j,i) not in forced)
        k=len(r); total += (v + GAMMA*m)/(k*(k-1)/2)
    return total/R

def regression_test():
    csv_path = r"ssp\borda4.csv"
    rankings, _ = read_csv_and_build_rankings(csv_path)
    pos = build_position_lookup(rankings)
    global PREF_GLOBAL, PREF_BY_ROOT, ROOT_COUNTS, TOTAL_PAIRS_GLOBAL
    PREF_GLOBAL, PREF_BY_ROOT, ROOT_COUNTS, TOTAL_PAIRS_GLOBAL = \
        build_pref_matrices(rankings, CANDIDATE_CNT)

    for _ in range(200):
        tree = random_tree(CANDIDATE_CNT, 0)
        old = slow_cost_of_tree_re_rooted(tree, rankings, pos)
        new = cost_of_tree_re_rooted_fast_wrapper(tree, None, None)
        assert abs(old-new) < 1e-12, f"Mismatch {old} vs {new}"
    print(" passed")


class TreeGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Ranking Tree Synthesiser")
        self.geometry("520x500")
        self.csv_path_var = tk.StringVar()
        self.gamma_var   = tk.StringVar(value="0.3")
        self.trials_var  = tk.StringVar(value="20")
        self.seed_var    = tk.StringVar(value="42")
        self.log = None
        self.create_widgets()

    def create_widgets(self):
        row = ttk.Frame(self); row.pack(pady=4, padx=8, fill=tk.X)
        ttk.Label(row, text="CSV file").pack(side=tk.LEFT)
        e = ttk.Entry(row, textvariable=self.csv_path_var, width=32,
                      state="readonly")
        e.pack(side=tk.LEFT, padx=6, expand=True, fill=tk.X)
        ttk.Button(row, text="Browse...",
                   command=self.browse_csv).pack(side=tk.LEFT)
        
        row = ttk.Frame(self); row.pack(pady=4, padx=8, fill=tk.X)
        ttk.Label(row,text="Gamma").pack(side=tk.LEFT)
        ttk.Entry(row,textvariable=self.gamma_var,width=8).pack(side=tk.LEFT,padx=6)
        row = ttk.Frame(self); row.pack(pady=4, padx=8, fill=tk.X)
        ttk.Label(row,text="Trials").pack(side=tk.LEFT)
        ttk.Entry(row,textvariable=self.trials_var,width=8).pack(side=tk.LEFT,padx=6)
        row = ttk.Frame(self); row.pack(pady=4, padx=8, fill=tk.X)
        ttk.Label(row,text="Seed").pack(side=tk.LEFT)
        ttk.Entry(row,textvariable=self.seed_var,width=8).pack(side=tk.LEFT,padx=6)

        self.log = scrolledtext.ScrolledText(self, height=13, state='disabled')
        self.log.pack(padx=8, pady=8, fill=tk.BOTH, expand=True)
        ttk.Button(self,text="Run",command=self.run).pack(pady=6)
        row = ttk.Frame(self); row.pack(pady=4, padx=8, fill=tk.X)
        ttk.Label(row, text="Cost function").pack(side=tk.LEFT)
        self.cost_choice = tk.StringVar(value="Re-rooted (fast)")
        ttk.OptionMenu(row, self.cost_choice,
               self.cost_choice.get(),
               "Re-rooted (fast)",
               "Re-rooted (slow)",
               "Violation+Missed",
               "Connex Subtree+Missed").pack(side=tk.LEFT, padx=6)

        row = ttk.Frame(self); row.pack(pady=4, padx=8, fill=tk.X)
        ttk.Label(row, text="Total candidates").pack(side=tk.LEFT)
        self.cand_var = tk.StringVar(value="11")
        ttk.Entry(row, textvariable=self.cand_var, width=8).pack(side=tk.LEFT, padx=6)
    
    def browse_csv(self):
        fname = filedialog.askopenfilename(
            title="Select rankings CSV",
           filetypes=[("CSV files", "*.csv"), ("All files", "*.*")])
        if fname:
            self.csv_path_var.set(fname)

    def log_line(self,msg):
        self.log['state']='normal'
        self.log.insert(tk.END,msg+"\n"); self.log.see(tk.END)
        self.log['state']='disabled'

    
    def run(self):
        try:
            global GAMMA, rng, CANDIDATE_CNT, COST_OF_TREE_FUNCTION
            GAMMA = float(self.gamma_var.get())
            trials = int(self.trials_var.get())
            CANDIDATE_CNT = int(self.cand_var.get())
            seed_in = self.seed_var.get().strip().lower()
            rng = random.Random(None if seed_in == "random" else int(seed_in))
        except ValueError:
            messagebox.showerror("Bad input", "Numbers expected.")
            return

        choice = self.cost_choice.get()
        if choice == "Re-rooted (fast)":
            COST_OF_TREE_FUNCTION = cost_of_tree_re_rooted_fast_wrapper
        elif choice == "Re-rooted (slow)":
            COST_OF_TREE_FUNCTION = slow_cost_of_tree_re_rooted
        elif choice == "Violation+Missed":
            COST_OF_TREE_FUNCTION = violation_plus_missed_cost_of_tree
        else:
            COST_OF_TREE_FUNCTION = connex_subtree_plus_missed_cost_of_tree

        self.log['state']='normal'; self.log.delete('1.0',tk.END); self.log['state']='disabled'

        # build matrices once
        csv_path = self.csv_path_var.get()
        if not csv_path:
            messagebox.showerror("Missing file",
                                 "Please choose a CSV file first.")
            return
        rankings,k = read_csv_and_build_rankings(csv_path)
        position   = build_position_lookup(rankings)
        global PREF_GLOBAL, PREF_BY_ROOT, ROOT_COUNTS, TOTAL_PAIRS_GLOBAL
        PREF_GLOBAL, PREF_BY_ROOT, ROOT_COUNTS, TOTAL_PAIRS_GLOBAL = \
            build_pref_matrices(rankings, CANDIDATE_CNT)

        best_tree, best_cost = None, math.inf
        for t in range(trials):
            self.log_line(f"Trial {t+1}/{trials}")
            parent = random_tree(CANDIDATE_CNT,0)
            parent = local_search(parent, rankings, position,
                                  update_iter_callback=lambda m:self.log_line(f"  {m}"),
                                  max_iter=100)
            c = COST_OF_TREE_FUNCTION(parent, rankings, position)
            self.log_line(f"  Trial {t+1} cost: {c:.6f}")
            if c < best_cost:
                best_tree, best_cost = parent[:], c

        self.log_line(f"\nBest cost = {best_cost:.6f}")
        print_tree(best_tree)
        draw_tree(best_tree)

if __name__ == "__main__":        
    TreeGUI().mainloop()
