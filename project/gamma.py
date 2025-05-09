import csv
import random
import networkx as nx
import matplotlib.pyplot as plt
from ete3 import Tree, TreeStyle
seed= 42
seeded = random.Random(seed)
GAMMA = 0.3

def read_csv_and_build_rankings(csv_path):
    rankings = []
    k = None
    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader, None)  
        for row in reader:
            if len(row) < 13:
                continue  
            candidate_scores = []
            for col_idx in range(1, 12):
                cell = row[col_idx].strip()
                if cell == "":
                    candidate_scores.append(None)
                else:
                    try:
                        candidate_scores.append(int(cell))
                    except ValueError:
                        candidate_scores.append(None)
            labeled = []
            for cand_id, sc in enumerate(candidate_scores):
                if sc is not None:
                    labeled.append((cand_id, sc))
            labeled.sort(key=lambda x: x[1], reverse=True)
            ranking = [x[0] for x in labeled]
            if k is None:
                k = len(ranking)
            if len(ranking) == k:
                rankings.append(ranking)
    return rankings, k


def build_position_lookup(rankings):
    """
    For each ranking r:
       position[r][candidate] = index
    Only the k ranked candidates are present.
    """
    R = len(rankings)
    position = [dict() for _ in range(R)]
    for r in range(R):
        for idx, item in enumerate(rankings[r]):
            position[r][item] = idx
    return position

def get_subtree_nodes(parent, cut_child):
    """
    nodes in subtree rooted at 'cut_child'.
    """
    stack = [cut_child]
    subtree = set([cut_child])
    n = len(parent)
    children = [[] for _ in range(n)]
    for i in range(n):
        p = parent[i]
        if p is not None:
            children[p].append(i)
    while stack:
        current = stack.pop()
        for ch in children[current]:
            if ch not in subtree:
                subtree.add(ch)
                stack.append(ch)
    return subtree

def build_forced_pairs(parent):
    n = len(parent)
    children = [[] for _ in range(n)]
    for i in range(n):
        p = parent[i]
        if p is not None:
            children[p].append(i)
    forced_pairs = set()
    def dfs(u, ancestor):
        forced_pairs.add((ancestor, u))
        for ch in children[u]:
            dfs(ch, ancestor)
    for i in range(n):
        for ch in children[i]:
            dfs(ch, i)
    return forced_pairs

def build_ranking_pairs(rankings):
    ranking_pairs = []
    for r in rankings:
        pair_set = set()
        for i in range(len(r)):
            for j in range(i+1, len(r)):
                pair_set.add((r[i], r[j]))
        ranking_pairs.append(pair_set)
    return ranking_pairs


def violation_count(ordering, M, memo):
    if len(ordering) <= 1:
        return 0
    if ordering in memo:
        return memo[ordering]
    first = ordering[0]
    cost_first = sum(M[first][x] for x in ordering[1:])
    cost_rest = violation_count(ordering[1:], M, memo)
    total = cost_first + cost_rest
    memo[ordering] = total
    return total

def missed_order_count(ordering, S, memo):
    if len(ordering) <= 1:
        return 0
    if ordering in memo:
        return memo[ordering]
    first = ordering[0]
    cost_first = sum(S[first][x] for x in ordering[1:])
    cost_rest = missed_order_count(ordering[1:], S, memo)
    total = cost_first + cost_rest
    memo[ordering] = total
    return total

def violation_plus_missed_cost_of_tree(parent, rankings, position):
    n = len(parent)        
    R = len(rankings)
    if R == 0:
        return 0.0
    k = len(rankings[0])   
    forced_pairs = build_forced_pairs(parent)
    M = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i != j and (j, i) in forced_pairs:
                M[i][j] = 1
    S = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i != j and ((i, j) not in forced_pairs and (j, i) not in forced_pairs):
                S[i][j] = 1
    memo_violation = {}
    total_violations = 0
    for r in range(R):
        ordering = tuple(rankings[r])
        total_violations += violation_count(ordering, M, memo_violation)
    memo_missed = {}
    total_missed = 0
    for r in range(R):
        ordering = tuple(rankings[r])
        total_missed += missed_order_count(ordering, S, memo_missed)
    denom = R * (k * (k - 1) / 2.0)
    cost = (total_violations + GAMMA*total_missed) / denom
    return cost

def connex_subtree_plus_missed_cost_of_tree(parent, rankings, position):
    R = len(rankings)
    if R == 0:
        return 0.0
    k = len(rankings[0])
    total_remaining = 0  
    for ranking in rankings:
        integrated = set()  
        L = 0
        for candidate in ranking:
            if L == 0:
                integrated.add(candidate)
                L = 1
            else:
                connected = False
                p = parent[candidate]
                if p is not None and p in integrated:
                    connected = True
                else:
                    for x in integrated:
                        if parent[x] == candidate:
                            connected = True
                            break
                if connected:
                    integrated.add(candidate)
                    L += 1
                else:
                    break
        remaining = k - L
        total_remaining += remaining
    normalized_cost = total_remaining / (R * k)
    
    forced_pairs = build_forced_pairs(parent)
    n = len(parent)
    S = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i != j and ((i, j) not in forced_pairs and (j, i) not in forced_pairs):
                S[i][j] = 1
    memo_missed = {}
    total_missed = 0
    for r in range(R):
        ordering = tuple(rankings[r])
        total_missed += missed_order_count(ordering, S, memo_missed)
    denom = R * (k * (k - 1) / 2.0)
    return normalized_cost+( GAMMA*total_missed/ denom)

def random_tree(n, root=0):
    prufer = [seeded.randrange(n) for _ in range(n - 2)]
    degree = [1] * n
    for x in prufer:
        degree[x] += 1
    edges = []
    leaf_set = set(i for i in range(n) if degree[i] == 1)
    for x in prufer:
        leaf = min(leaf_set)  # tie-break
        edges.append((leaf, x))
        degree[leaf] -= 1
        if degree[leaf] == 0:
            leaf_set.remove(leaf)
        degree[x] -= 1
        if degree[x] == 1:
            leaf_set.add(x)
    u, v = sorted(leaf_set)
    edges.append((u, v))
    adj = {i: [] for i in range(n)}
    for (u, v) in edges:
        adj[u].append(v)
        adj[v].append(u)
    parent = [None] * n
    visited = [False] * n
    queue = [root]
    visited[root] = True
    parent[root] = None
    while queue:
        u = queue.pop(0)
        for v in adj[u]:
            if not visited[v]:
                visited[v] = True
                parent[v] = u
                queue.append(v)
    return parent

def re_root_tree(parent, new_root):
    new_parent = parent.copy()
    cur = new_root
    prev = None
    while cur is not None:
        next_node = new_parent[cur]
        new_parent[cur] = prev
        prev = cur
        cur = next_node
    return new_parent

def cost_of_tree_re_rooted(parent, rankings, position):
    R = len(rankings)
    if R == 0:
        return 0.0
    n = len(parent)
    rerooted_trees = {}
    forced_pairs_by_root = {}
    for candidate in range(n):
        rt = re_root_tree(parent, candidate)
        rerooted_trees[candidate] = rt
        forced_pairs_by_root[candidate] = build_forced_pairs(rt)
    ranking_pairs_list = []
    for r in rankings:
        pair_set = set()
        for i in range(len(r)):
            for j in range(i + 1, len(r)):
                pair_set.add((r[i], r[j]))
        ranking_pairs_list.append(pair_set)
    total_cost = 0.0
    for r in range(R):
        new_root = rankings[r][0]
        forced_pairs = forced_pairs_by_root[new_root]
        violation_r = 0
        missed_r = 0
        for (i, j) in forced_pairs:
            if i in position[r] and j in position[r]:
                if position[r][j] < position[r][i]:
                    violation_r += 1
        for (i, j) in ranking_pairs_list[r]:
            if (i, j) not in forced_pairs and (j, i) not in forced_pairs:
                missed_r += 1
        
        k = len(rankings[r])
        denom = k * (k - 1) / 2.0
        total_cost += (violation_r + GAMMA*missed_r) / denom
    
    return total_cost / R



COST_OF_TREE_FUNCTION = cost_of_tree_re_rooted #TODO: maybe refactor this

def local_search(parent, rankings, position, max_iter=100):
    n = len(parent)
    current_cost = COST_OF_TREE_FUNCTION(parent, rankings, position)
    iters = 0
    while iters < max_iter:
        best_move = None
        best_cost = current_cost
        children = [[] for _ in range(n)]
        for i in range(n):
            p = parent[i]
            if p is not None:
                children[p].append(i)
        for c in range(n):
            p = parent[c]
            if p is None:
                continue 
            subtree = get_subtree_nodes(parent, c)
            parent[c] = None
            for x in range(n):
                if x in subtree:
                    continue  # avoid cycle
                if x == p:
                    continue 
                parent[c] = x
                new_cost = COST_OF_TREE_FUNCTION(parent, rankings, position)
                if new_cost < best_cost:
                    best_cost = new_cost
                    best_move = (p, c, x)
                parent[c] = None
            parent[c] = p 
        if best_move is None:
            break
        else:
            old_p, c, new_p = best_move
            parent[c] = new_p
            current_cost = best_cost
            iters += 1
    return parent


def print_tree(parent):
    n = len(parent)
    children = [[] for _ in range(n)]
    root = None
    for i in range(n):
        p = parent[i]
        if p is not None:
            children[p].append(i)
        else:
            root = i
    def dfs_print(node, level=0):
        print("  " * level + str(node))
        for ch in children[node]:
            dfs_print(ch, level + 1)
    print("Tree structure (root at top):")
    if root is not None:
        dfs_print(root)
    else:
        print(" (No root found)")

def draw_tree(parent):
    G = nx.DiGraph()
    n = len(parent)
    G.add_nodes_from(range(n))
    for i in range(n):
        p = parent[i]
        if p is not None:
            G.add_edge(p, i)
    try:
        pos = nx.nx_agraph.graphviz_layout(G, prog='dot')
    except Exception:
        pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_size=800, node_color='lightblue',
            arrows=True, arrowstyle='-|>')
    plt.title("Final Ranking Tree")
    plt.show()


def main():
    csv_path = r"ssp\borda4.csv"
    rankings, k = read_csv_and_build_rankings(csv_path)
    print(f"Loaded {len(rankings)} voters' rankings from {csv_path}.")
    print(f"Each ranking : {k} preferred candidates.")
    position = build_position_lookup(rankings)
    n = 11 
    num_trials = 2
    best_tree = None
    best_cost = float('inf')
    for trial in range(num_trials):
        parent_candidate = random_tree(n, root=0)
        optimized_tree = local_search(parent_candidate, rankings, position, max_iter=100)
        cost_candidate = COST_OF_TREE_FUNCTION(optimized_tree, rankings, position)
        print(f"Trial {trial+1}/{num_trials} cost: {cost_candidate:.6f}")
        if cost_candidate < best_cost:
            best_cost = cost_candidate
            best_tree = optimized_tree[:]
    print("\nBest tree found:")
    print_tree(best_tree)
    print(f"Final cost = {best_cost:.6f}")
    draw_tree(best_tree)

if __name__ == "__main__":
    main()
