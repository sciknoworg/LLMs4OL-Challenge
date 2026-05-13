import networkx as nx
from collections import defaultdict

TAXONOMY_RELATIONS = {"is-a"}


def build_graph(triples):
    G = nx.DiGraph()
    for s, p, o in triples:
        G.add_edge(s, o, label=p)
    return G


def edge_f1(gold_edges, pred_edges):
    gold_set = set(gold_edges)
    pred_set = set(pred_edges)

    intersection = gold_set & pred_set

    if len(pred_set) == 0 or len(gold_set) == 0:
        return 0.0

    precision = len(intersection) / len(pred_set)
    recall = len(intersection) / len(gold_set)

    if precision + recall == 0:
        return 0.0

    return 2 * precision * recall / (precision + recall)


def get_neighborhood(triples):
    neigh = defaultdict(set)
    for s, p, o in triples:
        neigh[s].add((p, o))
    return neigh


def neighborhood_similarity(gold_triples, pred_triples):
    gold_neigh = get_neighborhood(gold_triples)
    pred_neigh = get_neighborhood(pred_triples)

    all_nodes = set(gold_neigh.keys()) | set(pred_neigh.keys())
    scores = []

    for n in all_nodes:
        g = gold_neigh.get(n, set())
        p = pred_neigh.get(n, set())

        union = g | p
        if len(union) == 0:
            continue

        jaccard = len(g & p) / len(union)
        scores.append(jaccard)

    return sum(scores) / len(scores) if scores else 0.0


def build_taxonomy_graph(triples):
    G = nx.DiGraph()
    for s, p, o in triples:
        if p in TAXONOMY_RELATIONS:
            G.add_edge(s, o)
    return G


def taxonomy_similarity(gold_triples, pred_triples):
    G_gold = build_taxonomy_graph(gold_triples)
    G_pred = build_taxonomy_graph(pred_triples)

    all_nodes = set(G_gold.nodes()) | set(G_pred.nodes())
    scores = []

    for n in all_nodes:
        gold_anc = nx.ancestors(G_gold, n) if n in G_gold else set()
        pred_anc = nx.ancestors(G_pred, n) if n in G_pred else set()

        union = gold_anc | pred_anc
        if len(union) == 0:
            continue

        jaccard = len(gold_anc & pred_anc) / len(union)
        scores.append(jaccard)

    return sum(scores) / len(scores) if scores else 0.0


def graph_similarity(gold_triples, pred_triples):
    f1_edges = edge_f1(gold_triples, pred_triples)
    neigh_sim = neighborhood_similarity(gold_triples, pred_triples)
    tax_sim = taxonomy_similarity(gold_triples, pred_triples)

    final_score = (f1_edges + neigh_sim + tax_sim) / 3

    return {
        "edge_f1": f1_edges,
        "neighborhood_similarity": neigh_sim,
        "taxonomy_similarity": tax_sim,
        "graph_similarity": final_score,
    }

gold = [
    ("temperature sensor", "instance-of", "sensor"),
    ("sensor", "is-a", "device"),
    ("temperature sensor", "measures", "room temperature"),
]

pred = [
    ("temperature sensor", "instance-of", "device"),  # wrong triplet!
    ("sensor", "is-a", "device"),
    ("temperature sensor", "measures", "room temperature"),
]

score = graph_similarity(gold, pred)
print(score)

