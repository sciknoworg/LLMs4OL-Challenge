import networkx as nx
from collections import defaultdict
from rapidfuzz import fuzz
from sentence_transformers import SentenceTransformer
import numpy as np
from scipy.optimize import linear_sum_assignment

model = SentenceTransformer("nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True)

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
    
    if len(gold_set) == 0 and len(pred_set) == 0:
        return 1.0
    
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
        neigh[s].add(('OUT', p, o))
        neigh[o].add(('IN', p, s))
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

def normalize(text):
    return " ".join(str(text).lower().strip().split())

def normalize_triples(triples):
    return [tuple(normalize(x) for x in t) for t in triples]

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
        gold_desc = nx.descendants(G_gold, n) if n in G_gold else set()
        pred_desc = nx.descendants(G_pred, n) if n in G_pred else set()

        gold_rel = gold_anc | gold_desc
        pred_rel = pred_anc | pred_desc

        union = gold_rel | pred_rel
        if len(union) == 0:
            continue

        jaccard = len(gold_rel & pred_rel) / len(union)
        scores.append(jaccard)

    return sum(scores) / len(scores) if scores else 0.0


def triple_similarity_fuzzy(t1, t2):
    s = fuzz.ratio(t1[0], t2[0]) / 100
    p = fuzz.ratio(t1[1], t2[1]) / 100
    o = fuzz.ratio(t1[2], t2[2]) / 100
    return (s + p + o) / 3

def align_triples_fuzzy(gold, pred, threshold=0.90):
    if not gold or not pred:
        return pred

    cost_matrix = np.zeros((len(pred), len(gold)))
    for i, p in enumerate(pred):
        for j, g in enumerate(gold):
            cost_matrix[i, j] = 1.0 - triple_similarity_fuzzy(p, g)

    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    matched = list(pred)
    for r, c in zip(row_ind, col_ind):
        if 1.0 - cost_matrix[r, c] >= threshold:
            matched[r] = gold[c]
    return matched

def build_embeddings(gold, pred):
    terms = set()
    for triple in gold + pred:
        terms.update(triple)
    vectors = model.encode(list(terms), normalize_embeddings=True, convert_to_numpy=True)
    return dict(zip(terms, vectors))

def triple_similarity_semantic(t1, t2, emb):
    s = np.dot(emb[t1[0]], emb[t2[0]])
    p = np.dot(emb[t1[1]], emb[t2[1]])
    o = np.dot(emb[t1[2]], emb[t2[2]])
    return (s + p + o) / 3

def align_triples_semantic(gold, pred, emb, threshold=0.80):
    if not gold or not pred:
        return pred

    cost_matrix = np.zeros((len(pred), len(gold)))
    for i, p in enumerate(pred):
        for j, g in enumerate(gold):
            cost_matrix[i, j] = 1.0 - triple_similarity_semantic(p, g, emb)

    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    matched = list(pred)
    for r, c in zip(row_ind, col_ind):
        if 1.0 - cost_matrix[r, c] >= threshold:
            matched[r] = gold[c]
    return matched

def exact_match(gold_triples, pred_triples):
    gold_triples = normalize_triples(gold_triples)
    pred_triples = normalize_triples(pred_triples)
    
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


def fuzzy_match(gold_triples, pred_triples, threshold=0.90):
    gold_triples = normalize_triples(gold_triples)
    pred_triples = normalize_triples(pred_triples)
    
    aligned_pred = align_triples_fuzzy(gold_triples, pred_triples, threshold)
    edge_score = edge_f1(gold_triples, aligned_pred)
    neigh_score = neighborhood_similarity(gold_triples, aligned_pred)
    tax_score = taxonomy_similarity(gold_triples, aligned_pred)
    return {
        "edge_f1": edge_score,
        "neighborhood_similarity": neigh_score,
        "taxonomy_similarity": tax_score,
        "graph_similarity": (edge_score + neigh_score + tax_score) / 3,
    }

def semantic_match(gold_triples, pred_triples, threshold=0.90):
    gold_triples = normalize_triples(gold_triples)
    pred_triples = normalize_triples(pred_triples)
    
    embeddings = build_embeddings(gold_triples, pred_triples)
    aligned_pred = align_triples_semantic(gold_triples, pred_triples, embeddings, threshold)
    edge_score = edge_f1(gold_triples, aligned_pred)
    neigh_score = neighborhood_similarity(gold_triples, aligned_pred)
    tax_score = taxonomy_similarity(gold_triples, aligned_pred)
    return {
        "edge_f1": edge_score,
        "neighborhood_similarity": neigh_score,
        "taxonomy_similarity": tax_score,
        "graph_similarity": (edge_score + neigh_score + tax_score) / 3,
    }


def evaluate(y_true, y_predict):
    return {
        "exact_match": exact_match(y_true, y_predict),
        "semantic_match": semantic_match(y_true, y_predict),
        "fuzzy_match": fuzzy_match(y_true, y_predict)
    }
