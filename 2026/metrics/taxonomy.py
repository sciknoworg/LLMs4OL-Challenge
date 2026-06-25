from typing import Dict, List, Union
from rapidfuzz import fuzz
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm

_model = SentenceTransformer("nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True)


def exact_match(y_true: List[Dict[str, str]], y_pred: List[Dict[str, str]]) -> Dict[str, Union[float, int]]:
    # Convert ground truth and predictions to sets of tuples for easy comparison
    ground_truth_set = {(item["parent"].lower(), item["child"].lower()) for item in y_true}
    predictions_set = {(item["parent"].lower(), item["child"].lower()) for item in y_pred}
    total_predicted = len(predictions_set)
    total_ground_truth = len(ground_truth_set)
    # Calculate correctly predicted pairs (intersection of sets)
    correct_predictions = ground_truth_set.intersection(predictions_set)
    total_correct = len(correct_predictions)
    # Calculate precision, recall, and F1-score
    precision = total_correct / total_predicted if total_predicted > 0 else 0
    recall = total_correct / total_ground_truth if total_ground_truth > 0 else 0
    if precision + recall > 0:
        f1_score = 2 * (precision * recall) / (precision + recall)
    else:
        f1_score = 0
    return {
        "f1_score": f1_score,
        "precision": precision,
        "recall": recall
    }

def normalize(text: str) -> str:
    return " ".join(text.lower().strip().split())

def edge_similarity(pred_edge, gold_edge):
    parent_score = fuzz.ratio(pred_edge[0], gold_edge[0]) / 100.0
    child_score = fuzz.ratio(pred_edge[1], gold_edge[1]) / 100.0
    return (parent_score + child_score) / 2

def fuzzy_match(y_true: List[Dict[str, str]], y_pred: List[Dict[str, str]], threshold: float = 0.90):
    gold = [(normalize(x["parent"]), normalize(x["child"])) for x in y_true]
    pred = [(normalize(x["parent"]), normalize(x["child"])) for x in y_pred]

    if not gold or not pred:
        return {"f1_score": 0.0, "precision": 0.0, "recall": 0.0}

    cost_matrix = np.zeros((len(pred), len(gold)))
    for i, p in enumerate(pred):
        for j, g in enumerate(gold):
            cost_matrix[i, j] = 1.0 - edge_similarity(p, g)

    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    tp_score = 0.0
    for r, c in zip(row_ind, col_ind):
        sim = 1.0 - cost_matrix[r, c]
        if sim >= threshold:
            tp_score += sim

    precision = tp_score / len(pred) if pred else 0
    recall = tp_score / len(gold) if gold else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0

    return {
        "f1_score": f1,
        "precision": precision,
        "recall": recall
    }

def semantic_match(y_true, y_pred, threshold: float = 0.90, batch_size: int = 2048):
    gold = [(normalize(x["parent"]), normalize(x["child"])) for x in y_true]
    pred = [(normalize(x["parent"]), normalize(x["child"])) for x in y_pred]

    if not gold or not pred:
        return {"f1_score": 0.0, "precision": 0.0, "recall": 0.0}

    terms = sorted({t for edge in gold + pred for t in edge})
    embeddings = _model.encode(
        terms,
        convert_to_numpy=True,
        normalize_embeddings=True,
        batch_size=2048,  
    )
    embedding_dict = dict(zip(terms, embeddings))

    pred_parent = np.stack([embedding_dict[p[0]] for p in pred]).astype(np.float32)
    pred_child  = np.stack([embedding_dict[p[1]] for p in pred]).astype(np.float32)
    gold_parent = np.stack([embedding_dict[g[0]] for g in gold]).astype(np.float32)
    gold_child  = np.stack([embedding_dict[g[1]] for g in gold]).astype(np.float32)

    n_pred, n_gold = len(pred), len(gold)
    cost_matrix = np.empty((n_pred, n_gold), dtype=np.float32)

    for start in tqdm(range(0, n_pred, batch_size)):
        end = min(start + batch_size, n_pred)
        parent_sim = np.clip(pred_parent[start:end] @ gold_parent.T, 0.0, 1.0)
        child_sim  = np.clip(pred_child[start:end]  @ gold_child.T,  0.0, 1.0)
        sim_batch = (parent_sim + child_sim) / 2.0
        cost_matrix[start:end, :] = 1.0 - sim_batch

    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    tp_score = 0.0
    for r, c in zip(row_ind, col_ind):
        sim = 1.0 - cost_matrix[r, c]
        if sim >= threshold:
            tp_score += sim

    precision = tp_score / n_pred if n_pred else 0
    recall = tp_score / n_gold if n_gold else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0

    return {"precision": precision, "recall": recall, "f1_score": f1}

def evaluate(y_true, y_predict):
    return {
        "exact_match": exact_match(y_true, y_predict),
        "semantic_match": semantic_match(y_true, y_predict),
        "fuzzy_match": fuzzy_match(y_true, y_predict)
    }
