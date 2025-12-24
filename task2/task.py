import os
import math
import numpy as np
from typing import Tuple, List, Iterable


def parse_edges(csv_text: str) -> Tuple[List[Tuple[str, str]], List[str]]:
    lines = [ln.strip() for ln in csv_text.splitlines() if ln.strip()]
    edges: List[Tuple[str, str]] = []
    nodes = set()

    for ln in lines:
        a, b = (x.strip() for x in ln.split(",", 1))
        edges.append((a, b))
        nodes.update((a, b))

    return edges, sorted(nodes)


def order_vertices(all_vertices: List[str], root: str) -> List[str]:
    rest = [v for v in all_vertices if v != root]
    rest.sort()
    return [root] + rest


def index_map(vertices: List[str]) -> dict[str, int]:
    return {v: i for i, v in enumerate(vertices)}


def compute_entropy(mats: List[np.ndarray]) -> Tuple[float, float]:
    n = mats[0].shape[0]
    k = len(mats)

    acc = 0.0
    denom = (n - 1)

    for m in mats:
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                p = m[i, j] / denom
                if p > 0:
                    acc += p * math.log2(p)

    H = -acc
    H_max = (1 / math.e) * n * k
    h = (H / H_max) if H_max > 0 else 0.0
    return H, h


def generate_all_edge_permutations(
    edges: List[Tuple[str, str]],
    vertices: List[str]
) -> List[List[Tuple[str, str]]]:
    n = len(vertices)

    all_possible = [
        (vertices[i], vertices[j])
        for i in range(n)
        for j in range(n)
        if i != j
    ]

    existing = set(edges)
    candidates = [e for e in all_possible if e not in existing]

    variants: List[List[Tuple[str, str]]] = []
    for remove_pos in range(len(edges)):
        for new_e in candidates:
            changed = edges.copy()
            changed[remove_pos] = new_e
            variants.append(changed)

    return variants


def build_relations(
    edges: List[Tuple[str, str]],
    vertices: List[str],
    idx: dict[str, int]
) -> List[np.ndarray]:
    n = len(vertices)

    adj = np.zeros((n, n), dtype=bool)
    for a, b in edges:
        adj[idx[a], idx[b]] = True

    r1 = adj.astype(int)
    r2 = r1.T

    closure = adj.copy()
    for _ in range(1, n):
        closure = closure | (closure @ adj)

    r3 = (closure & ~adj).astype(int)
    r4 = r3.T

    parents = r2.astype(bool)
    r5 = np.zeros((n, n), dtype=int)
    for i in range(n):
        for j in range(i + 1, n):
            if np.any(parents[i] & parents[j]):
                r5[i, j] = r5[j, i] = 1

    return [r1, r2, r3, r4, r5]


def main(s: str, e: str) -> Tuple[float, float]:
    base_edges, all_nodes = parse_edges(s)
    vertices = order_vertices(all_nodes, e)
    idx = index_map(vertices)

    variants = generate_all_edge_permutations(base_edges, vertices)

    best_H = -float("inf")
    best_h = 0.0
    best_edges = None

    for edges in variants:
        mats = build_relations(edges, vertices, idx)
        H, h_val = compute_entropy(mats)

        if H > best_H:
            best_H = H
            best_h = h_val
            best_edges = edges.copy()

    if best_edges is not None:
        print("\nЛучший вариант перестановки:")
        print(f"Исходные рёбра: {base_edges}")
        print(f"Новые рёбра: {best_edges}")

    return best_H, best_h


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(here, "task2.csv")

    with open(csv_path, "r", encoding="utf-8") as f:
        text = f.read()

    root = input("Введите значение корневой вершины: ").strip()
    H, h = main(text, root)

    print("\nРезультат:")
    print(f"H(M,R) = {H:.4f}")
    print(f"h(M,R) = {h:.4f}")
