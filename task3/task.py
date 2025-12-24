import json
import numpy as np
from typing import Any, List, Tuple


def _warshall_bool(mat: np.ndarray) -> np.ndarray:
    n = mat.shape[0]
    clo = mat.copy()
    for k in range(n):
        for i in range(n):
            clo[i, :] = clo[i, :] | (clo[i, k] & clo[k, :])
    return clo


def _scc_from_equivalence(eq_closure: np.ndarray) -> List[List[int]]:
    n = eq_closure.shape[0]
    used = [False] * n
    comps: List[List[int]] = []

    for i in range(n):
        if used[i]:
            continue
        comp = []
        for j in range(n):
            if (not used[j]) and eq_closure[i, j] and eq_closure[j, i]:
                used[j] = True
                comp.append(j + 1)  # обратно к 1..n
        comps.append(sorted(comp))
    return comps


def _normalize_ranking(rank: Any) -> List[List[int]]:
    out: List[List[int]] = []
    for cl in rank:
        if isinstance(cl, list):
            out.append([int(x) for x in cl])
        else:
            out.append([int(cl)])
    return out


def _build_Y(rank: List[List[int]], n: int) -> np.ndarray:
    pos = [0] * n
    for p, cluster in enumerate(rank):
        for obj in cluster:
            pos[obj - 1] = p

    Y = np.zeros((n, n), dtype=int)
    for i in range(n):
        for j in range(n):
            if pos[i] >= pos[j]:
                Y[i, j] = 1
    return Y


def _toposort(order_mat: np.ndarray) -> List[int]:
    # возвращает порядок индексов вершин графа кластеров
    m = order_mat.shape[0]
    seen = [False] * m
    out: List[int] = []

    def dfs(v: int) -> None:
        seen[v] = True
        for u in range(m):
            if order_mat[v, u] == 1 and not seen[u]:
                dfs(u)
        out.append(v)

    for v in range(m):
        if not seen[v]:
            dfs(v)

    out.reverse()
    return out


def _compute_kernel_and_ranking(json_a: str, json_b: str) -> Tuple[List[List[int]], List[Any]]:
    ra_raw = json.loads(json_a)
    rb_raw = json.loads(json_b)

    ra = _normalize_ranking(ra_raw)
    rb = _normalize_ranking(rb_raw)

    items = set()
    for r in (ra, rb):
        for cl in r:
            items.update(cl)

    if not items:
        return [], []

    n = max(items)

    YA = _build_Y(ra, n)
    YB = _build_Y(rb, n)

    # Этап 1: ядро противоречий
    YAB = YA * YB
    YAB_p = YA.T * YB.T

    kernel: List[List[int]] = []
    for i in range(n):
        for j in range(i + 1, n):
            if YAB[i, j] == 0 and YAB_p[i, j] == 0:
                kernel.append([i + 1, j + 1])

    # Этап 2: согласование
    C = YA * YB
    for a, b in kernel:
        i, j = a - 1, b - 1
        C[i, j] = 1
        C[j, i] = 1

    E = (C * C.T).astype(bool)
    E_star = _warshall_bool(E)

    clusters = _scc_from_equivalence(E_star)

    k = len(clusters)
    Gc = np.zeros((k, k), dtype=int)
    for i in range(k):
        for j in range(k):
            if i == j:
                continue
            vi = clusters[i][0] - 1
            vj = clusters[j][0] - 1
            if C[vi, vj] == 1:
                Gc[i, j] = 1

    order = _toposort(Gc)

    consistent: List[Any] = []
    for idx in order:
        cl = clusters[idx]
        consistent.append(cl[0] if len(cl) == 1 else cl)

    return kernel, consistent


def main(json_a: str, json_b: str) -> str:
    _, consistent = _compute_kernel_and_ranking(json_a, json_b)
    return json.dumps(consistent, ensure_ascii=False)


if __name__ == "__main__":
    def _read(path: str) -> str:
        with open(path, "r", encoding="utf-8") as f:
            return f.read().strip()

    a = _read("range_a.json")
    b = _read("range_b.json")

    print(main(a, b))

    ker, cons = _compute_kernel_and_ranking(a, b)
    print("kernel:", ker)
    print("consistent:", cons)
