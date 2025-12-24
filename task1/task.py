import os
import numpy as np


def main(csv_text: str, root: str) -> tuple[list[list[int]], ...]:

    rows = [row.strip() for row in csv_text.splitlines() if row.strip()]

    edge_pairs = []
    node_set = set()

    for row in rows:
        left, right = (x.strip() for x in row.split(",", 1))
        node_set.update((left, right))
        edge_pairs.append((left, right))

    ordered_nodes = [root] + sorted(v for v in node_set if v != root)
    count = len(ordered_nodes)

    index = {node: pos for pos, node in enumerate(ordered_nodes)}

    base = np.zeros((count, count), dtype=bool)
    for a, b in edge_pairs:
        base[index[a], index[b]] = True

    r1 = base.astype(int)
    r2 = r1.T

    closure = base.copy()
    for _ in range(count - 1):
        closure |= closure @ base

    r3 = (closure & ~base).astype(int)
    r4 = r3.T

    parent = r2.astype(bool)
    r5 = np.zeros((count, count), dtype=int)

    for i in range(count):
        for j in range(i + 1, count):
            if (parent[i] & parent[j]).any():
                r5[i, j] = r5[j, i] = 1

    return (
        r1.tolist(),
        r2.tolist(),
        r3.tolist(),
        r4.tolist(),
        r5.tolist()
    )


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(base_dir, "task1.csv")

    with open(file_path, encoding="utf-8") as f:
        data = f.read()

    root = input("Введите значение корневой вершины: ").strip()
    results = main(data, root)

    names = [
        "r1 (управление)",
        "r2 (подчинение)",
        "r3 (опосредованное управление)",
        "r4 (опосредованное подчинение)",
        "r5 (соподчинение)"
    ]

    for title, mat in zip(names, results):
        print(f"\nМатрица для отношения {title}:")
        for row in mat:
            print(row)
