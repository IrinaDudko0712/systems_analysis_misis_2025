import os


def main(csv_text: str) -> list[list[int]]:
    rows = [r for r in csv_text.strip().splitlines() if r.strip()]

    edge_list: list[tuple[str, str]] = []
    vertices = set()

    for r in rows:
        a, b = (x.strip() for x in r.split(",", 1))
        vertices.update((a, b))
        edge_list.append((a, b))

    ordered_vertices = sorted(vertices)
    size = len(ordered_vertices)

    adj = [[0 for _ in range(size)] for _ in range(size)]

    for a, b in edge_list:
        ia = ordered_vertices.index(a)
        ib = ordered_vertices.index(b)
        adj[ia][ib] = 1
        adj[ib][ia] = 1

    return adj


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(here, "task0.csv")

    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    ans = main(text)
    for line in ans:
        print(line)
