import json
import numpy as np


def read_json_file(file_path: str) -> str:
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read().strip()


def membership(x: float, points) -> float:
    pts = sorted(points, key=lambda p: p[0])
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]

    if len(pts) < 2:
        return 0.0

    if x <= xs[0]:
        return float(ys[0])
    if x >= xs[-1]:
        return float(ys[-1])

    for i in range(len(xs) - 1):
        x0, x1 = xs[i], xs[i + 1]
        if x0 <= x <= x1:
            y0, y1 = ys[i], ys[i + 1]
            dx = x1 - x0
            if dx == 0:
                return float((y0 + y1) / 2)
            return float(y0 + (y1 - y0) * (x - x0) / dx)

    return 0.0


def fuzzify(value: float, ling_var) -> dict:
    res = {}
    for term in ling_var:
        res[term["id"]] = membership(value, term["points"])
    return res


def get_output_range(control_ling_var) -> tuple[float, float]:
    xs = []
    for term in control_ling_var:
        xs.extend(p[0] for p in term["points"])
    if not xs:
        return 0.0, 10.0
    return float(min(xs)), float(max(xs))


def aggregate_membership(mu_input: dict, rules, control_ling_var, s_values: np.ndarray) -> np.ndarray:
    mu_agg = np.zeros(len(s_values), dtype=float)

    for input_id, output_id in rules:
        act = float(mu_input.get(input_id, 0.0))
        if act == 0.0:
            continue

        out_term = next((t for t in control_ling_var if t["id"] == output_id), None)
        if out_term is None:
            continue

        mu_out = np.array([membership(s, out_term["points"]) for s in s_values], dtype=float)
        mu_agg = np.maximum(mu_agg, np.minimum(act, mu_out))

    return mu_agg


def defuzzify_mean_of_max(s_values: np.ndarray, mu_agg: np.ndarray) -> float:
    if mu_agg.size == 0:
        return 0.0
    mx = float(np.max(mu_agg))
    if mx == 0.0:
        return 0.0

    idx = np.where(np.isclose(mu_agg, mx, atol=1e-6))[0]
    if idx.size == 0:
        return 0.0

    return float((s_values[idx[0]] + s_values[idx[-1]]) / 2.0)


def compute_optimal_control(T: float, temp_ling_var, control_ling_var, rules, steps: int = 1001) -> float:
    s_min, s_max = get_output_range(control_ling_var)
    s_values = np.linspace(s_min, s_max, steps)

    mu_input = fuzzify(float(T), temp_ling_var)
    mu_agg = aggregate_membership(mu_input, rules, control_ling_var, s_values)

    return defuzzify_mean_of_max(s_values, mu_agg)


def main(lvinput_json: str, lvoutput_json: str, rules_json: str, T: float) -> float:
    temp_data = json.loads(lvinput_json)
    control_data = json.loads(lvoutput_json)
    rules = json.loads(rules_json)

    temp_ling_var = temp_data["температура"]
    control_ling_var = control_data["нагрев"]

    return compute_optimal_control(float(T), temp_ling_var, control_ling_var, rules)


if __name__ == "__main__":
    lvinput_json = read_json_file("lvinput.json")
    lvoutput_json = read_json_file("lvoutput.json")
    rules_json = read_json_file("rules.json")

    T = 19.0
    optimal_s = main(lvinput_json, lvoutput_json, rules_json, T)
    print(f"Для температуры {T:.1f}°C оптимальное управление: {optimal_s:.2f}")
