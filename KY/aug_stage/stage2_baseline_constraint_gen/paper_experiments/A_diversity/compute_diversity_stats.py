#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


def _load_json(p: Path) -> Dict[str, Any]:
    return json.loads(p.read_text(encoding="utf-8"))


def _iter_persona_constraints(obj: Dict[str, Any]) -> Iterable[Tuple[int, Dict[str, Any]]]:
    # yield (persona_index, constraint_dict)
    personas = obj.get("group_personas") or []
    for i, p in enumerate(personas):
        sr = (p.get("structured_requirement") or {})
        for bucket in ("hard_constraints", "soft_constraints"):
            cons = sr.get(bucket) or []
            if isinstance(cons, list):
                for c in cons:
                    if isinstance(c, dict):
                        yield i, c


def _quantiles(xs: List[float], ps: List[float]) -> Dict[str, float]:
    if not xs:
        return {str(p): float("nan") for p in ps}
    xs2 = sorted(xs)
    out: Dict[str, float] = {}
    n = len(xs2)
    for p in ps:
        # nearest-rank over sorted list
        k = int(round(p * (n - 1)))
        k = max(0, min(n - 1, k))
        out[str(p)] = float(xs2[k])
    return out


def _as_strings(v: Any) -> List[str]:
    if v is None:
        return []
    if isinstance(v, str):
        return [v]
    if isinstance(v, list):
        return [str(x) for x in v]
    return [str(v)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", required=True, help="outputs_stage2/.../<run>/ directory (contains *.json + meta/)")
    ap.add_argument("--out", default="", help="write stats json")
    ap.add_argument("--md", default="", help="write markdown summary")
    ap.add_argument(
        "--keywords",
        default="park,trail,museum,garden,zoo,aquarium,market,brewery,view,sunset,river,lake,beach,mountain",
        help="comma-separated keywords for creative-proxy detection",
    )
    args = ap.parse_args()

    in_dir = Path(args.in_dir).expanduser().resolve()
    files = sorted([p for p in in_dir.glob("*.json") if p.is_file()])
    if not files:
        raise SystemExit(f"No json files found in {in_dir}")

    keywords = [k.strip().lower() for k in args.keywords.split(",") if k.strip()]

    # Per-persona aggregates
    per_persona_counts: List[int] = []
    per_persona_field_sets: List[set[str]] = []
    per_persona_cat_sets: List[set[str]] = []

    field_ctr = Counter()
    op_ctr = Counter()
    cat_ctr = Counter()

    total_constraints = 0
    keyword_field_constraints = 0
    keyword_value_constraints = 0
    keyword_proxy_constraints = 0

    def _has_keyword(v: Any) -> bool:
        ss = " ".join([s.lower() for s in _as_strings(v)])
        return any(k in ss for k in keywords)

    def _cat(field: str) -> str:
        if not field:
            return "other"
        if field.startswith("restaurant."):
            return "restaurant"
        if field.startswith("accommodation."):
            return "accommodation"
        if field.startswith("attraction."):
            return "attraction"
        if field.startswith("flight."):
            return "flight"
        if field.startswith("ground."):
            return "ground"
        return "other"

    for p in files:
        obj = _load_json(p)
        personas = obj.get("group_personas") or []

        ex_counts = [0 for _ in range(len(personas))]
        ex_field_sets = [set() for _ in range(len(personas))]
        ex_cat_sets = [set() for _ in range(len(personas))]

        for idx, c in _iter_persona_constraints(obj):
            field = str(c.get("field") or "")
            op = str(c.get("op") or "")
            val = c.get("value")

            ex_counts[idx] += 1
            ex_field_sets[idx].add(field)
            ex_cat_sets[idx].add(_cat(field))

            field_ctr[field] += 1
            op_ctr[op] += 1
            cat_ctr[_cat(field)] += 1

            total_constraints += 1
            is_kw_field = field.endswith("_keyword")
            is_kw_val = _has_keyword(val)
            if is_kw_field:
                keyword_field_constraints += 1
            if is_kw_val:
                keyword_value_constraints += 1
            if is_kw_field or is_kw_val:
                keyword_proxy_constraints += 1

        per_persona_counts.extend(ex_counts)
        per_persona_field_sets.extend(ex_field_sets)
        per_persona_cat_sets.extend(ex_cat_sets)

    counts_f = [float(x) for x in per_persona_counts]
    stats = {
        "n_examples": len(files),
        "n_personas_total": len(per_persona_counts),
        "constraints_per_persona": {
            "mean": float(statistics.mean(counts_f)) if counts_f else 0.0,
            "median": float(statistics.median(counts_f)) if counts_f else 0.0,
            "min": int(min(per_persona_counts)) if per_persona_counts else 0,
            "max": int(max(per_persona_counts)) if per_persona_counts else 0,
            "quantiles": _quantiles(counts_f, [0.1, 0.25, 0.5, 0.75, 0.9]),
        },
        "field_diversity": {
            "unique_fields_total": len([k for k in field_ctr.keys() if k]),
            "avg_unique_fields_per_persona": float(statistics.mean([len(s) for s in per_persona_field_sets])) if per_persona_field_sets else 0.0,
            "avg_unique_categories_per_persona": float(statistics.mean([len(s) for s in per_persona_cat_sets])) if per_persona_cat_sets else 0.0,
            "category_counts": dict(cat_ctr),
        },
        "operator_diversity": {
            "unique_ops_total": len([k for k in op_ctr.keys() if k]),
            "top_ops": op_ctr.most_common(30),
        },
        "creative_proxy": {
            "total_constraints": total_constraints,
            "keyword_field_ratio": (keyword_field_constraints / total_constraints) if total_constraints else 0.0,
            "keyword_value_ratio": (keyword_value_constraints / total_constraints) if total_constraints else 0.0,
            "keyword_proxy_ratio": (keyword_proxy_constraints / total_constraints) if total_constraints else 0.0,
            "keywords": keywords,
        },
        "top_fields": field_ctr.most_common(30),
    }

    print(json.dumps(stats, ensure_ascii=False, indent=2))

    if args.out:
        out_p = Path(args.out).expanduser().resolve()
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")

    if args.md:
        md_p = Path(args.md).expanduser().resolve()
        md_p.parent.mkdir(parents=True, exist_ok=True)
        md = []
        md.append("# A. Diversity / Creativity Stats")
        md.append("")
        md.append(f"- Examples: **{stats['n_examples']}**")
        md.append(f"- Personas: **{stats['n_personas_total']}**")
        c = stats["constraints_per_persona"]
        md.append(f"- Constraints/persona (mean/median): **{c['mean']:.2f} / {c['median']:.2f}** (min={c['min']}, max={c['max']})")
        md.append("")
        md.append("## Category counts (by constraint)")
        md.append("")
        for k, v in sorted(stats["field_diversity"]["category_counts"].items(), key=lambda x: x[1], reverse=True):
            md.append(f"- {k}: {v}")
        md.append("")
        md.append("## Operator (top)")
        md.append("")
        for op, n in stats["operator_diversity"]["top_ops"][:10]:
            md.append(f"- {op}: {n}")
        md.append("")
        cp = stats["creative_proxy"]
        md.append("## Creative proxy ratios")
        md.append("")
        md.append(f"- keyword_field_ratio: **{cp['keyword_field_ratio']:.3f}**")
        md.append(f"- keyword_value_ratio: **{cp['keyword_value_ratio']:.3f}**")
        md.append(f"- keyword_proxy_ratio (union): **{cp['keyword_proxy_ratio']:.3f}**")
        md_p.write_text("\n".join(md) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
