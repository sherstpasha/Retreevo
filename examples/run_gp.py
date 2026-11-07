import argparse
import json
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from thefittest.optimizers import SelfCGP
from thefittest.base import FunctionalNode, Tree

from retreevo.gp import GeneticProgrammingRetriever
from retreevo.metrics import NDCG
from retreevo.gp.operator_pool import BM25Config, NeuralConfig, _node_config
from retreevo.retrieval.model_pool import DenseModel, RerankModel, DiversityModel


def format_expression(
    tree: Tree, aliases: dict[DenseModel | RerankModel | DiversityModel, str]
) -> str:
    stack: list[str] = []
    for node in reversed(tree._nodes):
        if not isinstance(node, FunctionalNode):
            stack.append(node._name)
            continue
        arguments = [stack.pop() for _ in range(node._n_args)]
        config = _node_config(node)
        if isinstance(config, BM25Config):
            name = (
                f"bm25[algorithm={config.algorithm}, k1={config.k1!r}, "
                f"b={config.b!r}, delta={config.delta!r}, top_k={config.top_k}]"
            )
        elif isinstance(config, NeuralConfig):
            name = f"{aliases[config.model]}[top_k={config.top_k}]"
        else:
            name = node._name
        stack.append(f"{name}({', '.join(arguments)})")
    return stack[0]


def save_expressions(
    stats: dict[str, list[object]], path: Path, metric_name: str = "NDCG"
) -> None:
    aliases: dict[DenseModel | RerankModel | DiversityModel, str] = {}
    for population in stats["population_g"]:
        for tree in population:
            for node in tree._nodes:
                config = (
                    _node_config(node) if isinstance(node, FunctionalNode) else None
                )
                if isinstance(config, NeuralConfig) and config.model not in aliases:
                    if isinstance(config.model, DenseModel):
                        kind = "dense"
                    elif isinstance(config.model, DiversityModel):
                        kind = "diversity"
                    else:
                        kind = "rerank"
                    aliases[config.model] = f"{kind}_{len(aliases) + 1}"
    lines = [
        "Все позиции популяций по поколениям, включая повторы.",
        "c — исходный корпус для текущего запроса.",
        "Генотип — исходное выражение; фенотип — выражение после лечения.",
        "Если они совпадают, выражение указано один раз.",
        "",
        "МОДЕЛИ",
    ]
    lines.extend(
        f"{alias} = {json.dumps(asdict(model), ensure_ascii=False, sort_keys=True)}"
        for model, alias in aliases.items()
    )
    for generation, (genotypes, phenotypes, fitness) in enumerate(
        zip(
            stats["population_g"], stats["population_ph"], stats["fitness"], strict=True
        ),
        1,
    ):
        lines.extend(["", f"ПОКОЛЕНИЕ {generation}"])
        for index, (genotype, phenotype, score) in enumerate(
            zip(genotypes, phenotypes, fitness, strict=True), 1
        ):
            original = format_expression(genotype, aliases)
            repaired = format_expression(phenotype, aliases)
            lines.extend(
                [
                    "",
                    f"{generation:02d}.{index:02d} | {metric_name} = {float(score):.6f}",
                ]
            )
            if original == repaired:
                lines.append(original)
            else:
                lines.extend([f"Генотип: {original}", f"Фенотип: {repaired}"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def summarize(stats: dict[str, list[object]]) -> list[dict[str, int | float]]:
    history = []
    best_so_far = 0.0
    for generation, fitness in enumerate(stats["fitness"], 1):
        best_so_far = max(best_so_far, float(np.max(fitness)))
        history.append(
            {
                "generation": generation,
                "best": float(np.max(fitness)),
                "best_so_far": best_so_far,
                "mean": float(np.mean(fitness)),
                "median": float(np.median(fitness)),
                "min": float(np.min(fitness)),
                "unique_genotypes": len(
                    {tree.signature() for tree in stats["population_g"][generation - 1]}
                ),
                "unique_phenotypes": len(
                    {
                        tree.signature()
                        for tree in stats["population_ph"][generation - 1]
                    }
                ),
            }
        )
    return history


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", type=Path, default=Path(".cache/retreevo/beir/scifact_debug_2x")
    )
    parser.add_argument("--population", type=int, default=20)
    parser.add_argument("--generations", type=int, default=10)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--init-level", type=int, default=3)
    parser.add_argument("--max-level", type=int, default=3)
    parser.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    parser.add_argument(
        "--output", type=Path, default=Path(".cache/retreevo/gp_ndcg_20x10")
    )
    args = parser.parse_args()
    if args.init_level < 2 or args.max_level < args.init_level:
        parser.error("init-level must be >=2 and max-level must be >=init-level")
    if args.top_k is not None and args.top_k < 1:
        parser.error("top-k must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    report_path = args.output / "report.json"
    started = perf_counter()
    report = {
        "status": "running",
        "dataset": str(args.dataset),
        "optimizer": "SelfCGP",
        "population": args.population,
        "generations": args.generations,
        "initial_population_included": True,
        "seed": args.seed,
        "device": args.device,
        "metric": "NDCG",
        "top_k": args.top_k,
        "init_level": args.init_level,
        "max_level": args.max_level,
        "use_fitness_cache": True,
        "history": [],
    }

    def save_progress(optimizer: SelfCGP) -> None:
        report["history"] = summarize(optimizer.get_stats())
        save_expressions(optimizer.get_stats(), args.output / "expressions.txt")
        report["elapsed_seconds"] = perf_counter() - started
        report_path.write_text(json.dumps(report, indent=2) + "\n")
        latest = report["history"][-1]
        print(
            f"Generation {latest['generation']}/{args.generations}: "
            f"best={latest['best_so_far']:.4f}, mean={latest['mean']:.4f}, "
            f"elapsed={report['elapsed_seconds']:.1f}s",
            flush=True,
        )

    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(
        f"Starting SelfCGP {args.population} x {args.generations}: {args.dataset}",
        flush=True,
    )
    try:
        with GeneticProgrammingRetriever(
            args.dataset,
            metric=NDCG(),
            n_iter=args.generations,
            pop_size=args.population,
            top_k=args.top_k,
            optimizer=SelfCGP,
            optimizer_args={
                "init_level": args.init_level,
                "max_level": args.max_level,
                "on_generation": save_progress,
            },
            random_state=args.seed,
            device=args.device,
        ) as retriever:
            retriever.fit()
            stats = retriever.get_stats()
            save_expressions(stats, args.output / "expressions.txt")
            report["documents"] = len(retriever.dataset_.corpus)
            report["queries"] = len(retriever.dataset_.queries)
            report["query_ids"] = list(retriever.dataset_.queries)
            report["history"] = summarize(stats)
            report["winner"] = {
                "genotype": str(retriever.genotype_),
                "phenotype": str(retriever.get_tree()),
                "fitness": retriever.fitness_,
            }
            report["fitness_evaluations"] = len(retriever._fitness_cache)
            report["worker_loads"] = retriever._pool.worker.loads
        report["status"] = "complete"
    except KeyboardInterrupt:
        report["status"] = "interrupted"
        raise
    except BaseException as error:
        report["status"] = "failed"
        report["error"] = repr(error)
        raise
    finally:
        report["elapsed_seconds"] = perf_counter() - started
        report_path.write_text(json.dumps(report, indent=2) + "\n")

    history = report["history"]
    figure, ax = plt.subplots(figsize=(9, 5))
    for key, label in (("best_so_far", "Best"), ("mean", "Mean"), ("median", "Median")):
        ax.plot(
            [row["generation"] for row in history],
            [row[key] for row in history],
            marker="o",
            label=label,
        )
    ax.set(
        xlabel="Generation (1 = initial population)",
        ylabel="NDCG",
        title=f"SciFact debug: {report['documents']} documents, {report['queries']} queries",
        ylim=(0, 1.02),
    )
    ax.grid(alpha=0.2)
    ax.legend()
    figure.tight_layout()
    figure.savefig(args.output / "fitness.png", dpi=150)
    plt.close(figure)
    print(
        f"Finished in {report['elapsed_seconds']:.1f}s; NDCG={report['winner']['fitness']:.4f}",
        flush=True,
    )
    print(f"Results: {report_path}", flush=True)


if __name__ == "__main__":
    main()
