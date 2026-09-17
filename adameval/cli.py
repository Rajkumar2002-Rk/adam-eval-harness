"""CLI. Exit codes: 0 pass, 1 below threshold, 2 usage error, 3 budget exceeded."""
from __future__ import annotations

import sys
from pathlib import Path

import click

from . import gates, inject, io, llm, report
from .harness import GROUND_TRUTH, SPECS, run_once
from .spec import LEVELS, load

EXIT_OK, EXIT_BELOW_THRESHOLD, EXIT_USAGE, EXIT_BUDGET = 0, 1, 2, 3


@click.group()
def main() -> None:
    """Eval harness and regression gate for LLM-generated SDTM -> ADaM derivations."""


@main.command()
def selftest() -> None:
    """Validate the gate layer against synthetic defects. Costs nothing."""
    bad = 0
    for ds in ("ADSL", "ADAE"):
        spec, truth = load(SPECS[ds]), io.read(GROUND_TRUTH[ds])
        if clean := gates.run_all(truth, truth, spec):
            click.echo(f"FAIL {ds}: {len(clean)} findings on ground truth")
            bad += 1
        for name, desc in inject.registry(ds).items():
            found = gates.run_all(inject.apply(ds, name, truth), truth, spec)
            tiers = sorted({f.layer for f in found})
            if not found:
                click.echo(f"  MISS {ds}/{name}: not caught")
                bad += 1
            else:
                tag = "SILENT" if tiers == ["TRUTH"] else "+".join(tiers)
                click.echo(f"  ok   {ds}/{name:30} {tag}")
    sys.exit(EXIT_OK if not bad else EXIT_BELOW_THRESHOLD)


@main.command()
@click.option("--dataset", type=click.Choice(["ADSL", "ADAE", "both"]), default="both")
@click.option("--level", type=click.Choice([*LEVELS, "all"]), default="all")
@click.option("--repeats", default=3, show_default=True)
@click.option("--max-attempts", default=4, show_default=True,
              help="Hard cap on repair retries per run.")
@click.option("--repair-from", default="STRUCTURAL", show_default=True,
              help="Comma-separated tiers whose findings may be shown to the model. "
                   "Adding TRUTH hands it the answer key - record it if you do.")
@click.option("--budget", default=10.0, show_default=True, help="Hard spend ceiling, USD.")
@click.option("--model", default=llm.MODEL, show_default=True)
@click.option("--fail-under", default=0.0, show_default=True,
              help="Exit 1 if mean score across runs falls below this.")
@click.option("--out", default="runs.json", show_default=True)
@click.option("--mock", is_flag=True, help="Use canned responses. Costs nothing.")
def run(dataset, level, repeats, max_attempts, repair_from, budget, model,
        fail_under, out, mock) -> None:
    """Run the derivation grid and gate the results."""
    datasets = ["ADSL", "ADAE"] if dataset == "both" else [dataset]
    levels = list(LEVELS) if level == "all" else [level]
    tiers = {t.strip().upper() for t in repair_from.split(",") if t.strip()}

    try:
        client = _mock_client() if mock else llm.AnthropicClient(model=model)
    except llm.NoCredentials as e:
        raise click.ClickException(str(e)) from e
    ledger = llm.Ledger(budget_usd=budget, model=model if not mock else "mock")

    runs: list[report.Run] = []
    try:
        for ds in datasets:
            for lv in levels:
                for r in range(repeats):
                    click.echo(f"-> {ds} {lv} repeat {r + 1}/{repeats} "
                               f"(spent ${ledger.spent:.2f})")
                    run_ = run_once(client, ledger, ds, lv, repeat=r,
                                    max_attempts=max_attempts, repair_from=tiers)
                    runs.append(run_)
                    click.echo(f"   score {run_.score:.2f} after "
                               f"{len(run_.attempts)} attempt(s)")
    except llm.BudgetExceeded as e:
        click.echo(f"\nBUDGET STOP: {e}", err=True)
        report.write(runs, out)
        sys.exit(EXIT_BUDGET)

    report.write(runs, out)
    mean = sum(r.score for r in runs) / len(runs) if runs else 0.0
    silent = sum(1 for r in runs for f in r.final.findings if f.silent)
    click.echo(f"\n{len(runs)} runs | mean score {mean:.3f} | "
               f"{silent} silent defects | ${ledger.spent:.2f} | -> {out}")
    if mean < fail_under:
        click.echo(f"FAIL: mean score {mean:.3f} < --fail-under {fail_under}", err=True)
        sys.exit(EXIT_BELOW_THRESHOLD)
    sys.exit(EXIT_OK)


def _mock_client() -> llm.Client:
    fixtures = sorted(Path("tests/fixtures").glob("*.py"))
    if not fixtures:
        raise click.ClickException("no mock fixtures in tests/fixtures/")
    bodies = [f"```python\n{p.read_text()}```" for p in fixtures]
    return llm.MockClient(bodies)


if __name__ == "__main__":
    main()
