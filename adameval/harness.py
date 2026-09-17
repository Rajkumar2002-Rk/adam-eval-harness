"""The run loop: prompt -> execute -> gate -> targeted repair -> record."""
from __future__ import annotations

from pathlib import Path

from . import gates, io, llm, prompt as P, runner
from .report import Attempt, Run
from .spec import Level, load

GROUND_TRUTH = {"ADSL": "data/adam/adsl.xpt", "ADAE": "data/adam/adae.xpt"}
SPECS = {"ADSL": "specs/adsl.yaml", "ADAE": "specs/adae.yaml"}


def run_once(client: llm.Client, ledger: llm.Ledger, dataset: str, level: Level,
             repeat: int = 0, max_attempts: int = 4,
             repair_from: set[str] | None = None, cwd: str | Path = ".") -> Run:
    repair_from = repair_from or {"STRUCTURAL"}
    spec = load(Path(cwd) / SPECS[dataset])
    truth = io.read(Path(cwd) / GROUND_TRUTH[dataset])

    run = Run(dataset=dataset, level=level, repeat=repeat,
              model=getattr(client, "model", "mock"), target_variables=spec.names)

    system = P.system_blocks()
    messages = [P.initial_message(spec, level)]

    for n in range(1, max_attempts + 1):
        text, usage = client.complete(system, messages)
        ledger.charge(usage)
        attempt = Attempt(n=n, usage=usage)

        code = llm.extract_code(text)
        if not code:
            attempt.findings = [gates.Finding("NO_CODE", "EXECUTION", None,
                                              "response contained no python block")]
            run.attempts.append(attempt)
            messages += [{"role": "assistant", "content": text},
                         P.repair_message([{"code": "NO_CODE", "variable": None,
                                            "problem": "no python code block found"}], n)]
            continue

        attempt.code = code
        ex = runner.execute(code, dataset, cwd=cwd)
        attempt.executed = ex.ok

        if not ex.ok:
            attempt.findings = ex.findings
        else:
            run.provenance = ex.provenance
            attempt.findings = gates.run_all(ex.dataset, truth, spec, ex.provenance)

        run.attempts.append(attempt)
        if not attempt.findings:
            break
        if n == max_attempts:
            break

        feedback = runner.redact_for_repair(attempt.findings, repair_from)
        if not feedback:
            # Nothing we are permitted to tell the model. Stopping here is the
            # honest outcome: a production pipeline would have nothing to say
            # either, which is exactly what makes these defects silent.
            break
        messages += [{"role": "assistant", "content": text},
                     P.repair_message(feedback, n)]

    return run
