"""Step 29 - The same probes on a second provider.

The leakage finding is currently demonstrated on one dataset and one family of
synchronisers. If the same one-line probes fire on a provider whose events are
frame-tagged differently, the finding is about how event and tracking data are
joined in general; if they stay quiet, it is about this pipeline.

This script runs the probes in probes.py through the provider adapters in
providers/. The IDSSE adapter is the reference. The PFF FC World Cup 2022 adapter
is implemented but its data are NOT in this environment: when the files are
absent the run records "dataset not available" and produces no numbers for that
provider. No cross-provider claim may be made from a run in that state.

To enable PFF:
    export PFF_WC2022_DIR=/path/to/pff_wc2022     # layout: providers/pff.py
    python step29_cross_provider.py

Run:  python step29_cross_provider.py [--providers idsse,pff] [--max-matches N]
Out:  work/cross_provider_probes.csv, work/cross_provider_status.json
"""

import sys

import pandas as pd

import config as C
import probes as PR
import providers
import provenance as P
from common import banner

PROBE_NAMES = ["nearest_teammate", "ball_direction", "passer_direction",
               "masella_min_abs", "masella_max_abs"]


def run_provider(name, leads, max_matches=None):
    ad = providers.get_adapter(name)
    status = {"provider": name, "description": ad.describe(),
              "expected_input": ad.expected_input(), "available": ad.available()}
    if not ad.available():
        status["status"] = "dataset not available - not run"
        return [], status
    rows = []
    try:
        ms = ad.matches()[:max_matches] if max_matches else ad.matches()
        for m in ms:
            for lead in leads:
                s = ad.snapshot(m, lead)
                base = PR.run_probe("nearest_teammate", s)
                for probe in PROBE_NAMES:
                    r = PR.run_probe(probe, s)
                    rows.append({"provider": name, "match": m, "lead_s": lead,
                                 "probe": probe, "n": r["n"],
                                 "n_defined": r["n_defined"],
                                 "accuracy": r["acc"],
                                 "accuracy_where_defined": r["acc_defined"],
                                 "nearest_teammate": base["acc"],
                                 "excess": r["acc"] - base["acc"]})
        status["status"] = "run"
        status["matches"] = ms
    except providers.DatasetUnavailable as exc:
        status["status"] = f"dataset not available - not run ({exc})"
    return rows, status


if __name__ == "__main__":
    names = ["idsse", "pff"]
    max_matches = None
    for a in sys.argv[1:]:
        if a.startswith("--providers"):
            names = a.split("=", 1)[1].split(",") if "=" in a else names
        if a.startswith("--max-matches="):
            max_matches = int(a.split("=", 1)[1])
    banner("Step 29 - the same probes on a second provider")
    run_id = P.canonical_run_id()
    leads = C.LEAD_SWEEP_LEADS
    P.print_header("cross-provider probes", run_id, providers=names,
                   offsets_s=leads, probes=PROBE_NAMES)

    rows, statuses = [], []
    for name in names:
        r, st = run_provider(name, leads, max_matches)
        rows += r
        statuses.append(st)
        print(f"\n{name}: {st['status']}")
        if st["status"] != "run":
            print("  expected input:")
            for line in st["expected_input"]:
                print("    " + line)

    df = pd.DataFrame(rows)
    if len(df):
        P.stamp(df, run_id, model_variant="cross_provider").to_csv(
            f"{C.WORK}/cross_provider_probes.csv", index=False)
        print("\nmean probe accuracy by provider and offset:")
        print(df.pivot_table(index=["provider", "lead_s"], columns="probe",
                             values="accuracy").round(3).to_string())
    P.write_json(f"{C.WORK}/cross_provider_status.json",
                 P.base_metadata("cross_provider", run_id, providers=statuses,
                                 offsets_s=leads, probes=PROBE_NAMES))

    ran = [s["provider"] for s in statuses if s["status"] == "run"]
    print(f"\nproviders actually run: {ran or 'none'}")
    if len(ran) < 2:
        print("A cross-provider claim needs at least two providers actually run.")
        print("With only one, this run supports no such claim, and none is made.")
    print(f"\nSaved -> {C.WORK}/cross_provider_status.json")
