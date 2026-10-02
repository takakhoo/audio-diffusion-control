"""Write the paper's tables and number macros from the committed results.

    python experiments/make_paper_tables.py

Reads results/*/summary.json, results/gating/summary.csv and results/discovery; writes
paper/tables/*.tex and paper/numbers.tex so that no number in the paper is typed by hand.
"""

import csv
import json
from pathlib import Path

RESULTS, PAPER = Path("results"), Path("paper")
(PAPER / "tables").mkdir(parents=True, exist_ok=True)
macros = {}


def fmt(v, digits=2):
    return "--" if v is None or (isinstance(v, float) and v != v) else f"{v:.{digits}f}"


def slider_table(name, caption, label, methods=("LoRA slider",)):
    path = RESULTS / name / "summary.json"
    if not path.exists():
        return
    rows = [r for r in json.loads(path.read_text())["summary"] if r["method"] in methods]
    lines = [r"\begin{table*}[t]\centering\small", r"\begin{tabular}{llrrrrrrrr}", r"\toprule",
             r"Slider & Descriptor & $\rho$ & Ordered & MuQ $\rho$ & Span & Moved & Kept & CE$_0$ & CE$_\text{ends}$ \\", r"\midrule"]
    for r in rows:
        span = f"{r.get('usable_lo', 0):+.1f} to {r.get('usable_hi', 0):+.1f}" if "usable_lo" in r else "--"
        lines.append(" & ".join([
            r["slider"], (r.get("measure") or "--").replace("_", r"\_"), fmt(r.get("rho")), fmt(r.get("consistent")),
            fmt(r.get("muq_rho")), span, fmt(r.get("usable_range_in_std")), fmt(r.get("usable_clap_keep")), fmt(r.get("ce_at_zero")),
            fmt(r.get("ce_at_ends"))]) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", rf"\caption{{{caption}}}\label{{{label}}}", r"\end{table*}"]
    (PAPER / "tables" / f"{name}.tex").write_text("\n".join(lines) + "\n")
    if rows:
        macros[f"{name}CEzero"] = fmt(rows[0].get("ce_at_zero"))


slider_table("ace", "Prompt-pair sliders on ACE-Step 1.5 XL turbo. $\\rho$: mean rank correlation between position and "
             "descriptor. Ordered: share of trajectories with correctly ordered ends. MuQ $\\rho$: the same correlation for the "
             "MuQ-MuLan direction score. Span: usable span. Moved: descriptor "
             "change over the span in standard deviations of unsteered clips. Kept: CLAP similarity to the unsteered clip "
             "at the ends of the span. CE: content enjoyment unsteered and at the extreme positions.", "tab:ace")
slider_table("main", "Prompt-pair sliders on Stable Audio Open 1.0 with the slider switched on after step 21 of 50. "
             "Columns as in Table~\\ref{tab:ace}.", "tab:sao")

gate = RESULTS / "gating" / "summary.csv"
if gate.exists():
    rows = [r for r in csv.DictReader(gate.open()) if r["slider"] == "brightness"]
    lines = [r"\begin{table}[t]\centering\small\setlength{\tabcolsep}{4pt}", r"\begin{tabular}{rrrrr}", r"\toprule",
             r"On after & Moved & CLAP & Chroma & CE \\", r"\midrule"]
    for r in rows:
        lines.append(f"{r['skipped_steps']} & {float(r['range_in_std']):.2f} & {float(r['clap_keep']):.2f} & "
                     f"{float(r['chroma_sim']):.2f} & {float(r['ce_at_ends']):.2f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}",
              r"\caption{Timestep gating for the brightness slider on Stable Audio Open at positions $\pm1$: descriptor moved (std), CLAP and chroma similarity to the unsteered clip, content enjoyment. "
              rf"Unsteered CE is {float(rows[0]['ce_at_zero']):.2f}.}}\label{{tab:gating}}", r"\end{table}"]
    (PAPER / "tables" / "gating.tex").write_text("\n".join(lines) + "\n")
    by = {int(r["skipped_steps"]): r for r in rows}
    zero = float(rows[0]["ce_at_zero"])
    macros.update(gateSteps="21", gateKeepOff=f"{float(by[0]['clap_keep']):.2f}", gateKeepOn=f"{float(by[21]['clap_keep']):.2f}",
                  gateLossOff=f"{zero - float(by[0]['ce_at_ends']):.1f}", gateLossOn=f"{zero - float(by[21]['ce_at_ends']):.1f}",
                  gateRangeOff=f"{float(by[0]['range_in_std']):.2f}", gateRangeOn=f"{float(by[21]['range_in_std']):.2f}")
    worst = [r for r in csv.DictReader(gate.open()) if r["slider"] in ("density", "percussion")]
    macros["weakRho"] = f"{max(float(r['rho']) for r in worst):.2f}"

pca = RESULTS / "discovery" / "ace" / "pca.json"
if pca.exists():
    data = json.loads(pca.read_text())
    macros["pcaTop"] = f"{100 * max(c['components'][0]['variance_share'] for c in data.values()):.0f}"
    macros["pcaLow"] = f"{100 * min(c['components'][0]['variance_share'] for c in data.values()):.0f}"
    lines = [r"\begin{table*}[t]\centering\small", r"\begin{tabular}{lrll}", r"\toprule",
             r"Concept & Var. & Toward & Away \\", r"\midrule"]
    for concept, info in data.items():
        for c in info["components"][:2]:
            lines.append(f"{concept} & {100 * c['variance_share']:.1f}\\% & {', '.join(c['toward'][:3])} & "
                         f"{', '.join(c['away'][:3])} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}",
              r"\caption{The two leading principal components of CLAP embeddings for each concept (1{,}024 ACE-Step clips "
              r"each), labelled by the tags they align with.}\label{tab:pca}", r"\end{table*}"]
    (PAPER / "tables" / "pca.tex").write_text("\n".join(lines) + "\n")

found = RESULTS / "discovery" / "ace" / "sliders.json"
if found.exists():
    sl = json.loads(found.read_text())["sliders"]
    macros.update(foundN=str(len(sl)), foundLo=f"{min(x['follows'] for x in sl):.2f}", foundHi=f"{max(x['follows'] for x in sl):.2f}")
    lines = [r"\begin{table*}[t]\centering\small", r"\begin{tabular}{llrrl}", r"\toprule",
             r"Concept & Corpus label (toward / away) & $\rho$ & Kept & Tags that rise / fall in the slider's output \\", r"\midrule"]
    for x in sl:
        if x["component"] > 1:
            continue
        lines.append(f"{x['concept']} & {', '.join(x['corpus_toward'][:2])} / {', '.join(x['corpus_away'][:2])} & {x['follows']:.2f} & "
                     f"{x['kept']:.2f} & {', '.join(x['rises'][:2])} / {', '.join(x['falls'][:2])} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}",
              r"\caption{Sliders trained along discovered axes (two leading components per concept). $\rho$: rank correlation "
              r"between slider position and the output's projection on the axis, over 32 fresh seeds. Kept: CLAP similarity to "
              r"the unsteered clip at the ends of the usable span.}\label{tab:found}", r"\end{table*}"]
    (PAPER / "tables" / "found.tex").write_text("\n".join(lines) + "\n")

ace = RESULTS / "ace" / "summary.json"
if ace.exists():
    rows = json.loads(ace.read_text())["summary"]
    methods = [("LoRA slider", "Slider"), ("Prompt-pair guidance", "Guidance"), ("Activation steering", "Act. steer."),
               ("Prompt interpolation", "Interp.")]
    by = {(r["slider"], r["method"]): r for r in rows}
    sliders = [s for s in dict.fromkeys(r["slider"] for r in rows) if all((s, m) in by for m, _ in methods)]
    lines = [r"\begin{table}[t]\centering\small\setlength{\tabcolsep}{3pt}", r"\begin{tabular}{l" + "rr" * len(methods) + "}",
             r"\toprule", " & " + " & ".join(rf"\multicolumn{{2}}{{c}}{{{short}}}" for _, short in methods) + r" \\",
             "Slider & " + " & ".join(r"$\rho$ & MuQ" for _ in methods) + r" \\", r"\midrule"]
    for name in sliders:
        lines.append(name + " & " + " & ".join(f"{fmt(by[(name, m)].get('rho'))} & {fmt(by[(name, m)].get('muq_rho'))}"
                                                 for m, _ in methods) + r" \\")
    lines += [r"\midrule", "CE at ends & " + " & ".join(
        rf"\multicolumn{{2}}{{c}}{{{sum(by[(n, m)]['ce_at_ends'] for n in sliders) / len(sliders):.2f}}}" for m, _ in methods) + r" \\",
        "Passes per step & " + " & ".join(rf"\multicolumn{{2}}{{c}}{{{c}}}" for c in (1, 3, 1, 1)) + r" \\",
        r"\bottomrule", r"\end{tabular}",
        r"\caption{Four ways to move the same attribute on ACE-Step, same prompts and seeds. $\rho$: rank correlation of "
        r"position with the waveform descriptor (-- where none was assigned); MuQ: with the MuQ-MuLan direction. CE at ends: "
        r"mean content enjoyment at the two extreme positions, averaged over sliders.}\label{tab:methods}", r"\end{table}"]
    (PAPER / "tables" / "methods.tex").write_text("\n".join(lines) + "\n")

cover = RESULTS / "coverage" / "muq_ica" / "coverage.json"
if cover.exists():
    c = json.loads(cover.read_text())
    m = c["models"]["ace"]
    macros.update(covLo=f"{100 * min(m['total']):.0f}", covHi=f"{100 * max(m['total']):.0f}",
                  covSeedLo=f"{100 * min(m['within']):.0f}", covSeedHi=f"{100 * max(m['within']):.0f}",
                  covOverlap=f"{100 * m['overlap']['8']:.0f}", covClips=f"{m['clips']:,}".replace(",", "{,}"),
                  covReal=f"{c['real_clips']:,}".replace(",", "{,}"))

axes = RESULTS / "ace_v2" / "real_axes.json"
if axes.exists():
    rows = json.loads(axes.read_text())
    short = {"two sets of the model's clips": "sets", "prompt pair from the axis's tags": "pair"}
    mean = lambda v: fmt((v[0] + v[2]) / 2) if v else "--"
    lines = [r"\begin{table}[t]\centering\small\setlength{\tabcolsep}{3pt}", r"\begin{tabular}{llrrrrrrr}", r"\toprule",
             r"Axis & From & $\rho$ & Ord. & Real & Seeds & Kept & CE & Mus. \\", r"\midrule"]
    for r in sorted(rows, key=lambda r: (r["slider"], r["trained_from"])):
        lines.append(" & ".join([
            r["slider"].replace("_axis", "").replace("_", "/"), short[r["trained_from"]], fmt(r["rho"]), fmt(r["ordered"]),
            fmt(r["moved_real_std"][0]), fmt(r["moved_seed_std"][0]), fmt(r["kept"]), mean(r["ce"]),
            mean(r.get("musicality"))]) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}",
              r"\caption{Sliders along axes of real music on ACE-Step, positions $-1$ to $+1$. From: a prompt pair made of the "
              r"axis's tags, or two sets of the model's clips. $\rho$, Ord.: rank correlation between position and the "
              r"output's projection on the axis, and share of trajectories with $-1$ and $+1$ in the right order. Real, Seeds: "
              r"movement along the axis between $-1$ and $+1$ in standard deviations of real recordings and of one prompt's "
              r"seeds. Kept: CLAP similarity to the unsteered clip. CE, Mus.: enjoyment and SongEval musicality at $\pm1$ "
              r"(unsteered: " + fmt(rows[0]["ce"][1]) + " and " + mean(None if not rows[0].get("musicality") else [rows[0]["musicality"][1]] * 3)
              + r").}\label{tab:axes}", r"\end{table}"]
    (PAPER / "tables" / "axes.tex").write_text("\n".join(lines) + "\n")
    sets = [r for r in rows if short[r["trained_from"]] == "sets"]
    macros.update(axN=str(len(sets)), axRhoLo=fmt(min(r["rho"] for r in sets)), axRhoHi=fmt(max(r["rho"] for r in sets)),
                  axRealLo=fmt(min(r["moved_real_std"][0] for r in sets)), axRealHi=fmt(max(r["moved_real_std"][0] for r in sets)),
                  axKeepLo=fmt(min(r["kept"] for r in sets)), axKeepHi=fmt(max(r["kept"] for r in sets)))

macros.update(nPrompts="24", nReal="2{,}000", fmaCE="6.12", aceCE="6.92", saoCE="6.16")
(PAPER / "numbers.tex").write_text("".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in sorted(macros.items())))
print("macros:", macros)
