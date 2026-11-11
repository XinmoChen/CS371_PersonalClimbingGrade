"""Build the two compact tables of the paper from the full tables that make_tables.py writes.

Table 5 (main_indoor_compact.tex) keeps seven columns of main_indoor.tex and shows the published
values in parentheses. The helped and hurt table of Section 5.4 (helped_indoor_compact.tex) keeps
the Protocol A columns and the Protocol B continuous columns of helped_indoor.tex. Every cell is
copied from the full tables, so the compact tables change only when the full ones do.

Usage: python3 make_compact_tables.py [tables_dir]   (default: TAB_DIR or ./tables, as in make_tables.py)
"""
import os
import re
import sys
from pathlib import Path

TABLES = Path(sys.argv[1] if len(sys.argv) > 1 else
              os.environ.get('TAB_DIR') or os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tables'))

PM = re.compile(r"\\,\{\\scriptsize\$\\pm\$[^}]*\}")
BOLD = re.compile(r"^\\textbf\{(.*)\}$")


def rows(path):
    """Return the data rows of a tabular as lists of stripped cells."""
    out = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line.endswith(r"\\") or line.startswith(("\\multicolumn", "&", "\\textbf{Model}", "\\textbf{Split}", "\\textbf{Personal")):
            continue
        out.append([c.strip() for c in line[:-2].split("&")])
    return out


def no_pm(cell):
    m = BOLD.match(cell)
    if m:
        return r"\textbf{" + PM.sub("", m.group(1)) + "}"
    return PM.sub("", cell)


def main_table():
    full = rows(TABLES / "main_indoor.tex")
    published = {}
    ours = []
    for r in full:
        if r[1] == "" and r[2]:          # published row: RMSE empty, RMSE_r given
            published[r[0]] = (r[2], r[3])
        else:
            ours.append(r)
    names = {"B1 SSE, linear reg.": "B1 SSE, linear regression"}
    # full columns: 0 model, 1 RMSE, 2 RMSE_r, 3 uRMSE, 4 RMSE_dis, 5 F1, 6 AP,
    #               7 B RMSE, 8 B RMSE_r, 9 B RMSE_dis, 10 B F1
    lines = [
        r"\begin{tabular}{l ccccc cc}",
        r"\toprule",
        r" & \multicolumn{5}{c}{\textbf{Protocol A (published split)}} & \multicolumn{2}{c}{\textbf{Protocol B (forward split)}} \\",
        r"\cmidrule(lr){2-6}\cmidrule(lr){7-8}",
        r"\textbf{Model} & RMSE & RMSE$_r$ & uRMSE & RMSE$_{\mathrm{dis}}$ & F1 & RMSE & RMSE$_{\mathrm{dis}}$ \\",
        r"\midrule",
    ]
    for r in ours:
        c = [no_pm(x) for x in r]
        rmse_r, urmse = c[2], c[3]
        if r[0] in published:
            p_r, p_u = published[r[0]]
            rmse_r += r"\,{\scriptsize(" + p_r + ")}"
            urmse += r"\,{\scriptsize(" + p_u + ")}"
        name = names.get(r[0], r[0])
        cells = [name, c[1], rmse_r, urmse, c[4], c[5], c[7], c[9]]
        if r[0] == "C Community and context":
            lines.append(r"\midrule")
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def helped_table():
    full = rows(TABLES / "helped_indoor.tex")
    a = {r[1]: r[2:] for r in full if r[0] == "A"}
    b = {r[1]: r[2:] for r in full if r[0] == "B"}
    lines = [
        r"\begin{tabular}{lcccccc}",
        r"\toprule",
        r" & \multicolumn{4}{c}{\textbf{Protocol A}} & \multicolumn{2}{c}{\textbf{Protocol B}} \\",
        r"\cmidrule(lr){2-5}\cmidrule(lr){6-7}",
        r" & \multicolumn{2}{c}{Continuous} & \multicolumn{2}{c}{Displayed} & \multicolumn{2}{c}{Continuous} \\",
        r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}",
        r"\textbf{Personal term in} & Helped & Hurt & Helped & Hurt & Helped & Hurt \\",
        r"\midrule",
    ]
    for name, va in a.items():
        lines.append(" & ".join([name] + va + b[name][:2]) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    for fname, text in [("main_indoor_compact.tex", main_table()),
                        ("helped_indoor_compact.tex", helped_table())]:
        (TABLES / fname).write_text(text)
        print("wrote", TABLES / fname)
