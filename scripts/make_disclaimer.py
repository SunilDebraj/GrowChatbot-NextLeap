"""Generate assets/disclaimer.txt from the PRD 10.5 block, verbatim.

Markdown presentation is stripped (the blockquote marker and bold delimiters)
because the asset is plain text rendered into HTML; the wording is not touched.
"""
import pathlib
import re

prd = pathlib.Path("PRD.md").read_text(encoding="utf-8")
m = re.search(r"### 10\.5[^\n]*\n(.*?)\n---", prd, re.S)
if not m:
    raise SystemExit("could not find PRD.md 10.5")

text = re.sub(r"^>\s*", "", m.group(1).strip()).replace("**", "")
pathlib.Path("assets/disclaimer.txt").write_text(text + "\n", encoding="utf-8")

norm = lambda s: re.sub(r"\s+", " ", s).strip()
written = pathlib.Path("assets/disclaimer.txt").read_text(encoding="utf-8")
print("verbatim:", norm(written) == norm(text))
print("chars:", len(written))
print(written)
