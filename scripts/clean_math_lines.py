import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "energy_data.jsonl"
# keep previous backup as-is and write an additional backup before a second pass
BACKUP = ROOT / "energy_data.jsonl.bak"
BACKUP2 = ROOT / "energy_data.jsonl.bak2"

# Patterns indicating math or geometry formulas. Conservative set.
PATTERNS = [
    r"\$",                # LaTeX math delimiters
    r"\\hat",           # \hat{...}
    r"\\frac",          # \frac{...}{...}
    r"\\begin",         # \begin{...}
    r"\\pi",            # \pi
    r"\\sum",           # \sum
    r"\\sqrt",          # \sqrt
    r"\\\(|\\\)",   # escaped parentheses from LaTeX
    r"\\\[|\\\]",   # escaped brackets
    r"\bpi\b",          # pi
    r"\bπ\b",           # pi symbol
    r"\bsin\s*\(|\bcos\s*\(|\btan\s*\(",
    r"\barea\b|\bperimeter\b|\bradius\b|\bdiameter\b",
    r"\b[mM]\^?2\b|\bcm\^?2\b|\bcm2\b|\bm2\b",
    r"\b\\frac|\b\\math", # additional LaTeX math
]

pattern = re.compile("(?:" + ")|(?:".join(PATTERNS) + ")", re.IGNORECASE)

if not TARGET.exists():
    print(f"Target file not found: {TARGET}")
    raise SystemExit(1)

# Read all lines

with TARGET.open('r', encoding='utf-8') as f:
    lines = f.readlines()

# Backup before cleaning (preserve original and previous backup)
with BACKUP2.open('w', encoding='utf-8') as f:
    f.writelines(lines)

# Extend patterns to also catch heavy escaping and quantum-related phrases
PATTERNS_EXT = [
    r"\\\\+",                    # many backslashes (escaped LaTeX fragments)
    r"\bquantum\b",               # quantum-related text
    r"\bdensity operator\b",
    r"\bmaster equation\b",
    r"\bHamiltonian\b",
    r"\bLiouville\b",
    r"\bspin\b",
    r"\bsinglet\b|\btriplet\b",
    r"hat\\{",                     # hat{ pattern
]

# compile combined pattern (keep original conservative matches + extensions)
pattern = re.compile("(?:" + ")|(?:".join(PATTERNS + PATTERNS_EXT) + ")", re.IGNORECASE)

keep = []
removed_count = 0
for i, line in enumerate(lines, start=1):
    if pattern.search(line):
        removed_count += 1
    else:
        keep.append(line)

with TARGET.open('w', encoding='utf-8') as f:
    f.writelines(keep)

print(f"Total lines: {len(lines)}")
print(f"Removed lines: {removed_count}")
print(f"Kept lines: {len(keep)}")
print(f"Backup written to: {BACKUP2}")
