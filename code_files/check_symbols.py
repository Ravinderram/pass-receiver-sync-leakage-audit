"""Static check for undefined names, run before trusting any script.

This exists because of a specific failure. step20_databallpy.py called
require_databallpy() and the function was not there: it had been removed by a
string-slice edit that replaced everything between two other function
definitions, and require_databallpy happened to sit between them. Nothing
caught it. py_compile passes, because the file is syntactically valid; the
version checker passes, because it looks for text markers and the marker it
looked for was in a different function.

pyflakes catches it in a second. Run this after editing anything.

Run:  python check_symbols.py
"""

import glob
import subprocess
import sys

# unused imports are a style matter and this project has several deliberate ones
IGNORE = ("imported but unused", "unable to detect undefined names")

if __name__ == "__main__":
    files = sorted(glob.glob("*.py") + glob.glob("verify/*.py")
                   + glob.glob("providers/*.py"))
    try:
        r = subprocess.run([sys.executable, "-m", "pyflakes", *files],
                           capture_output=True, text=True)
    except FileNotFoundError:
        sys.exit("pyflakes not installed:  pip install pyflakes")

    lines = [l for l in (r.stdout + r.stderr).splitlines()
             if l.strip() and not any(k in l for k in IGNORE)]

    fatal = [l for l in lines if "undefined name" in l]
    other = [l for l in lines if "undefined name" not in l]

    print(f"checked {len(files)} files\n")
    if fatal:
        print("UNDEFINED NAMES — these will raise at runtime:\n")
        for l in fatal:
            print("  " + l)
        print("\nA name can go missing without breaking syntax. That is what")
        print("happened to require_databallpy in step20.")
        sys.exit(1)
    if other:
        print("warnings:\n")
        for l in other:
            print("  " + l)
        sys.exit(0)
    print("No undefined names, no warnings.")
