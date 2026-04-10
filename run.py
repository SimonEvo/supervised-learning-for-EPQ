"""
BDT Demo 统一入口
修改下方 DEMO_CHOICE，然后运行：  python run.py
"""

import subprocess
import sys
import os

# ══════════════════════════════════════════════════════════════════════
DEMO_CHOICE = "hzz"   # "classic" / "improved" / "hzz"
# ══════════════════════════════════════════════════════════════════════

demos = {
    "classic":  os.path.join("demo_classic",  "main.py"),
    "improved": os.path.join("demo_improved", "main.py"),
    "hzz":      os.path.join("demo_hzz",      "main.py"),
}

if DEMO_CHOICE not in demos:
    print(f"错误：DEMO_CHOICE 应为 {list(demos.keys())} 之一，当前为 {DEMO_CHOICE!r}")
    sys.exit(1)

script = os.path.join(os.path.dirname(os.path.abspath(__file__)), demos[DEMO_CHOICE])
subprocess.run([sys.executable, script], check=True)
