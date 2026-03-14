#!/usr/bin/env python3
"""
CT-BBKD GitHub Repository Setup Script
========================================
Initialises a local git repo and prints instructions
to push it to GitHub.

Usage: python scripts/github_setup.py
"""

import os
import subprocess
import sys

RESET  = '\033[0m'
GREEN  = '\033[92m'
BLUE   = '\033[94m'
YELLOW = '\033[93m'
CYAN   = '\033[96m'
BOLD   = '\033[1m'

def run(cmd, cwd=None, check=True):
    result = subprocess.run(
        cmd, shell=True, cwd=cwd,
        capture_output=True, text=True
    )
    if check and result.returncode != 0:
        print(f"  ❌ Command failed: {cmd}")
        print(f"     {result.stderr.strip()}")
        return False
    return result.stdout.strip()

def banner():
    print(f"""
{BLUE}╔══════════════════════════════════════════════════════════╗
║      CT-BBKD GitHub Repository Setup                    ║
╚══════════════════════════════════════════════════════════╝{RESET}
""")

def setup_git():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(root)

    banner()
    print(f"  {BOLD}Project root:{RESET} {root}\n")

    # Check git installed
    git_ver = run("git --version", check=False)
    if not git_ver:
        print(f"  {YELLOW}⚠ Git not found. Install git first.{RESET}")
        sys.exit(1)
    print(f"  ✓ {git_ver}")

    # Init repo
    if not os.path.exists('.git'):
        run("git init")
        print(f"  ✓ Git repository initialized")
    else:
        print(f"  ✓ Git repository already exists")

    # Create .gitignore additions
    gi_content = open('.gitignore').read() if os.path.exists('.gitignore') else ''
    if '*.db' not in gi_content:
        with open('.gitignore', 'a') as f:
            f.write('\n*.db\n*.sqlite\n__pycache__/\n')
        print(f"  ✓ .gitignore updated")

    # Stage all files
    run("git add .")
    print(f"  ✓ Files staged")

    # Initial commit
    status = run("git status --short")
    if status:
        run('git commit -m "feat: initial CT-BBKD system commit" --allow-empty')
        print(f"  ✓ Initial commit created")
    else:
        print(f"  ✓ Nothing to commit (already up to date)")

    # Summary of files
    files = run("git ls-files").split('\n') if run("git ls-files") else []
    print(f"\n  {BOLD}Repository Contents ({len(files)} files):{RESET}")
    for f in sorted(files):
        print(f"    {CYAN}{f}{RESET}")

    # GitHub instructions
    print(f"""
{GREEN}╔══════════════════════════════════════════════════════════╗
║          Next Steps — Push to GitHub                    ║
╚══════════════════════════════════════════════════════════╝{RESET}

  {BOLD}1. Create a new GitHub repo:{RESET}
     Go to https://github.com/new
     Name: ct-bbkd
     ✓ Keep it public (for paper reviewers to access)
     ✓ Do NOT initialise with README (we have one)

  {BOLD}2. Add remote and push:{RESET}
     {CYAN}git remote add origin https://github.com/YOUR-USERNAME/ct-bbkd.git
     git branch -M main
     git push -u origin main{RESET}

  {BOLD}3. Add repository topics:{RESET}
     knowledge-distillation, continual-learning, black-box,
     machine-learning, pytorch, flask, real-time-dashboard

  {BOLD}4. Enable GitHub Actions:{RESET}
     Go to Actions tab → "I understand my workflows" → Enable
     CI will run on every push automatically.

  {BOLD}5. Add to paper:{RESET}
     "Code available at: https://github.com/YOUR-USERNAME/ct-bbkd"

  {BOLD}Structure pushed:{RESET}
     backend/app.py           ← Flask REST API (FastAPI-compatible)
     backend/core/distillation.py ← SDD + EWC-KD + DAR + AAR
     frontend/dashboard.html  ← Real-time monitoring dashboard
     tests/test_api.py        ← 12 passing tests
     scripts/demo.py          ← CLI demo runner
     scripts/run.sh           ← One-command startup
     .github/workflows/ci.yml ← GitHub Actions CI
     README.md                ← Full documentation
     requirements.txt
""")

if __name__ == '__main__':
    setup_git()
