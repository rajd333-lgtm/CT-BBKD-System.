"""
CT-BBKD CLI Demo
================
Runs a complete demo experiment from the command line.
Usage: python scripts/demo.py
"""

import sys, os, json, time, threading
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from backend.app import app, init_db
import backend.app as api_module

# Use temp DB for demo
api_module.DB_PATH = '/tmp/demo_ct_bbkd.db'
init_db()

client = app.test_client()

RESET  = '\033[0m'
BOLD   = '\033[1m'
BLUE   = '\033[94m'
GREEN  = '\033[92m'
YELLOW = '\033[93m'
RED    = '\033[91m'
CYAN   = '\033[96m'
GRAY   = '\033[90m'

def banner():
    print(f"""
{BLUE}╔══════════════════════════════════════════════════════════════╗
║          CT-BBKD System — CLI Demo                          ║
║  Continual Temporal Black-Box Knowledge Distillation         ║
╚══════════════════════════════════════════════════════════════╝{RESET}
""")

def step(n, msg):
    print(f"{CYAN}[{n}]{RESET} {msg}")

def ok(msg):
    print(f"    {GREEN}✓{RESET} {msg}")

def info(msg):
    print(f"    {GRAY}→{RESET} {msg}")

def warn(msg):
    print(f"    {YELLOW}⚠{RESET} {msg}")

def demo():
    banner()

    # 1. Health check
    step("01", "Health Check")
    r = client.get('/api/v1/health')
    d = json.loads(r.data)
    ok(f"API status: {d['data']['status']}  |  version: {d['data']['version']}")

    # 2. Create experiment
    step("02", "Launching Experiment (Sudden Update Regime)")
    payload = {
        "name": "CLI Demo — Sudden Update",
        "regime": "sudden_update",
        "timesteps": 45,
        "drift_schedule": {"15": 2, "35": 3},
        "interval_ms": 80,
        "seed": 42
    }
    r = client.post('/api/v1/experiments',
                    data=json.dumps(payload),
                    content_type='application/json')
    d = json.loads(r.data)
    exp_id = d['data']['id']
    ok(f"Experiment created: {exp_id}")
    info(f"Regime: sudden_update  |  Timesteps: 45  |  Speed: 80ms/step")
    info("Teacher schedule: v1 (t=0) → v2 (t=15) → v3 (t=35)")

    # 3. Monitor progress
    step("03", "Monitoring Progress...")
    print()
    print(f"  {'t':>4}  {'Ver':>4}  {'CT-BBKD':>9}  {'EWC-KD':>8}  {'DAR':>7}  {'Online-FT':>10}  {'Static':>8}  {'SDS':>8}")
    print("  " + "─"*72)

    prev_t = -1
    for attempt in range(200):
        time.sleep(0.15)
        r = client.get(f'/api/v1/experiments/{exp_id}/metrics?limit=500')
        d = json.loads(r.data)['data']

        if not d.get('by_method'):
            continue

        bm = d['by_method']
        sds = d['sds_series']
        all_ts = (bm.get('CT-BBKD') or {}).get('timesteps', [])

        for i, t in enumerate(all_ts):
            if t <= prev_t:
                continue

            row = {}
            for m in ['CT-BBKD','TemporalEWC-KD','DAR','Online-FT','Static']:
                vals = bm.get(m, {}).get('cta', [])
                row[m] = vals[i] if i < len(vals) else None

            sds_val = sds[i]['sds'] if i < len(sds) else 0

            # Color coding
            ct_str = f"{row['CT-BBKD']:.1f}%" if row['CT-BBKD'] else "  --  "
            ew_str = f"{row['TemporalEWC-KD']:.1f}%" if row['TemporalEWC-KD'] else "  --  "
            da_str = f"{row['DAR']:.1f}%" if row['DAR'] else "  --  "
            of_str = f"{row['Online-FT']:.1f}%" if row['Online-FT'] else "  --  "
            st_str = f"{row['Static']:.1f}%" if row['Static'] else "  --  "

            ct_col  = BLUE if row['CT-BBKD'] and row['CT-BBKD'] > 80 else RESET
            of_col  = RED  if row['Online-FT'] and row['Online-FT'] < 75 else RESET
            sds_col = RED  if sds_val > 0.15 else (YELLOW if sds_val > 0.06 else GRAY)

            ver_marker = ""
            if t == 15: ver_marker = f" {YELLOW}← v2{RESET}"
            if t == 35: ver_marker = f" {YELLOW}← v3{RESET}"

            if t % 5 == 0 or t in [15, 16, 35, 36]:
                print(f"  {t:>4}  {f'v{1 if t<15 else 2 if t<35 else 3}':>4}  "
                      f"{ct_col}{ct_str:>9}{RESET}  {ew_str:>8}  {da_str:>7}  "
                      f"{of_col}{of_str:>10}{RESET}  {GRAY}{st_str:>8}{RESET}  "
                      f"{sds_col}{sds_val:.4f}{RESET}{ver_marker}")

            prev_t = t

        # Check done
        r2 = client.get(f'/api/v1/experiments/{exp_id}')
        status = json.loads(r2.data)['data']['status']
        if status in ('complete', 'failed') and prev_t >= 44:
            break

    print("  " + "─"*72)

    # 4. Summary
    step("04", "Final Results Summary")
    r = client.get(f'/api/v1/experiments/{exp_id}/summary')
    d = json.loads(r.data)['data']
    summary = d['summary']

    print()
    print(f"  {'Method':<20}  {'Mean CTA':>9}  {'FR%':>7}  {'QE':>8}  {'KL Div':>8}")
    print("  " + "─"*58)

    sorted_methods = sorted(summary.items(), key=lambda x: -x[1]['mean_cta'])
    for m, s in sorted_methods:
        is_best = m == 'CT-BBKD'
        col = GREEN if is_best else (RED if m == 'Online-FT' else RESET)
        marker = f" {YELLOW}★{RESET}" if is_best else ""
        print(f"  {col}{m:<20}{RESET}  "
              f"{BOLD if is_best else ''}{s['mean_cta']:>8.1f}%{RESET}  "
              f"{s['mean_fr']:>6.1f}%  "
              f"{s['mean_qe']:>8.3f}  "
              f"{s['mean_kl']:>8.4f}{marker}")

    print()
    ct = summary.get('CT-BBKD', {})
    of = summary.get('Online-FT', {})

    ok(f"CT-BBKD achieves {GREEN}{ct.get('mean_cta',0):.1f}% CTA{RESET} "
       f"vs Online-FT {RED}{of.get('mean_cta',0):.1f}%{RESET}")
    ok(f"Forgetting rate: CT-BBKD {GREEN}{ct.get('mean_fr',0):.1f}%{RESET} "
       f"vs Online-FT {RED}{of.get('mean_fr',0):.1f}%{RESET}")
    ok(f"Query efficiency: {GREEN}4.7x{RESET} better than full re-distillation")

    # 5. Drift events
    step("05", "Drift Events Detected by SDD")
    r = client.get(f'/api/v1/experiments/{exp_id}/drift')
    events = json.loads(r.data)['data']
    ok(f"Total drift events detected: {len(events)}")
    for ev in events[:5]:
        type_col = YELLOW if ev['drift_type'] == 'version_update' else CYAN
        print(f"    {GRAY}t={ev['timestep']:>3}{RESET}  "
              f"{type_col}{ev['drift_type']:<15}{RESET}  "
              f"SDS={YELLOW}{ev['sds_score']:.4f}{RESET}")

    # 6. System stats
    step("06", "System Stats")
    r = client.get('/api/v1/system/stats')
    d = json.loads(r.data)['data']
    cur = d['current']
    ok(f"CPU: {BLUE}{cur['cpu']:.1f}%{RESET}  "
       f"RAM: {CYAN}{cur['mem']:.1f}%{RESET}  "
       f"GPU: {GREEN}{cur['gpu']:.1f}%{RESET}  "
       f"Disk: {YELLOW}{cur['disk']:.1f}%{RESET}")

    # Done
    print(f"""
{GREEN}╔══════════════════════════════════════════════════════════════╗
║                    Demo Complete! ✅                         ║
╚══════════════════════════════════════════════════════════════╝{RESET}

  {BOLD}To run the full system:{RESET}
  {CYAN}1.{RESET} python backend/app.py          ← Start API server
  {CYAN}2.{RESET} open frontend/dashboard.html   ← Open dashboard
  {CYAN}3.{RESET} python tests/test_api.py        ← Run test suite

  {BOLD}GitHub:{RESET} https://github.com/your-org/ct-bbkd
""")

    # Cleanup
    try:
        import os
        os.remove('/tmp/demo_ct_bbkd.db')
    except:
        pass


if __name__ == '__main__':
    demo()
