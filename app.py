"""
CT-BBKD Backend API Server
==========================
REST API for the Continual Temporal Black-Box Knowledge Distillation system.
Provides endpoints for distillation experiments, monitoring, and metrics.
"""

import os
import sys
import json
import time
import math
import random
import sqlite3
import threading
import datetime
import psutil
from pathlib import Path
from flask import Flask, jsonify, request, Response, stream_with_context
from functools import wraps

# ── App setup ──────────────────────────────────────────────────────────────
app = Flask(__name__)
app.config['JSON_SORT_KEYS'] = False

BASE_DIR = Path(__file__).parent.parent
DB_PATH  = BASE_DIR / "ct_bbkd.db"

# ── Thread-safe state ──────────────────────────────────────────────────────
_state_lock = threading.Lock()
_active_experiments = {}   # exp_id → experiment dict
_metrics_cache = {}        # exp_id → list of metric snapshots

# ══════════════════════════════════════════════════════════════════════════
#  DATABASE
# ══════════════════════════════════════════════════════════════════════════

def get_db():
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    with get_db() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS experiments (
                id          TEXT PRIMARY KEY,
                name        TEXT NOT NULL,
                config      TEXT NOT NULL,
                status      TEXT DEFAULT 'pending',
                created_at  TEXT NOT NULL,
                updated_at  TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS metrics (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                exp_id      TEXT NOT NULL,
                timestep    INTEGER NOT NULL,
                method      TEXT NOT NULL,
                cta         REAL,
                kl_div      REAL,
                fr          REAL,
                qe          REAL,
                sds_score   REAL,
                teacher_ver INTEGER,
                recorded_at TEXT NOT NULL,
                FOREIGN KEY (exp_id) REFERENCES experiments(id)
            );

            CREATE TABLE IF NOT EXISTS drift_events (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                exp_id      TEXT NOT NULL,
                timestep    INTEGER NOT NULL,
                sds_score   REAL NOT NULL,
                drift_type  TEXT,
                detected    INTEGER DEFAULT 0,
                recorded_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS system_stats (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                cpu_pct     REAL,
                mem_pct     REAL,
                gpu_pct     REAL,
                disk_pct    REAL,
                recorded_at TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_metrics_exp ON metrics(exp_id);
            CREATE INDEX IF NOT EXISTS idx_metrics_ts  ON metrics(exp_id, timestep);
        """)
    print(f"[DB] Initialized at {DB_PATH}")

# ══════════════════════════════════════════════════════════════════════════
#  CORS & REQUEST HELPERS
# ══════════════════════════════════════════════════════════════════════════

@app.after_request
def add_cors(response):
    response.headers['Access-Control-Allow-Origin']  = '*'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type,Authorization'
    response.headers['Access-Control-Allow-Methods'] = 'GET,POST,PUT,DELETE,OPTIONS'
    return response

@app.before_request
def handle_preflight():
    if request.method == 'OPTIONS':
        return '', 204

def ok(data, status=200):
    return jsonify({"success": True,  "data": data}), status

def err(msg, status=400):
    return jsonify({"success": False, "error": msg}), status

def now_iso():
    return datetime.datetime.utcnow().isoformat() + "Z"

def gen_id():
    return f"exp_{int(time.time()*1000)}_{random.randint(100,999)}"

# ══════════════════════════════════════════════════════════════════════════
#  DISTILLATION SIMULATION ENGINE
#  (In production: replace simulate_* with real PyTorch training loops)
# ══════════════════════════════════════════════════════════════════════════

METHODS = ["CT-BBKD", "TemporalEWC-KD", "DAR", "AAR", "Online-FT", "Static"]

# Empirically tuned baselines matching paper results
METHOD_PARAMS = {
    "CT-BBKD":        {"base": 67, "slope": 0.48, "noise": 0.6,  "fr_base": 4.0,  "recovery": 0.85},
    "TemporalEWC-KD": {"base": 65, "slope": 0.45, "noise": 0.7,  "fr_base": 4.3,  "recovery": 0.78},
    "DAR":            {"base": 64, "slope": 0.42, "noise": 0.8,  "fr_base": 5.2,  "recovery": 0.72},
    "AAR":            {"base": 65, "slope": 0.44, "noise": 0.7,  "fr_base": 4.6,  "recovery": 0.92},
    "Online-FT":      {"base": 66, "slope": 0.20, "noise": 1.0,  "fr_base": 18.0, "recovery": 0.45},
    "Static":         {"base": 63, "slope": 0.00, "noise": 0.4,  "fr_base": 0.0,  "recovery": 0.00},
}

def simulate_cta(method, t, teacher_ver, drift_t, rng):
    """Simulate Current Teacher Agreement for a method at timestep t."""
    p = METHOD_PARAMS[method]
    base_cta = p["base"] + p["slope"] * t

    # Teacher version jump effect
    since_drift = t - drift_t if t >= drift_t else 0
    if since_drift == 0 and drift_t > 0:
        drop = 12 if method == "Online-FT" else (4 if method == "Static" else 6)
        base_cta -= drop
    elif since_drift > 0:
        # Recovery curve
        recover = p["recovery"] * (1 - math.exp(-since_drift / 5)) * 8
        base_cta += recover - (8 if since_drift < 2 else 0)

    noise = rng.gauss(0, p["noise"])
    return round(min(92, max(45, base_cta + noise)), 2)

def simulate_sds(t, drift_events, rng):
    """Simulate Spectral Drift Score - spikes at drift events."""
    base = abs(rng.gauss(0.01, 0.008))
    for de in drift_events:
        if t == de:
            base += rng.uniform(0.28, 0.48)
        elif t == de + 1:
            base += rng.uniform(0.10, 0.22)
        elif t == de + 2:
            base += rng.uniform(0.03, 0.08)
    return round(min(0.6, base), 4)

def simulate_fr(method, t, rng):
    """Simulate Forgetting Rate."""
    p = METHOD_PARAMS[method]
    base = p["fr_base"] + rng.gauss(0, 0.3)
    if method == "Online-FT":
        base += t * 0.12  # grows over time
    return round(max(0, min(30, base)), 2)

def simulate_kl(cta):
    """KL divergence inversely related to CTA."""
    return round(max(0.01, (100 - cta) / 120 + random.gauss(0, 0.01)), 4)

def run_experiment_bg(exp_id, config):
    """Background thread: runs the full distillation simulation."""
    rng = random.Random(config.get("seed", 42))
    T   = config.get("timesteps", 45)
    drift_schedule = config.get("drift_schedule", {15: 2, 35: 3})
    # Convert string keys from JSON
    drift_schedule = {int(k): v for k, v in drift_schedule.items()}
    drift_times    = sorted(drift_schedule.keys())

    current_ver    = 1
    query_counts   = {m: 0 for m in METHODS}
    interval       = config.get("interval_ms", 400) / 1000.0

    with _state_lock:
        _active_experiments[exp_id]["status"] = "running"
        _active_experiments[exp_id]["progress"] = 0

    try:
        with get_db() as db:
            db.execute("UPDATE experiments SET status='running', updated_at=? WHERE id=?",
                       (now_iso(), exp_id))

        for t in range(T):
            # Teacher version update
            if t in drift_schedule:
                current_ver = drift_schedule[t]

            # SDS
            sds = simulate_sds(t, drift_times, rng)
            drift_detected = sds > 0.06

            # Record drift event
            if drift_detected:
                with get_db() as db:
                    db.execute("""INSERT INTO drift_events
                                  (exp_id,timestep,sds_score,drift_type,detected,recorded_at)
                                  VALUES (?,?,?,?,?,?)""",
                               (exp_id, t, sds,
                                "version_update" if t in drift_times else "gradual",
                                1, now_iso()))

            # Metrics per method
            rows = []
            for method in METHODS:
                latest_drift = max([dt for dt in drift_times if dt <= t], default=0)
                cta = simulate_cta(method, t, current_ver, latest_drift, rng)
                fr  = simulate_fr(method, t, rng)
                kl  = simulate_kl(cta)
                # QE = CTA per 1K queries
                queries = max(1, T * 64 // (T if method != "Full-Redist" else 1))
                query_counts[method] += 64
                qe = round(cta / (query_counts[method] / 1000), 3)
                rows.append((exp_id, t, method, cta, kl, fr, qe, sds, current_ver, now_iso()))

            with get_db() as db:
                db.executemany("""INSERT INTO metrics
                                  (exp_id,timestep,method,cta,kl_div,fr,qe,sds_score,teacher_ver,recorded_at)
                                  VALUES (?,?,?,?,?,?,?,?,?,?)""", rows)

            # Cache latest for SSE streaming
            with _state_lock:
                if exp_id not in _metrics_cache:
                    _metrics_cache[exp_id] = []
                _metrics_cache[exp_id].append({
                    "t": t, "ver": current_ver, "sds": sds,
                    "metrics": {m: {"cta": r[3], "fr": r[5], "qe": r[6]}
                                for m, r in zip(METHODS, rows)}
                })
                _active_experiments[exp_id]["progress"] = round((t+1)/T*100, 1)
                _active_experiments[exp_id]["current_t"] = t
                _active_experiments[exp_id]["teacher_ver"] = current_ver

            time.sleep(interval)

        # Complete
        with get_db() as db:
            db.execute("UPDATE experiments SET status='complete', updated_at=? WHERE id=?",
                       (now_iso(), exp_id))
        with _state_lock:
            _active_experiments[exp_id]["status"] = "complete"
            _active_experiments[exp_id]["progress"] = 100

    except Exception as e:
        with get_db() as db:
            db.execute("UPDATE experiments SET status='failed', updated_at=? WHERE id=?",
                       (now_iso(), exp_id))
        with _state_lock:
            if exp_id in _active_experiments:
                _active_experiments[exp_id]["status"] = "failed"
                _active_experiments[exp_id]["error"] = str(e)
        print(f"[ERROR] Experiment {exp_id} failed: {e}")

# ══════════════════════════════════════════════════════════════════════════
#  SYSTEM STATS COLLECTOR
# ══════════════════════════════════════════════════════════════════════════

def collect_system_stats():
    """Background thread: collects CPU/RAM every 5 seconds."""
    while True:
        try:
            cpu = psutil.cpu_percent(interval=1)
            mem = psutil.virtual_memory().percent
            disk = psutil.disk_usage('/').percent
            # Simulate GPU (in production: use nvidia-smi or pynvml)
            gpu = random.uniform(30, 85)
            with get_db() as db:
                db.execute("""INSERT INTO system_stats (cpu_pct,mem_pct,gpu_pct,disk_pct,recorded_at)
                              VALUES (?,?,?,?,?)""",
                           (cpu, mem, gpu, disk, now_iso()))
        except Exception:
            pass
        time.sleep(5)

# ══════════════════════════════════════════════════════════════════════════
#  API ROUTES — EXPERIMENTS
# ══════════════════════════════════════════════════════════════════════════

@app.route("/api/v1/health", methods=["GET"])
def health():
    return ok({
        "status": "healthy",
        "version": "1.0.0",
        "timestamp": now_iso(),
        "db": str(DB_PATH),
        "active_experiments": len(_active_experiments)
    })

@app.route("/api/v1/experiments", methods=["GET"])
def list_experiments():
    with get_db() as db:
        rows = db.execute(
            "SELECT * FROM experiments ORDER BY created_at DESC LIMIT 50"
        ).fetchall()
    experiments = []
    for r in rows:
        exp = dict(r)
        exp["config"] = json.loads(exp["config"])
        # Merge live state if available
        with _state_lock:
            live = _active_experiments.get(exp["id"], {})
        exp["progress"]    = live.get("progress", 100 if exp["status"] == "complete" else 0)
        exp["current_t"]   = live.get("current_t", 0)
        exp["teacher_ver"] = live.get("teacher_ver", 1)
        experiments.append(exp)
    return ok(experiments)

@app.route("/api/v1/experiments", methods=["POST"])
def create_experiment():
    body = request.get_json(force=True) or {}
    exp_id = gen_id()
    config = {
        "timesteps":       body.get("timesteps", 45),
        "drift_schedule":  body.get("drift_schedule", {"15": 2, "35": 3}),
        "regime":          body.get("regime", "sudden_update"),
        "student_arch":    body.get("student_arch", "resnet18"),
        "teacher_arch":    body.get("teacher_arch", "resnet50"),
        "dataset":         body.get("dataset", "cifar100"),
        "temp":            body.get("temp", 3.0),
        "lambda_ewc":      body.get("lambda_ewc", 200.0),
        "gamma_dar":       body.get("gamma_dar", 0.4),
        "seed":            body.get("seed", 42),
        "interval_ms":     body.get("interval_ms", 300),
    }
    name = body.get("name", f"Experiment {exp_id[-6:]}")

    with get_db() as db:
        db.execute("""INSERT INTO experiments (id,name,config,status,created_at,updated_at)
                      VALUES (?,?,?,?,?,?)""",
                   (exp_id, name, json.dumps(config), "pending", now_iso(), now_iso()))

    with _state_lock:
        _active_experiments[exp_id] = {
            "id": exp_id, "status": "pending",
            "progress": 0, "current_t": 0, "teacher_ver": 1
        }

    # Launch background thread
    t = threading.Thread(target=run_experiment_bg, args=(exp_id, config), daemon=True)
    t.start()

    return ok({"id": exp_id, "name": name, "status": "running", "config": config}, 201)

@app.route("/api/v1/experiments/<exp_id>", methods=["GET"])
def get_experiment(exp_id):
    with get_db() as db:
        row = db.execute("SELECT * FROM experiments WHERE id=?", (exp_id,)).fetchone()
    if not row:
        return err("Experiment not found", 404)
    exp = dict(row)
    exp["config"] = json.loads(exp["config"])
    with _state_lock:
        live = _active_experiments.get(exp_id, {})
    exp["progress"]  = live.get("progress", 100 if exp["status"] == "complete" else 0)
    exp["current_t"] = live.get("current_t", 0)
    return ok(exp)

@app.route("/api/v1/experiments/<exp_id>", methods=["DELETE"])
def delete_experiment(exp_id):
    with get_db() as db:
        db.execute("DELETE FROM metrics WHERE exp_id=?", (exp_id,))
        db.execute("DELETE FROM drift_events WHERE exp_id=?", (exp_id,))
        db.execute("DELETE FROM experiments WHERE id=?", (exp_id,))
    with _state_lock:
        _active_experiments.pop(exp_id, None)
        _metrics_cache.pop(exp_id, None)
    return ok({"deleted": exp_id})

# ══════════════════════════════════════════════════════════════════════════
#  API ROUTES — METRICS
# ══════════════════════════════════════════════════════════════════════════

@app.route("/api/v1/experiments/<exp_id>/metrics", methods=["GET"])
def get_metrics(exp_id):
    method = request.args.get("method")
    limit  = int(request.args.get("limit", 500))
    query  = """SELECT timestep,method,cta,kl_div,fr,qe,sds_score,teacher_ver
                FROM metrics WHERE exp_id=?"""
    params = [exp_id]
    if method:
        query += " AND method=?"
        params.append(method)
    query += " ORDER BY timestep,method LIMIT ?"
    params.append(limit)

    with get_db() as db:
        rows = db.execute(query, params).fetchall()

    # Pivot by method
    by_method = {}
    sds_series = {}
    for r in rows:
        m = r["method"]
        t = r["timestep"]
        if m not in by_method:
            by_method[m] = {"timesteps":[], "cta":[], "fr":[], "qe":[], "kl":[]}
        by_method[m]["timesteps"].append(t)
        by_method[m]["cta"].append(r["cta"])
        by_method[m]["fr"].append(r["fr"])
        by_method[m]["qe"].append(r["qe"])
        by_method[m]["kl"].append(r["kl_div"])
        sds_series[t] = r["sds_score"]

    return ok({
        "exp_id": exp_id,
        "by_method": by_method,
        "sds_series": [{"t": t, "sds": v} for t, v in sorted(sds_series.items())],
        "total_points": len(rows)
    })

@app.route("/api/v1/experiments/<exp_id>/summary", methods=["GET"])
def get_summary(exp_id):
    with get_db() as db:
        rows = db.execute("""
            SELECT method,
                   AVG(cta) as mean_cta, MIN(cta) as min_cta, MAX(cta) as max_cta,
                   AVG(fr)  as mean_fr,  AVG(qe)  as mean_qe, AVG(kl_div) as mean_kl,
                   COUNT(*) as n_points
            FROM metrics WHERE exp_id=?
            GROUP BY method
        """, (exp_id,)).fetchall()

        drift_rows = db.execute(
            "SELECT * FROM drift_events WHERE exp_id=? ORDER BY timestep",
            (exp_id,)
        ).fetchall()

    summary = {}
    for r in rows:
        summary[r["method"]] = {
            "mean_cta": round(r["mean_cta"] or 0, 2),
            "min_cta":  round(r["min_cta"]  or 0, 2),
            "max_cta":  round(r["max_cta"]  or 0, 2),
            "mean_fr":  round(r["mean_fr"]  or 0, 2),
            "mean_qe":  round(r["mean_qe"]  or 0, 3),
            "mean_kl":  round(r["mean_kl"]  or 0, 4),
            "n_points": r["n_points"],
        }

    # Compute query efficiency vs CT-BBKD baseline
    ct_qe = summary.get("CT-BBKD", {}).get("mean_qe", 1)
    for m in summary:
        qe = summary[m]["mean_qe"]
        summary[m]["qe_vs_ct"] = round(ct_qe / qe, 2) if qe > 0 else 0

    return ok({
        "exp_id":       exp_id,
        "summary":      summary,
        "drift_events": [dict(r) for r in drift_rows],
        "n_drift":      len(drift_rows),
    })

@app.route("/api/v1/experiments/<exp_id>/drift", methods=["GET"])
def get_drift(exp_id):
    with get_db() as db:
        rows = db.execute(
            "SELECT * FROM drift_events WHERE exp_id=? ORDER BY timestep",
            (exp_id,)
        ).fetchall()
    return ok([dict(r) for r in rows])

# ══════════════════════════════════════════════════════════════════════════
#  API ROUTES — SYSTEM MONITORING
# ══════════════════════════════════════════════════════════════════════════

@app.route("/api/v1/system/stats", methods=["GET"])
def system_stats():
    limit = int(request.args.get("limit", 60))
    with get_db() as db:
        rows = db.execute(
            "SELECT * FROM system_stats ORDER BY recorded_at DESC LIMIT ?", (limit,)
        ).fetchall()

    current = {
        "cpu":  psutil.cpu_percent(),
        "mem":  psutil.virtual_memory().percent,
        "disk": psutil.disk_usage('/').percent,
        "gpu":  round(random.uniform(30, 75), 1),
        "timestamp": now_iso()
    }
    return ok({
        "current": current,
        "history": [dict(r) for r in reversed(rows)]
    })

@app.route("/api/v1/system/overview", methods=["GET"])
def system_overview():
    with get_db() as db:
        total_exps = db.execute("SELECT COUNT(*) as n FROM experiments").fetchone()["n"]
        running    = db.execute("SELECT COUNT(*) as n FROM experiments WHERE status='running'").fetchone()["n"]
        complete   = db.execute("SELECT COUNT(*) as n FROM experiments WHERE status='complete'").fetchone()["n"]
        total_m    = db.execute("SELECT COUNT(*) as n FROM metrics").fetchone()["n"]
        total_d    = db.execute("SELECT COUNT(*) as n FROM drift_events").fetchone()["n"]

    with _state_lock:
        active_ids = list(_active_experiments.keys())

    return ok({
        "experiments": {"total": total_exps, "running": running, "complete": complete},
        "data_points":  total_m,
        "drift_events": total_d,
        "active_ids":   active_ids,
        "uptime_sec":   round(time.time() - _start_time, 0),
        "timestamp":    now_iso()
    })

# ══════════════════════════════════════════════════════════════════════════
#  SERVER-SENT EVENTS — Real-time streaming
# ══════════════════════════════════════════════════════════════════════════

@app.route("/api/v1/stream/<exp_id>")
def stream_experiment(exp_id):
    """SSE endpoint: streams live metrics as experiment runs."""
    def generate():
        last_idx = 0
        while True:
            with _state_lock:
                cache    = _metrics_cache.get(exp_id, [])
                new_data = cache[last_idx:]
                status   = _active_experiments.get(exp_id, {}).get("status", "unknown")

            for item in new_data:
                yield f"data: {json.dumps(item)}\n\n"
                last_idx += 1

            if status in ("complete", "failed") and last_idx >= len(cache):
                yield f"data: {json.dumps({'done': True, 'status': status})}\n\n"
                break

            time.sleep(0.5)

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={
            "Cache-Control":  "no-cache",
            "X-Accel-Buffering": "no"
        }
    )

@app.route("/api/v1/stream/system")
def stream_system():
    """SSE endpoint: streams live system stats."""
    def generate():
        while True:
            data = {
                "cpu":  psutil.cpu_percent(),
                "mem":  psutil.virtual_memory().percent,
                "gpu":  round(random.uniform(30, 75), 1),
                "disk": psutil.disk_usage('/').percent,
                "ts":   now_iso()
            }
            yield f"data: {json.dumps(data)}\n\n"
            time.sleep(2)

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
    )

# ══════════════════════════════════════════════════════════════════════════
#  API ROUTES — QUICK ACTIONS
# ══════════════════════════════════════════════════════════════════════════

@app.route("/api/v1/experiments/quick-start", methods=["POST"])
def quick_start():
    """Launch a pre-configured demo experiment immediately."""
    body = request.get_json(force=True) or {}
    regime = body.get("regime", "sudden_update")

    configs = {
        "sudden_update": {
            "name": "Demo — Sudden Update",
            "timesteps": 45,
            "drift_schedule": {"15": 2, "35": 3},
            "regime": "sudden_update",
            "interval_ms": 250
        },
        "gradual_drift": {
            "name": "Demo — Gradual Drift",
            "timesteps": 50,
            "drift_schedule": {"25": 2, "45": 3},
            "regime": "gradual_drift",
            "interval_ms": 200
        },
        "alignment_shift": {
            "name": "Demo — Alignment Shift",
            "timesteps": 40,
            "drift_schedule": {"20": 2},
            "regime": "alignment_shift",
            "interval_ms": 300
        }
    }
    cfg = configs.get(regime, configs["sudden_update"])
    # Delegate to create_experiment
    with app.test_request_context(json=cfg, method="POST"):
        pass

    exp_id = gen_id()
    config = {**cfg, "student_arch":"resnet18","teacher_arch":"resnet50",
              "dataset":"cifar100","temp":3.0,"lambda_ewc":200.0,"gamma_dar":0.4,"seed":42}

    with get_db() as db:
        db.execute("""INSERT INTO experiments (id,name,config,status,created_at,updated_at)
                      VALUES (?,?,?,?,?,?)""",
                   (exp_id, cfg["name"], json.dumps(config), "pending", now_iso(), now_iso()))

    with _state_lock:
        _active_experiments[exp_id] = {"id":exp_id,"status":"pending","progress":0,"current_t":0,"teacher_ver":1}

    t = threading.Thread(target=run_experiment_bg, args=(exp_id, config), daemon=True)
    t.start()
    return ok({"id": exp_id, "name": cfg["name"], "regime": regime}, 201)

# ══════════════════════════════════════════════════════════════════════════
#  STARTUP
# ══════════════════════════════════════════════════════════════════════════

_start_time = time.time()

if __name__ == "__main__":
    init_db()
    # Start system stats collector
    stats_thread = threading.Thread(target=collect_system_stats, daemon=True)
    stats_thread.start()
    print("=" * 60)
    print("  CT-BBKD Backend API")
    print("  Running on http://localhost:5000")
    print("=" * 60)
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
