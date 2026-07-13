#!/usr/bin/env python3
"""
main.py — Optimasi Parameter Backtest + Web Service untuk Render
Mode CLI  : python main.py --evals 30 --csv "crypto_m1_3bulan (1).csv.gz"
Mode Web  : python main.py  (tanpa argumen) → jalankan Flask di PORT
"""

import os
import sys
import json
import time
import argparse
import threading
import numpy as np
import pandas as pd
from functools import partial

# --- Flask untuk web service ---
try:
    from flask import Flask, jsonify, request, render_template_string, send_file
    FLASK_AVAILABLE = True
except ImportError:
    FLASK_AVAILABLE = False
    print("⚠️  Flask tidak terinstall. Mode web tidak tersedia.")
    print("   Install: pip install flask")

# --- scikit-optimize ---
try:
    from skopt import gp_minimize
    from skopt.space import Integer, Real, Categorical
    from skopt.utils import use_named_args
    SKOPT_AVAILABLE = True
except ImportError:
    SKOPT_AVAILABLE = False
    print("⚠️  scikit-optimize tidak terinstall. Install: pip install scikit-optimize")

# --- Import backtest engine ---
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from backtest_4 import run_backtest_for_params, STARTING_BALANCE
except ImportError:
    # fallback untuk file dengan tanda hubung
    import importlib.util
    spec = importlib.util.spec_from_file_location("backtest_4", "backtest-4.py")
    bt = importlib.util.module_from_spec(spec)
    sys.modules["backtest_4"] = bt
    spec.loader.exec_module(bt)
    run_backtest_for_params = bt.run_backtest_for_params
    STARTING_BALANCE = bt.STARTING_BALANCE

# ============================================================
# KONFIGURASI
# ============================================================
DEFAULT_CSV = "crypto_m1_3bulan (1).csv.gz"   # nama file dengan spasi dan .gz
N_EVALS = 30
CHECKPOINT_FILE = "optimization_results.csv"
BEST_PARAMS_FILE = "best_params.json"

# Flag global untuk menghentikan optimasi (jika dijalankan di background)
_optimization_running = False
_optimization_stop_flag = False

# Definisi ruang parameter
PARAM_SPACE = [
    ('min_conf', 'int', 30, 70, 45),
    ('struct_trail_lb', 'int', 1, 5, 2),
    ('struct_trail_buf_pct', 'float', 0.0005, 0.005, 0.0015),
    ('trail_thresh_05', 'float', 0.3, 0.8, 0.5),
    ('trail_lock_05', 'float', 0.1, 0.4, 0.18),
    ('trail_thresh_10', 'float', 0.6, 1.4, 1.0),
    ('trail_lock_10', 'float', 0.2, 0.6, 0.38),
    ('dense_scan', 'bool', None, None, True),
]

# ============================================================
# FUNGSI BANTU
# ============================================================
def build_params_from_args(args):
    min_conf = int(args[0])
    struct_lb = int(args[1])
    struct_buf = float(args[2])
    trail_th_05 = float(args[3])
    trail_lock_05 = float(args[4])
    trail_th_10 = float(args[5])
    trail_lock_10 = float(args[6])
    dense_scan = bool(args[7]) if len(args) > 7 else True

    ladder = sorted([
        (trail_th_05, trail_lock_05),
        (trail_th_10, trail_lock_10),
        (1.5, 0.55),
        (2.0, 0.70),
        (2.8, 0.85),
    ], key=lambda x: x[0])

    return {
        'min_conf': min_conf,
        'struct_trail_lb': struct_lb,
        'struct_trail_buf_pct': struct_buf,
        'trail_ladder': ladder,
        'dense_scan': dense_scan,
    }


def fitness(args, csv_path, verbose=False):
    params = build_params_from_args(args)
    try:
        result = run_backtest_for_params(params, csv_path=csv_path, verbose=verbose)
    except Exception as e:
        print(f"❌ Error pada backtest: {e}")
        return 1e9, None

    total = result['total_trades']
    if total < 10:
        return 1e9, None

    win_rate = result['win_rate']
    profit_factor = result['profit_factor']
    max_dd = result['max_drawdown_pct']

    if np.isinf(profit_factor):
        score = 1000
    else:
        score = profit_factor * (1 + win_rate / 100) / (1 + max_dd / 20)

    log_entry = {
        'min_conf': params['min_conf'],
        'struct_lb': params['struct_trail_lb'],
        'struct_buf': params['struct_trail_buf_pct'],
        'trail_th_05': params['trail_ladder'][0][0],
        'trail_lock_05': params['trail_ladder'][0][1],
        'trail_th_10': params['trail_ladder'][1][0],
        'trail_lock_10': params['trail_ladder'][1][1],
        'dense_scan': params['dense_scan'],
        'total_trades': total,
        'win_rate': win_rate,
        'profit_factor': profit_factor,
        'pnl_pct': result['pnl_pct'],
        'max_drawdown_pct': max_dd,
        'score': score,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    }
    return -score, log_entry


def save_checkpoint(log_entry, csv_path, iteration):
    log_entry['iteration'] = iteration
    df_new = pd.DataFrame([log_entry])
    if os.path.exists(csv_path):
        df_existing = pd.read_csv(csv_path)
        if 'iteration' in df_existing.columns and iteration in df_existing['iteration'].values:
            df_existing = df_existing[df_existing['iteration'] != iteration]
            df_combined = pd.concat([df_existing, df_new], ignore_index=True)
        else:
            df_combined = pd.concat([df_existing, df_new], ignore_index=True)
    else:
        df_combined = df_new
    df_combined.to_csv(csv_path, index=False)


def load_checkpoint(csv_path):
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        if 'iteration' in df.columns:
            df = df.sort_values('iteration')
            logs = df.to_dict('records')
            last_iter = int(df['iteration'].max()) if not df.empty else 0
            return logs, last_iter
    return [], 0


def save_best_params(params, score):
    with open(BEST_PARAMS_FILE, 'w') as f:
        json.dump({
            'params': params,
            'score': score,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        }, f, indent=2)
    print(f"💾 Best params disimpan ke {BEST_PARAMS_FILE} (score={score:.2f})")


# ============================================================
# INTI OPTIMASI (dipanggil dari CLI atau dari thread web)
# ============================================================
def run_optimization(csv_path, n_evals=N_EVALS, verbose=True, resume=True):
    global _optimization_running, _optimization_stop_flag

    if _optimization_running:
        print("⚠️  Optimasi sedang berjalan, tidak bisa memulai yang baru.")
        return

    _optimization_running = True
    _optimization_stop_flag = False

    try:
        if n_evals < 5:
            print(f"⚠️  n_evals={n_evals} terlalu kecil, dinaikkan ke 5 (minimum)")
            n_evals = 5

        print(f"\n🚀 Memulai optimasi parameter backtest")
        print(f"   Data: {csv_path}")
        print(f"   Evaluasi: {n_evals} kali")
        print(f"   Checkpoint: {CHECKPOINT_FILE}")
        print()

        logs, last_iter = load_checkpoint(CHECKPOINT_FILE)
        if logs and resume:
            print(f"📂 Checkpoint ditemukan: {len(logs)} iterasi sudah selesai (terakhir iterasi {last_iter})")
            best_log = max(logs, key=lambda x: x.get('score', -1e9))
            print(f"   Best score saat ini: {best_log['score']:.2f} (iterasi {best_log['iteration']})")
        else:
            print("📂 Checkpoint tidak ditemukan, mulai dari awal.")

        dimensions = []
        default_args = []
        for name, dtype, low, high, default in PARAM_SPACE:
            if dtype == 'int':
                dimensions.append(Integer(low, high, name=name))
            elif dtype == 'float':
                dimensions.append(Real(low, high, name=name))
            elif dtype == 'bool':
                dimensions.append(Categorical([0, 1], name=name))
            default_args.append(default)

        initial_x = None
        if logs and resume:
            best_log = max(logs, key=lambda x: x.get('score', -1e9))
            initial_x = (
                best_log['min_conf'],
                best_log['struct_lb'],
                best_log['struct_buf'],
                best_log['trail_th_05'],
                best_log['trail_lock_05'],
                best_log['trail_th_10'],
                best_log['trail_lock_10'],
                1 if best_log['dense_scan'] else 0,
            )
            print(f"   Initial point diambil dari best score: {initial_x}")

        eval_counter = last_iter + 1

        @use_named_args(dimensions)
        def objective(**params):
            nonlocal eval_counter   # <-- DEKLARASI NONLOCAL DI AWAL

            if _optimization_stop_flag:
                print("🛑 Optimasi dihentikan oleh user.")
                return 1e9
            args = [params.get(p[0]) for p in PARAM_SPACE]
            neg_score, log_entry = fitness(args, csv_path, verbose=verbose)
            if log_entry is not None:
                save_checkpoint(log_entry, CHECKPOINT_FILE, eval_counter)
                if neg_score < 0:
                    current_score = -neg_score
                    best_so_far = -1e9
                    if os.path.exists(BEST_PARAMS_FILE):
                        with open(BEST_PARAMS_FILE, 'r') as f:
                            best_data = json.load(f)
                            best_so_far = best_data.get('score', -1e9)
                    if current_score > best_so_far:
                        save_best_params(build_params_from_args(args), current_score)
            eval_counter += 1
            return neg_score

        if SKOPT_AVAILABLE:
            n_calls = max(5, n_evals - last_iter)
            if n_calls <= 0:
                print("✅ Semua evaluasi sudah selesai dari checkpoint.")
                if logs:
                    best_log = max(logs, key=lambda x: x.get('score', -1e9))
                    print(f"\n🏆 Best score: {best_log['score']:.2f}")
                    print("   Parameter terbaik:")
                    best_args = (
                        best_log['min_conf'],
                        best_log['struct_lb'],
                        best_log['struct_buf'],
                        best_log['trail_th_05'],
                        best_log['trail_lock_05'],
                        best_log['trail_th_10'],
                        best_log['trail_lock_10'],
                        1 if best_log['dense_scan'] else 0,
                    )
                    for k, v in build_params_from_args(best_args).items():
                        print(f"     {k}: {v}")
                return

            print(f"🔍 Menjalankan {n_calls} evaluasi tambahan...")
            res = gp_minimize(
                objective,
                dimensions,
                n_calls=n_calls,
                n_initial_points=min(10, max(5, n_calls//2)),
                acq_func='EI',
                random_state=42,
                verbose=verbose,
                x0=initial_x,
            )
            best_args = res.x
            best_score = -res.fun
        else:
            # Random search fallback
            print("🔁 Menggunakan random search")
            best_score = -1e9
            best_args = None
            n_random = max(50, n_evals * 2)
            for i in range(n_random):
                if _optimization_stop_flag:
                    print("🛑 Optimasi dihentikan oleh user.")
                    break
                args = []
                for dim in dimensions:
                    if isinstance(dim, Integer):
                        args.append(np.random.randint(dim.low, dim.high + 1))
                    elif isinstance(dim, Real):
                        args.append(np.random.uniform(dim.low, dim.high))
                    elif isinstance(dim, Categorical):
                        args.append(np.random.choice(dim.categories))
                neg_score, log_entry = fitness(args, csv_path, verbose=False)
                if log_entry is not None:
                    save_checkpoint(log_entry, CHECKPOINT_FILE, eval_counter)
                    eval_counter += 1
                    if neg_score < -best_score:
                        best_score = -neg_score
                        best_args = args
                        save_best_params(build_params_from_args(best_args), best_score)
                if i % 10 == 0:
                    print(f"  Random {i+1}/{n_random}: best score = {best_score:.2f}")

        if best_args is not None and not _optimization_stop_flag:
            best_params = build_params_from_args(best_args)
            print("\n✅ Optimasi selesai.")
            print(f"   Score terbaik: {best_score:.2f}")
            print("   Parameter terbaik:")
            for k, v in best_params.items():
                if k == 'trail_ladder':
                    print(f"     {k}: {v}")
                else:
                    print(f"     {k}: {v}")

            # Jalankan backtest final
            print("\n📊 Menjalankan backtest final dengan parameter terbaik...")
            final_result = run_backtest_for_params(best_params, csv_path=csv_path, verbose=True)
            print("\n=== HASIL FINAL ===")
            print(f"Total trades   : {final_result['total_trades']}")
            print(f"Win rate       : {final_result['win_rate']:.2f}%")
            print(f"Profit Factor  : {final_result['profit_factor']:.2f}")
            print(f"PnL (%)        : {final_result['pnl_pct']:.2f}%")
            print(f"Max Drawdown   : {final_result['max_drawdown_pct']:.2f}%")
            print(f"\n💾 Parameter terbaik disimpan ke {BEST_PARAMS_FILE}")
        else:
            print("❌ Tidak ada hasil optimasi (dihentikan atau error).")

    finally:
        _optimization_running = False
        _optimization_stop_flag = False


# ============================================================
# FLASK WEB SERVICE
# ============================================================
if FLASK_AVAILABLE:
    app = Flask(__name__)

    DASHBOARD_HTML = """
    <!DOCTYPE html>
    <html>
    <head><title>Optimasi Backtest SMC</title></head>
    <body>
        <h1>🤖 Optimasi Parameter Backtest</h1>
        <p><strong>Status:</strong> <span id="status">{{ status }}</span></p>
        <p><strong>Iterasi terakhir:</strong> {{ last_iter }}</p>
        <p><strong>Best score:</strong> {{ best_score }}</p>
        <p><a href="/start">▶️ Mulai optimasi (30 iterasi)</a></p>
        <p><a href="/stop">⏹ Hentikan optimasi</a></p>
        <p><a href="/status">📊 Status JSON</a></p>
        <hr>
        <h2>Log terakhir (10 baris)</h2>
        <pre>{{ log_preview }}</pre>
        <p><a href="/logs">📄 Download full log (CSV)</a></p>
    </body>
    </html>
    """

    @app.route('/')
    def dashboard():
        status = "Idle"
        if _optimization_running:
            status = "Running..."
        last_iter = 0
        best_score = "-"
        logs, _ = load_checkpoint(CHECKPOINT_FILE)
        if logs:
            last_iter = logs[-1].get('iteration', 0)
            best = max(logs, key=lambda x: x.get('score', -1e9))
            best_score = f"{best.get('score', 0):.2f}"

        log_preview = ""
        if logs:
            df = pd.DataFrame(logs[-10:])
            log_preview = df.to_string(index=False)

        return render_template_string(DASHBOARD_HTML,
                                      status=status,
                                      last_iter=last_iter,
                                      best_score=best_score,
                                      log_preview=log_preview)

    @app.route('/start')
    def start_optimization():
        global _optimization_running
        if _optimization_running:
            return jsonify({"status": "error", "message": "Optimasi sudah berjalan"}), 400
        thread = threading.Thread(
            target=run_optimization,
            args=(DEFAULT_CSV, N_EVALS, True, True),
            daemon=True
        )
        thread.start()
        return jsonify({"status": "started", "message": f"Optimasi dimulai dengan {N_EVALS} evaluasi"})

    @app.route('/stop')
    def stop_optimization():
        global _optimization_stop_flag
        if not _optimization_running:
            return jsonify({"status": "error", "message": "Optimasi tidak sedang berjalan"}), 400
        _optimization_stop_flag = True
        return jsonify({"status": "stopping", "message": "Permintaan berhenti dikirim"})

    @app.route('/status')
    def status_json():
        logs, last_iter = load_checkpoint(CHECKPOINT_FILE)
        best = None
        if logs:
            best = max(logs, key=lambda x: x.get('score', -1e9))
        best_params_data = None
        if os.path.exists(BEST_PARAMS_FILE):
            with open(BEST_PARAMS_FILE, 'r') as f:
                best_params_data = json.load(f)
        return jsonify({
            "running": _optimization_running,
            "last_iteration": last_iter,
            "best_score": best.get('score') if best else None,
            "best_params": best_params_data,
            "total_entries": len(logs),
        })

    @app.route('/logs')
    def download_logs():
        if os.path.exists(CHECKPOINT_FILE):
            return send_file(CHECKPOINT_FILE, as_attachment=True)
        return "Log file not found", 404

    def run_flask():
        port = int(os.environ.get('PORT', 8080))
        app.run(host='0.0.0.0', port=port, debug=False)


# ============================================================
# CLI MAIN
# ============================================================
def main_cli():
    parser = argparse.ArgumentParser(description="Optimasi parameter backtest SMC")
    parser.add_argument('--csv', default=DEFAULT_CSV, help='Path ke file CSV data M1')
    parser.add_argument('--evals', type=int, default=N_EVALS, help='Jumlah evaluasi')
    parser.add_argument('--no-resume', action='store_true', help='Abaikan checkpoint, mulai ulang')
    parser.add_argument('--verbose', action='store_true', help='Tampilkan detail setiap backtest')
    args = parser.parse_args()

    if not os.path.exists(args.csv):
        print(f"❌ File CSV tidak ditemukan: {args.csv}")
        print("   Pastikan path benar atau berikan --csv <path>")
        sys.exit(1)

    run_optimization(args.csv, n_evals=args.evals, verbose=args.verbose, resume=not args.no_resume)


# ============================================================
# ENTRY POINT
# ============================================================
if __name__ == "__main__":
    if len(sys.argv) > 1:
        main_cli()
    else:
        if FLASK_AVAILABLE:
            print("🌐 Menjalankan Flask web server...")
            print(f"   Akses dashboard di http://localhost:{os.environ.get('PORT', 8080)}")
            run_flask()
        else:
            print("❌ Flask tidak terinstall. Jalankan dengan argumen CLI atau install flask.")
            sys.exit(1)
