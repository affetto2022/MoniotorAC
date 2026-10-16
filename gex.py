"""Calcula Gamma Exposure (GEX) por strike e grava data.json a cada 5 min.
Uso:  pip install yfinance numpy
      python gex.py
      abra http://localhost:8000/dashboard.html
"""
import json, time, threading, functools, http.server, socketserver
from datetime import datetime
import numpy as np
import yfinance as yf

TICKER = "EWZ"
N_EXP = 4          # quantos vencimentos considerar
INTERVAL = 300     # segundos (5 min)
PORT = 8000
R = 0.045          # taxa livre de risco


def bs_gamma(S, K, T, iv):
    d1 = (np.log(S / K) + (R + iv ** 2 / 2) * T) / (iv * np.sqrt(T))
    return np.exp(-d1 ** 2 / 2) / np.sqrt(2 * np.pi) / (S * iv * np.sqrt(T))


def snapshot():
    tk = yf.Ticker(TICKER)
    spot = float(tk.fast_info["last_price"])
    now, rows = datetime.now(), []
    for exp in tk.options[:N_EXP]:
        T = max((datetime.strptime(exp, "%Y-%m-%d") - now).total_seconds() / 31557600, 1 / 365)
        ch = tk.option_chain(exp)
        for df, sign in ((ch.calls, 1), (ch.puts, -1)):
            for _, o in df.iterrows():
                iv, oi = o.impliedVolatility, o.openInterest
                if iv > 0.01 and oi > 0:
                    rows.append((o.strike, T, iv, oi, sign))
    K, T, iv, oi, sg = np.array(rows, dtype=float).T

    def gex_at(S):  # GEX em $ por 1% de movimento
        return sg * bs_gamma(S, K, T, iv) * oi * 100 * S * S * 0.01

    g = gex_at(spot)
    strikes = np.unique(K)
    strikes = strikes[(strikes > spot * 0.75) & (strikes < spot * 1.25)]
    by = [float(g[K == k].sum()) for k in strikes]
    calls = [float(g[(K == k) & (sg > 0)].sum()) for k in strikes]
    puts = [float(g[(K == k) & (sg < 0)].sum()) for k in strikes]

    grid = np.linspace(spot * 0.75, spot * 1.25, 101)
    agg = np.array([gex_at(s).sum() for s in grid])
    flip = None
    for i in range(len(grid) - 1):
        if agg[i] * agg[i + 1] < 0:
            x = grid[i] - agg[i] * (grid[i + 1] - grid[i]) / (agg[i + 1] - agg[i])
            if flip is None or abs(x - spot) < abs(flip - spot):
                flip = float(x)

    return {
        "ts": datetime.now().isoformat(timespec="seconds"),
        "ticker": TICKER, "spot": spot, "flip": flip,
        "call_wall": float(strikes[int(np.argmax(calls))]),
        "put_wall": float(strikes[int(np.argmin(puts))]),
        "total_gex": float(g.sum()),
        "strikes": strikes.tolist(), "gex": by,
        "grid": grid.tolist(), "agg": agg.tolist(),
    }


def serve():
    h = functools.partial(http.server.SimpleHTTPRequestHandler, directory=".")
    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer(("", PORT), h) as s:
        s.serve_forever()


if __name__ == "__main__":
    threading.Thread(target=serve, daemon=True).start()
    print(f"Dashboard: http://localhost:{PORT}/dashboard.html")
    while True:
        try:
            data = snapshot()
            with open("data.json", "w") as f:
                json.dump(data, f)
            print(data["ts"], "spot", round(data["spot"], 2), "flip", data["flip"])
        except Exception as e:
            print("Erro:", e)
        time.sleep(INTERVAL)
