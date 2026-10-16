import json, math, sys
for f in sys.argv[1:]:
    d = json.load(open(f)); L = d["log"]
    psi = [abs(math.degrees(e["s"][2])) for e in L]
    i_spin = next((i for i, p in enumerate(psi) if p > 45), None)
    qp = [i for i, e in enumerate(L) if (e["info_last"] or {}).get("status") == "ok"]
    first_qp = qp[0] if qp else None
    # steps where shield output steering is opposite in sign to MPPI's and differs by > 0.3
    rev = sum(1 for e in L if e["info_last"] and e["info_last"]["status"] == "ok"
              and e["u"][1] * e["info_last"]["u_nom"][1] < 0 and abs(e["u"][1] - e["info_last"]["u_nom"][1]) > 0.3)
    print(f"{f}: outcome={d['rec']['outcome']:<10} max|psi|={max(psi):5.1f}deg  first QP step={first_qp}  first |psi|>45 step={i_spin}  "
          f"y at spin={L[i_spin]['s'][1]:+.2f}" if i_spin is not None else f"{f}: outcome={d['rec']['outcome']} max|psi|={max(psi):.1f} no spin", f" reversals={rev}/{len(qp)}")
