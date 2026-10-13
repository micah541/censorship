"""Verification and tables for the feather-fork appendix (appendix_feather.tex).

For every quantity (W, T, C, G, F) in Cases 1 and 2 this script
  1. solves the Markov-chain linear system exactly (rational arithmetic),
  2. evaluates the closed forms stated in the appendix,
  3. runs a Monte Carlo simulation of the race,
checks that all three agree, and prints the LaTeX tables.

State d = number of blocks the compliant chain is behind; race starts at d=1,
ends at d=-1 (compliant wins) or d=k (compliant concedes).
Compliant hashrate p.  Case 2: a fraction gamma of the remaining hashrate
1-p joins the compliant chain at d=0, so the compliant chain finds the next
block with probability P = p + gamma*(1-p) at d=0 (and p elsewhere).
Attacker (miner of the offending block) has hashrate r and never joins.
"""
from fractions import Fraction as Fr
import random

PS = [Fr(1, 10), Fr(2, 10), Fr(3, 10), Fr(45, 100), Fr(6, 10)]
KS = [2, 5]
GAMMAS = [Fr(2, 10), Fr(5, 10)]
R = Fr(1, 10)


# ---------------------------------------------------------------- exact solve
def solve_chain(p, k, P, run, lo, hi):
    """V(d) = sum_{d'} Pr(d->d') [run(d,d') + V(d')],  V(-1)=lo, V(k)=hi."""
    n = k
    A = [[Fr(0)] * n for _ in range(n)]
    b = [Fr(0)] * n
    for d in range(n):
        up = P if d == 0 else p
        A[d][d] += 1
        for dn, pr in ((d - 1, up), (d + 1, 1 - up)):
            b[d] += pr * run(d, dn)
            if dn == -1:
                b[d] += pr * lo
            elif dn == k:
                b[d] += pr * hi
            else:
                A[d][dn] -= pr
    # Gaussian elimination over Fractions
    for i in range(n):
        piv = next(j for j in range(i, n) if A[j][i] != 0)
        A[i], A[piv] = A[piv], A[i]
        b[i], b[piv] = b[piv], b[i]
        for j in range(n):
            if j != i and A[j][i] != 0:
                c = A[j][i] / A[i][i]
                A[j] = [a - c * ai for a, ai in zip(A[j], A[i])]
                b[j] -= c * b[i]
    return [b[i] / A[i][i] for i in range(n)]


def exact(p, k, gamma=Fr(0), r=R):
    """Returns W, T, C at d=1 and F(1) = alpha*f + beta as (alpha, beta)."""
    P = p + gamma * (1 - p)
    W = solve_chain(p, k, P, lambda d, dn: 0, 1, 0)
    Wf = lambda d: Fr(1) if d == -1 else (Fr(0) if d == k else W[d])
    T = solve_chain(p, k, P, lambda d, dn: 1, 0, 0)

    def runC(d, dn):  # block of the p-group, orphaned unless compliant wins
        up = P if d == 0 else p
        return p / up * (1 - Wf(dn)) if dn == d - 1 else 0

    def runF(d, dn):  # attacker race block, orphaned if compliant wins
        up = P if d == 0 else p
        return -r / (1 - up) * Wf(dn) if dn == d + 1 else 0

    C = solve_chain(p, k, P, runC, 0, 0)
    beta = solve_chain(p, k, P, runF, -1, 0)[1]
    return dict(W=W[1], T=T[1], C=C[1], alpha=1 - W[1], beta=beta)


# --------------------------------------------------------------- closed forms
def c1_W(d, k, p):
    m = p / (1 - p); M = m ** (k + 1)
    return (M - m ** (d + 1)) / (M - 1)


def c1_T(d, k, p):
    return (k - d - (k + 1) * c1_W(d, k, p)) / (1 - 2 * p)


def c1_C(d, k, p):
    m = p / (1 - p); M = m ** (k + 1); w = c1_W(d, k, p)
    return p / ((1 - 2 * p) * (M - 1)) * (
        d * (m ** d + 1) + (1 + 1 / m) * w - k * (m ** k + 1) * (1 - w))


def c1_A(d, k, p):
    """E[# non-compliant race blocks ; compliant wins]."""
    m = p / (1 - p); M = m ** (k + 1); w = c1_W(d, k, p)
    return -(1 - p) / ((1 - 2 * p) * (M - 1)) * (
        d * (M + m ** (d + 2)) + (M + m) * w - k * (M + m ** (k + 2)) * (1 - w))


def closed(p, k, gamma=Fr(0), r=R):
    P = p + gamma * (1 - p)
    if gamma == 0:
        W, T, C = c1_W(1, k, p), c1_T(1, k, p), c1_C(1, k, p)
        return dict(W=W, T=T, C=C, alpha=1 - W,
                    beta=-W - r / (1 - p) * c1_A(1, k, p))
    h, tau = c1_W(0, k - 1, p), c1_T(0, k - 1, p)
    c, a = c1_C(0, k - 1, p), c1_A(0, k - 1, p)
    D = 1 - (1 - P) * h
    B = P / D
    W = P * h / D
    return dict(W=W, T=(tau + h) / D, C=((1 - B) * p * tau + B * c) / D,
                alpha=(1 - h) / D,
                beta=(-r / (1 - p) * B * a - h * (P + r * W)) / D)


def closed_k2(p, gamma, r=R):
    """k = 2 simplifications, Delta = 1 - p(1-P)."""
    P = p + gamma * (1 - p)
    Dl = 1 - p * (1 - P)
    return dict(W=p * P / Dl, T=(1 + p) / Dl, C=p * (1 - p) * (1 - P) / Dl ** 2,
                alpha=(1 - p) / Dl, beta=-p * P / Dl * (1 + r * p / Dl))


# ---------------------------------------------------------------- Monte Carlo
def monte_carlo(p, k, gamma, r=R, n=200_000, seed=1):
    p, gamma, r = float(p), float(gamma), float(r)
    P = p + gamma * (1 - p)
    rng = random.Random(seed)
    sW = sT = sC = sB = 0.0
    for _ in range(n):
        d = 1; t = bp = ba = 0
        while -1 < d < k:
            u = rng.random(); t += 1
            up = P if d == 0 else p
            if u < p:            # p-group block
                d -= 1; bp += 1
            elif u < up:         # gamma-joiner block (only at d=0)
                d -= 1
            elif u < up + r:     # attacker block
                d += 1; ba += 1
            else:
                d += 1
        win = d == -1
        sW += win; sT += t; sC += (not win) * bp
        sB += -(1 + ba) if win else 0
    return dict(W=sW / n, T=sT / n, C=sC / n, beta=sB / n)


# ---------------------------------------------------------------------- main
def check():
    for gamma in [Fr(0)] + GAMMAS:
        for k in KS + [3, 7]:
            for p in PS:
                e, c = exact(p, k, gamma), closed(p, k, gamma)
                assert all(e[x] == c[x] for x in e), (p, k, gamma)
                if k == 2 and gamma:
                    c2 = closed_k2(p, gamma)
                    assert all(e[x] == c2[x] for x in e), (p, gamma)
    print("% exact Markov solve == closed forms (all p, k in {2,3,5,7}, gamma in {0,.2,.5})")
    worst = 0.0
    for gamma in [Fr(0)] + GAMMAS:
        for k in KS:
            for p in PS:
                e, s = exact(p, k, gamma), monte_carlo(p, k, gamma)
                for x in s:
                    se = max(abs(float(e[x])), 1e-3)
                    worst = max(worst, abs(s[x] - float(e[x])) / se)
    print(f"% Monte Carlo (2e5 races each) max relative deviation: {worst:.3f}")


def fmt(x, sign=False):
    return f"{float(x):+.4f}" if sign else f"{float(x):.4f}"


def table(corner, rows, cols=None):
    """cols=None: four columns grouped as gamma in GAMMAS x k in KS."""
    if cols is None:
        n = len(GAMMAS) * len(KS)
        top = " & ".join([corner] + [r"\multicolumn{%d}{c}{$\gamma=%g$}" % (len(KS), float(g))
                                     for g in GAMMAS]) + r"\\"
        head = [top, " & ".join(["$p$"] + [f"$k={k}$" for g in GAMMAS for k in KS]) + r"\\ \hline"]
    else:
        n = len(cols)
        head = [" & ".join([corner] + cols) + r"\\ \hline"]
    out = [r"\[", r"\begin{tabular}[c]{l" + "c" * n + "}"] + head
    out += [" & ".join(row) + r"\\" for row in rows]
    out += [r"\end{tabular}", r"\]"]
    return "\n".join(out)


def tables():
    combos = [(g, k) for g in GAMMAS for k in KS]
    vals = {(g, k, p): exact(p, k, g) for g, k in combos for p in PS}
    P_ = lambda p: f"${float(p):g}$"
    for q, lab in (("W", "W(1)"), ("T", "T(1)"), ("C", "C(1)")):
        print(f"% {q}")
        print(table(f"${lab}$", [[P_(p)] + [f"${fmt(vals[g, k, p][q])}$" for g, k in combos]
                                 for p in PS]))
    for frac, lab in ((Fr(1, 3), "p/3"), (Fr(2, 3), "2p/3")):
        print(f"% G, q={lab}")
        print(table(f"$G(1),q={lab}$", [[P_(p)] + [
            f"${fmt((p - frac * p) * vals[g, k, p]['T'] - vals[g, k, p]['C'], True)}$"
            for g, k in combos] for p in PS]))
    print("% F, r=0.1")
    print(table("$F(1)$", [[P_(p)] + [
        f"${fmt(vals[g, k, p]['alpha'])}f{fmt(vals[g, k, p]['beta'], True)}$" for g, k in combos]
        for p in PS]))
    print("% f*, r=0.1")
    print(table(r"$f^{\ast}$", [[P_(p)] + [
        f"${fmt(-vals[g, k, p]['beta'] / vals[g, k, p]['alpha'])}$" for g, k in combos]
        for p in PS]))
    print("% Case 1 f*, r=0.1")
    print(table("$p$", [[P_(p)] + [
        f"${fmt(-exact(p, k)['beta'] / exact(p, k)['alpha'])}$" for k in KS] for p in PS],
        cols=[r"$f^{\ast},k=%d$" % k for k in KS]))


if __name__ == "__main__":
    check()
    tables()
