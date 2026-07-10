# Monte Carlo verification of closed-form formulas in the appendix.
# Each function simulates one quantity; the print blocks compare sim vs. exact.
# Case 1: all outside miners follow first-seen rule (p_1=p, p_2=1-p fixed).
# Case 2: gamma-fraction of outside miners switch to compliant at d=0,
#         so effective compliant probability at d=0 is P = p+gamma.

import random

# ---------------------------------------------------------------------------
# Case 1 simulation functions
# ---------------------------------------------------------------------------

def simulate_reorg(p, k, q, n_trials=5000, seed=42):
    """Returns (W, T, C, G) from a single pass over n_trials races."""
    rng = random.Random(seed)
    wins = 0
    total_steps = 0
    total_cost = 0.0
    total_G = 0.0
    for _ in range(n_trials):
        d, b, steps = 1, 0, 0
        while d != -1 and d != k:
            if rng.random() < p:
                d -= 1
                b += 1
            else:
                d += 1
            steps += 1
        win = (d == -1)
        if win:
            wins += 1
        else:
            total_cost += b          # orphaned compliant blocks on loss
        total_steps += steps
        total_G += (b if win else 0) - q * steps
    return wins / n_trials, total_steps / n_trials, total_cost / n_trials, total_G / n_trials

def simulate_F(p, k, f, r, n_trials=20000, seed=42):
    """Attacker gain under opportunity-cost model.
    r = attacker hashrate fraction; f = non-compliant block reward.
    Payoff: +f if attacker wins, -(1+b_nc) if compliant wins."""
    rng = random.Random(seed)
    total_F = 0.0
    for _ in range(n_trials):
        d, b_nc = 1, 0
        while d != -1 and d != k:
            u = rng.random()
            if u < p:
                d -= 1              # compliant mines
            elif u < p + r:
                d += 1
                b_nc += 1           # attacker mines a race block
            else:
                d += 1              # other non-compliant mines
        if d == k:
            total_F += f
        else:
            total_F += -1 - b_nc
    return total_F / n_trials

# ---------------------------------------------------------------------------
# Case 1 closed-form expressions
# ---------------------------------------------------------------------------

def G_analytical(p, k, q):
    m = p / (1 - p)
    part1 = (m**(k+1)*(q-p) - q - p*m) / ((1-2*p)*(m**(k+1)-1))
    r1 = (-m**(k+2)*(q-p)*k + m*q*k + p*k*m**(k+1)
          - m**(k+2)*(q-p) + q*m + p)
    r2 = (m**(k+1)*(q-p)*k - q*k - p*k*m**k
          + m**(k+1)*(m**(k+1)*(q-p) - q - p/m))
    pre = 1 / ((1-2*p)*(m**(k+1)-1)**2)
    c1, c2 = pre*r1, pre*r2
    return c1*m + c2 + part1

def F_analytical(p, k, f, r):
    m = p / (1 - p)
    mk1 = m**(k+1)
    denom = (1 - 2*p) * (mk1 - 1)
    Fp_k  = r * k * (mk1 + m**(k+2)) / denom
    Fp_m1 = r * (-1) * (mk1 + m) / denom
    rhs0  = f - Fp_k
    rhs1  = -1 - Fp_m1
    inv_pre = m / (mk1 - 1)
    c1 = inv_pre * (rhs0 - rhs1)
    c2 = inv_pre * (-rhs0 / m + m**k * rhs1)
    Fp_1 = r * 1 * (mk1 + m**3) / denom
    return c1 * m + c2 + Fp_1

# ---------------------------------------------------------------------------
# Case 1 verification tables
# ---------------------------------------------------------------------------

# (p, k, exact_W, exact_T, exact_C)
cases = [
    (0.3, 2, 9/79,  130/79,  1470/6241),
    (0.4, 2, 4/19,  35/19,   90/361),
]

q = 0.1
print("=== Case 1 ===")
print(f"\n{'Quantity':<10} {'p':>5} {'k':>3} {'sim':>10} {'exact':>10} {'diff':>8}")
print("-" * 45)
for p, k, ew, et, ec in cases:
    sw, st, sc, sg = simulate_reorg(p, k, q)   # single pass; reuse all four quantities
    eg = G_analytical(p, k, q)
    print(f"{'W(1)':<10} {p:>5.1f} {k:>3} {sw:>10.4f} {ew:>10.4f} {sw-ew:>+8.4f}")
    print(f"{'T(1)':<10} {p:>5.1f} {k:>3} {st:>10.4f} {et:>10.4f} {st-et:>+8.4f}")
    print(f"{'C(1)':<10} {p:>5.1f} {k:>3} {sc:>10.4f} {ec:>10.4f} {sc-ec:>+8.4f}")
    print(f"{'G(1)':<10} {p:>5.1f} {k:>3} {sg:>10.4f} {eg:>10.4f} {sg-eg:>+8.4f}")

print(f"\nF(1) -- attacker gain (r=0.1)")
print(f"{'p':>5} {'k':>3} {'f':>5} {'sim':>10} {'exact':>10} {'diff':>8}")
print("-" * 48)
r = 0.1
for p, k, *_ in cases:
    for f in [0.0, 0.2]:
        sf = simulate_F(p, k, f, r)
        ef = F_analytical(p, k, f, r)
        print(f"{p:>5.1f} {k:>3} {f:>5.1f} {sf:>10.4f} {ef:>10.4f} {sf-ef:>+8.4f}")

# ---------------------------------------------------------------------------
# Case 2 simulation functions
# ---------------------------------------------------------------------------

def simulate_case2_W(p, k, gamma, start_d=1, n_trials=10000, seed=42):
    P = p + gamma
    rng = random.Random(seed)
    wins = 0
    for _ in range(n_trials):
        d = start_d
        while d != -1 and d != k:
            prob_c = P if d == 0 else p
            if rng.random() < prob_c:
                d -= 1
            else:
                d += 1
        if d == -1:
            wins += 1
    return wins / n_trials

def simulate_case2_T(p, k, gamma, start_d=1, n_trials=10000, seed=42):
    P = p + gamma
    rng = random.Random(seed)
    total_steps = 0
    for _ in range(n_trials):
        d, steps = start_d, 0
        while d != -1 and d != k:
            prob_c = P if d == 0 else p
            if rng.random() < prob_c:
                d -= 1
            else:
                d += 1
            steps += 1
        total_steps += steps
    return total_steps / n_trials

def simulate_case2_C(p, k, gamma, start_d=1, n_trials=20000, seed=42):
    P = p + gamma
    rng = random.Random(seed)
    total_C = 0.0
    for _ in range(n_trials):
        d, b = start_d, 0
        while d != -1 and d != k:
            prob_c = P if d == 0 else p
            if rng.random() < prob_c:
                d -= 1
                b += 1
            else:
                d += 1
        if d == k:          # compliant lost: b blocks orphaned
            total_C += b
    return total_C / n_trials

def simulate_case2_G(p, k, gamma, q, start_d=1, n_trials=20000, seed=42):
    # Track only p-fraction miner blocks (not the gamma-fraction who join at d=0).
    # At d=0: u<p => p-fraction mines; p<=u<P => gamma-fraction mines (not earned by p-side).
    P = p + gamma
    rng = random.Random(seed)
    total_G = 0.0
    for _ in range(n_trials):
        d, b_p, steps = start_d, 0, 0
        while d != -1 and d != k:
            u = rng.random()
            if d == 0:
                if u < p:
                    d -= 1; b_p += 1    # p-fraction compliant block
                elif u < P:
                    d -= 1              # gamma-fraction block (not credited to p-side)
                else:
                    d += 1
            else:
                if u < p:
                    d -= 1; b_p += 1
                else:
                    d += 1
            steps += 1
        total_G += (b_p if d == -1 else 0) - q * steps
    return total_G / n_trials

def simulate_case2_F(p, k, gamma, f, r, n_trials=30000, seed=42):
    P = p + gamma
    rng = random.Random(seed)
    total_F = 0.0
    for _ in range(n_trials):
        d, b_nc = 1, 0
        while d != -1 and d != k:
            u = rng.random()
            if d == 0:
                if u < P:
                    d -= 1              # compliant wins at d=0 (prob P)
                elif u < P + r:
                    d += 1; b_nc += 1   # attacker mines
                else:
                    d += 1
            else:
                if u < p:
                    d -= 1
                elif u < p + r:
                    d += 1; b_nc += 1
                else:
                    d += 1
        if d == k:
            total_F += f
        else:
            total_F += -1 - b_nc
    return total_F / n_trials

# ---------------------------------------------------------------------------
# Case 2 closed-form expressions (k=2)
# ---------------------------------------------------------------------------

def W1_case2_exact(p, k, gamma):
    m = p / (1 - p)
    mk = m**k
    num = (p + gamma) * (mk - m)
    den = (mk - 1) - (1 - p - gamma) * (mk - m)
    return num / den

def T1_case2_exact(p, k, gamma):
    # k=2: T(1) = (1+p) / [(1-p)(1+Pm)]
    m = p / (1 - p)
    P = p + gamma
    if k == 2:
        return (1 + p) / ((1 - p) * (1 + P * m))
    # General k: solve for T(1) via boundary-value consistency
    mk = m**k
    A = (m - mk) / (1 - mk)
    C = k * (m - 1) / ((2*p - 1) * (1 - mk)) + 1 / (2*p - 1)
    return (A + C) / (1 - (1 - p - gamma) * A)

def C1_case2_exact(p, gamma):
    # k=2: C(1) = p(1-P) / [(1+Pm)(1-p(1-P))]
    P = p + gamma
    m = p / (1 - p)
    return p * (1 - P) / ((1 + P * m) * (1 - p * (1 - P)))

def G1_case2_exact(p, gamma, q):
    # Exact identity: G(1) = (p-q)*T(1) - C(1)
    return (p - q) * T1_case2_exact(p, 2, gamma) - C1_case2_exact(p, gamma)

def F1_case2_exact(p, gamma, f, r):
    # k=2: solve 2x2 system at d=0,1.
    # F(1) = p*F(0) + (1-p)*f
    # F(0) = -P + (1-P)*(1-p)*f - r*W(1)
    P = p + gamma
    m = p / (1 - p)
    W1 = P * m / (1 + P * m)
    F0 = (-P + (1 - P) * (1 - p) * f - r * W1) / (1 - p * (1 - P))
    return p * F0 + (1 - p) * f

# ---------------------------------------------------------------------------
# Case 2 verification tables
# ---------------------------------------------------------------------------

print("\n=== Case 2 ===")
k = 2

print(f"\nW(1), T(1), C(1) -- p+gamma joins at d=0")
print(f"{'Qty':<6} {'p':>5} {'gamma':>7} {'sim':>10} {'exact':>10} {'diff':>8}")
print("-" * 50)
for p in [0.3, 0.4]:
    for gamma in [0.0, 0.2, 0.4]:
        sw = simulate_case2_W(p, k, gamma)
        st = simulate_case2_T(p, k, gamma)
        sc = simulate_case2_C(p, k, gamma)
        print(f"{'W':<6} {p:>5.1f} {gamma:>7.1f} {sw:>10.4f} {W1_case2_exact(p,k,gamma):>10.4f} {sw-W1_case2_exact(p,k,gamma):>+8.4f}")
        print(f"{'T':<6} {p:>5.1f} {gamma:>7.1f} {st:>10.4f} {T1_case2_exact(p,k,gamma):>10.4f} {st-T1_case2_exact(p,k,gamma):>+8.4f}")
        print(f"{'C':<6} {p:>5.1f} {gamma:>7.1f} {sc:>10.4f} {C1_case2_exact(p,gamma):>10.4f} {sc-C1_case2_exact(p,gamma):>+8.4f}")

print(f"\nG(1) -- expected gain vs turning off")
print(f"{'p':>5} {'gamma':>7} {'q':>5} {'sim':>10} {'exact':>10} {'diff':>8}")
print("-" * 55)
for q in [0.1, 0.3]:
    for p in [0.3, 0.4]:
        for gamma in [0.0, 0.2, 0.4]:
            sg = simulate_case2_G(p, k, gamma, q)
            eg = G1_case2_exact(p, gamma, q)
            print(f"{p:>5.1f} {gamma:>7.1f} {q:>5.1f} {sg:>10.4f} {eg:>10.4f} {sg-eg:>+8.4f}")

r = 0.1
print(f"\nF(1) -- attacker gain vs mining compliant (r={r})")
print(f"{'p':>5} {'gamma':>7} {'f':>5} {'sim':>10} {'exact':>10} {'diff':>8}")
print("-" * 58)
for p in [0.3, 0.4]:
    for gamma in [0.0, 0.2, 0.4]:
        for f in [0.0, 0.2]:
            sf = simulate_case2_F(p, k, gamma, f, r)
            ef = F1_case2_exact(p, gamma, f, r)
            print(f"{p:>5.1f} {gamma:>7.1f} {f:>5.1f} {sf:>10.4f} {ef:>10.4f} {sf-ef:>+8.4f}")

# ---------------------------------------------------------------------------
# Section 3: k=infinity, p > 0.5 (reorg succeeds with probability 1)
# ---------------------------------------------------------------------------

def simulate_T_kinf(p, start_d, n_trials=50000, seed=42):
    """Expected race duration until d=-1, with no upper absorbing barrier."""
    rng = random.Random(seed)
    total_steps = 0
    for _ in range(n_trials):
        d, steps = start_d, 0
        while d != -1:
            d -= 1 if rng.random() < p else -1
            steps += 1
        total_steps += steps
    return total_steps / n_trials

def T_kinf_exact(p, d):
    # T(d) = (d+1)/(2p-1): from BVP with finiteness condition forcing c1=0,
    # or directly from Wald's identity since d_T = -1 almost surely.
    return (d + 1) / (2*p - 1)

print("\n=== Section 3: k=infinity, compliant chain wins with p > 0.5 ===")
print("\nT(d) -- expected steps to reach d=-1")
print(f"{'p':>5} {'d':>4} {'sim':>10} {'exact':>10} {'diff':>8}")
print("-" * 42)
for p in [0.6, 0.7, 0.8, 0.9]:
    for d in [1, 3, 5]:
        st = simulate_T_kinf(p, d)
        et = T_kinf_exact(p, d)
        print(f"{p:>5.1f} {d:>4} {st:>10.3f} {et:>10.3f} {st-et:>+8.3f}")

def V_lazy(p, d, h, q, a, b):
    """Expected excess income (above usual mining revenue) for miner using q-fraction withholding.
    Lazy BVP gives T(d,q) = (d+1)/(2p-1-hq).
    Excess income per period: a*h (participation bonus) + h^2*(1-q)*b (finder's fees)."""
    denom = 2*p - 1 - h*q
    if denom <= 0:
        return float('inf')
    return h * (a + h*(1-q)*b) * (d+1) / denom

def h_star(p, a, b):
    """Threshold: miners with h < h_star prefer reveal (q=0); h > h_star prefer full withhold (q=1).
    Set dV/dq = 0: a + h*b = b*(2p-1) => h* = (2p-1) - a/b."""
    return (2*p - 1) - a / b

def b_min(p, h_max, a):
    """Minimum finder's fee to deter withholding for all miners with hashrate <= h_max.
    Condition h_max <= h_star rearranges to b >= a/(2p-1-h_max)."""
    denom = 2*p - 1 - h_max
    if denom <= 0:
        return float('inf')
    return a / denom

def simulate_withholding(p, d, h, q, a, b, n_trials=40000, seed=42):
    """Monte Carlo estimate of V_lazy: miner reveals each found block with prob 1-q."""
    rng = random.Random(seed)
    total = 0.0
    for _ in range(n_trials):
        state, income = d, 0.0
        while state != -1:
            u = rng.random()
            income += a * h                 # excess participation bonus per period
            if u < h:
                if rng.random() >= q:       # reveal with prob 1-q
                    state -= 1
                    income += h * b         # finder's fee h*b per revealed block
                # else: withhold, state stays
            elif u < p:
                state -= 1                  # another compliant miner advances
            else:
                state += 1                  # non-compliant miner advances
        total += income
    return total / n_trials

# Demonstrate that V(q) is monotone: optimal q is always 0 or 1, never interior.
# Use p=0.8, a=0.1, b=0.2 so that h_star = 0.6 - 0.1/0.2 = 0.6 - 0.5 = 0.1 > 0.
p_ex, d_ex, a_ex, b_ex = 0.8, 1, 0.1, 0.2
hs = h_star(p_ex, a_ex, b_ex)
print(f"\n=== Mixed withholding: p={p_ex}, a={a_ex}, b={b_ex} ===")
print(f"h_star = {hs:.4f}  (all q give same V when h = h_star)")
print(f"\nV(d=1) vs q  [h < h_star: V decreasing (reveal best); h > h_star: V increasing (withhold best)]")
print(f"{'q':>6} {'h=0.05':>10} {'h=0.10(h*)':>12} {'h=0.15':>10} {'h=0.20':>10}")
print("-" * 50)
test_hs = [0.05, hs, 0.15, 0.20]
for q in [0.0, 0.25, 0.50, 0.75, 1.0]:
    vals = [V_lazy(p_ex, d_ex, h, q, a_ex, b_ex) for h in test_hs]
    print(f"{q:>6.2f} " + " ".join(f"{v:>10.4f}" for v in vals))

print("\nSimulation spot-check (exact = V_lazy formula)")
print(f"{'q':>6} {'h':>6} {'sim':>10} {'exact':>10} {'diff':>8}")
print("-" * 45)
for q, h in [(0.0, 0.05), (1.0, 0.05), (0.0, 0.15), (1.0, 0.15), (0.5, hs)]:
    sv = simulate_withholding(p_ex, d_ex, h, q, a_ex, b_ex)
    ev = V_lazy(p_ex, d_ex, h, q, a_ex, b_ex)
    print(f"{q:>6.2f} {h:>6.2f} {sv:>10.4f} {ev:>10.4f} {sv-ev:>+8.4f}")

# Minimum finder's fee: b >= a/(2p-1-h_max)
print(f"\nCorrected b_min = a/(2p-1-h_max) to deter withholding for all h <= h_max (a=0.1)")
print(f"{'p':>5} {'h_max=0.05':>12} {'h_max=0.10':>12} {'h_max=0.15':>12}")
print("-" * 45)
a_bmin = 0.1
for p in [0.6, 0.7, 0.8, 0.9]:
    row = [f"{b_min(p, hm, a_bmin):>12.3f}" if b_min(p, hm, a_bmin) < 1e8 else f"{'N/A':>12}" for hm in [0.05, 0.10, 0.15]]
    print(f"{p:>5.1f} {''.join(row)}")
