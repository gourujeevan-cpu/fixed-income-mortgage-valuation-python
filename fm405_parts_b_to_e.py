"""

Candidate Numbers: 67125 and 70216

FM405 Summative Work - Parts (b)-(e) [Fixed Version]
====================================================
Uses Part (a) calibration, then prices the mortgage, MBS strips,
and Monte Carlo checks with a consistent prepayment convention.

Key convention used here:
    - No prepayment is allowed at origination (t = 0).
    - From t = 0.5 onward, at each node the borrower may either
      continue or immediately prepay the outstanding balance.
    - If prepayment occurs at node i, valuation stops immediately
      at L[i]; there are no further cash flows from that node.
    - In Part (e), Monte Carlo mortgage valuation uses simulated
      BDT paths together with the optimal prepayment boundary
      computed from backward induction on the calibrated BDT tree.
"""

import numpy as np
from scipy.optimize import brentq
import matplotlib.pyplot as plt


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  SECTION 1: INPUTS                                                      ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

DATA_SOURCE = "Bank of England, UK Gilt Zero-Coupon Yield Curve"
DATA_DATE = "30 January 2026"
COMPOUNDING = "continuously_compounded"

YIELDS_PERCENT = [
    3.48, 3.55, 3.59, 3.63, 3.67, 3.72, 3.78, 3.84, 3.90, 3.97,
    4.04, 4.11, 4.18, 4.25, 4.31, 4.38, 4.45, 4.51, 4.57, 4.63
]

DELTA = 0.5
SIGMA_HL = 0.0185
SIGMA_BDT = 0.20

F0 = 100000
SPREAD_BP = 50
N_SIM = 100000


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  PART (a): CALIBRATION                                                  ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

def process_inputs(yields_pct, delta, compounding):
    y = np.array(yields_pct) / 100.0
    n = len(y)
    mat = np.arange(1, n + 1) * delta
    if compounding == "continuously_compounded":
        ycc = y
    elif compounding == "semi_annually_compounded":
        ycc = 2.0 * np.log(1.0 + y / 2.0)
    elif compounding == "annually_compounded":
        ycc = np.log(1.0 + y)
    else:
        raise ValueError(compounding)
    zcb = 100.0 * np.exp(-ycc * mat)
    return mat, ycc, zcb, ycc[0], n


def calibrate_ho_lee(r0, zcb, sigma, delta, n):
    sd = np.sqrt(delta)
    rt = [None] * n
    rt[0] = np.array([r0])
    th = np.zeros(n - 1)

    for i in range(n - 1):
        tgt = zcb[i + 1]

        def err(t, _i=i):
            nr = np.zeros(_i + 2)
            for j in range(_i + 1):
                nr[j] = rt[_i][j] + t * delta + sigma * sd
            nr[_i + 1] = rt[_i][_i] + t * delta - sigma * sd

            v = np.exp(-nr * delta) * 100.0
            for s in range(_i, -1, -1):
                vn = np.zeros(s + 1)
                for j in range(s + 1):
                    vn[j] = np.exp(-rt[s][j] * delta) * (0.5 * v[j] + 0.5 * v[j + 1])
                v = vn
            return v[0] - tgt

        th[i] = brentq(err, -5.0, 5.0, xtol=1e-14)

        nr = np.zeros(i + 2)
        for j in range(i + 1):
            nr[j] = rt[i][j] + th[i] * delta + sigma * sd
        nr[i + 1] = rt[i][i] + th[i] * delta - sigma * sd
        rt[i + 1] = nr

    return rt, th


def calibrate_bdt(r0, zcb, sigma, delta, n):
    sd = np.sqrt(delta)
    zt = [None] * n
    rt = [None] * n
    zt[0] = np.array([np.log(r0)])
    rt[0] = np.array([r0])
    th = np.zeros(n - 1)

    for i in range(n - 1):
        tgt = zcb[i + 1]

        def err(t, _i=i):
            nz = np.zeros(_i + 2)
            for j in range(_i + 1):
                nz[j] = zt[_i][j] + t * delta + sigma * sd
            nz[_i + 1] = zt[_i][_i] + t * delta - sigma * sd
            nr = np.exp(nz)

            v = np.exp(-nr * delta) * 100.0
            for s in range(_i, -1, -1):
                vn = np.zeros(s + 1)
                for j in range(s + 1):
                    vn[j] = np.exp(-rt[s][j] * delta) * (0.5 * v[j] + 0.5 * v[j + 1])
                v = vn
            return v[0] - tgt

        th[i] = brentq(err, -50.0, 50.0, xtol=1e-14)

        nz = np.zeros(i + 2)
        for j in range(i + 1):
            nz[j] = zt[i][j] + th[i] * delta + sigma * sd
        nz[i + 1] = zt[i][i] + th[i] * delta - sigma * sd
        zt[i + 1] = nz
        rt[i + 1] = np.exp(nz)

    return rt, zt, th


maturities, yields_cc, zcb_prices, r0, n_steps = process_inputs(
    YIELDS_PERCENT, DELTA, COMPOUNDING
)
T = n_steps * DELTA
FINAL_RATE_TIME = (n_steps - 1) * DELTA
hl_tree, hl_theta = calibrate_ho_lee(r0, zcb_prices, SIGMA_HL, DELTA, n_steps)
bdt_r_tree, bdt_z_tree, bdt_theta = calibrate_bdt(
    r0, zcb_prices, SIGMA_BDT, DELTA, n_steps
)


# ── Extend trees to i=20 (t=10.0) for display (lecture Table format) ──
# Mortgage valuation still uses n_pay=20 and columns i=0..19 only.
def extend_trees(hl_tree, hl_theta, bdt_r_tree, bdt_z_tree, bdt_theta,
                 yields_cc, sigma_hl, sigma_bdt, delta, n_steps):
    sqrt_dt = np.sqrt(delta)
    last = n_steps - 1
    slope = (yields_cc[-1] - yields_cc[-2]) / delta
    y_extra = yields_cc[-1] + slope * delta
    zcb_extra = 100.0 * np.exp(-y_extra * (n_steps + 1) * delta)

    def hl_err(th):
        nr = np.zeros(last + 2)
        for j in range(last + 1):
            nr[j] = hl_tree[last][j] + th * delta + sigma_hl * sqrt_dt
        nr[last + 1] = hl_tree[last][last] + th * delta - sigma_hl * sqrt_dt
        V = np.exp(-nr * delta) * 100.0
        for s in range(last, -1, -1):
            Vn = np.zeros(s + 1)
            for j in range(s + 1):
                Vn[j] = np.exp(-hl_tree[s][j] * delta) * (0.5 * V[j] + 0.5 * V[j + 1])
            V = Vn
        return V[0] - zcb_extra

    th_hl = brentq(hl_err, -5.0, 5.0, xtol=1e-14)
    hl_col = np.zeros(n_steps + 1)
    for j in range(n_steps):
        hl_col[j] = hl_tree[last][j] + th_hl * delta + sigma_hl * sqrt_dt
    hl_col[n_steps] = hl_tree[last][last] + th_hl * delta - sigma_hl * sqrt_dt

    def bdt_err(th):
        nz = np.zeros(last + 2)
        for j in range(last + 1):
            nz[j] = bdt_z_tree[last][j] + th * delta + sigma_bdt * sqrt_dt
        nz[last + 1] = bdt_z_tree[last][last] + th * delta - sigma_bdt * sqrt_dt
        nr = np.exp(nz)
        V = np.exp(-nr * delta) * 100.0
        for s in range(last, -1, -1):
            Vn = np.zeros(s + 1)
            for j in range(s + 1):
                Vn[j] = np.exp(-bdt_r_tree[s][j] * delta) * (0.5 * V[j] + 0.5 * V[j + 1])
            V = Vn
        return V[0] - zcb_extra

    th_bdt = brentq(bdt_err, -50.0, 50.0, xtol=1e-14)
    bdt_z_col = np.zeros(n_steps + 1)
    for j in range(n_steps):
        bdt_z_col[j] = bdt_z_tree[last][j] + th_bdt * delta + sigma_bdt * sqrt_dt
    bdt_z_col[n_steps] = bdt_z_tree[last][last] + th_bdt * delta - sigma_bdt * sqrt_dt

    hl_ext = list(hl_tree) + [hl_col]
    bdt_r_ext = list(bdt_r_tree) + [np.exp(bdt_z_col)]
    hl_th_ext = np.append(hl_theta, th_hl)
    bdt_th_ext = np.append(bdt_theta, th_bdt)
    return hl_ext, hl_th_ext, bdt_r_ext, bdt_th_ext

hl_tree_ext, hl_theta_ext, bdt_r_tree_ext, bdt_theta_ext = extend_trees(
    hl_tree, hl_theta, bdt_r_tree, bdt_z_tree, bdt_theta,
    yields_cc, SIGMA_HL, SIGMA_BDT, DELTA, n_steps
)
N_TREE = n_steps + 1  # 21 columns for display (i=0..20)


def print_rate_table(label, r_tree_ext, theta_ext, delta, n_cols):
    """
    Print the rate tree in lecture-table format:
        Row 1: Time T
        Row 2: Period i
        Row 3: θ_i × 100
        Rows j=0..n_cols-1: rates in %
    n_cols = 21 columns (i=0..20).
    """
    w = 8  # column width
    hdr = f"  {'':>{w}}"
    for i in range(n_cols):
        hdr += f"{i * delta:>{w}.1f}"
    print(f"\n  {label}")
    print(f"  {'=' * (w + n_cols * w)}")

    # Row 1 – Time T
    row = f"  {'T':>{w}}"
    for i in range(n_cols):
        row += f"{i * delta:>{w}.1f}"
    print(row)

    # Row 2 – Period i
    row = f"  {'i':>{w}}"
    for i in range(n_cols):
        row += f"{i:>{w}d}"
    print(row)

    # Row 3 – θ_i × 100 (θ_0 under column i=0, matching lecture table format)
    row = f"  {'θ×100':>{w}}"
    for k in range(n_cols):
        if k < len(theta_ext):
            row += f"{theta_ext[k] * 100:>{w}.4f}"
        else:
            row += f"{'':>{w}}"
    print(row)

    print(f"  {'-' * (w + n_cols * w)}")

    # Rate rows j=0..n_cols-1
    for j in range(n_cols):
        row = f"  {'j=' + str(j):>{w}}"
        for i in range(n_cols):
            if j < len(r_tree_ext[i]):
                row += f"{r_tree_ext[i][j] * 100:>{w}.4f}"
            else:
                row += f"{'':>{w}}"
        print(row)


print("\n" + "=" * 70)
print("  PART (a): CALIBRATED RATE TREES")
print("=" * 70)
print_rate_table("Ho-Lee Short Rate Tree (% per annum)",
                 hl_tree_ext, hl_theta_ext, DELTA, N_TREE)
print_rate_table("\n  BDT Short Rate Tree (% per annum)",
                 bdt_r_tree_ext, bdt_theta_ext, DELTA, N_TREE)


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  MORTGAGE HELPERS                                                       ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

def mortgage_coupon(F0, rm_sa, n_pay):
    r = rm_sa / 2.0
    if r < 1e-12:
        return F0 / n_pay
    return F0 * r * (1 + r) ** n_pay / ((1 + r) ** n_pay - 1)


def amort_schedule(F0, C, rm_sa, n_pay):
    r = rm_sa / 2.0
    L = np.zeros(n_pay + 1)
    L[0] = F0
    interest = np.zeros(n_pay)
    principal = np.zeros(n_pay)
    for i in range(n_pay):
        interest[i] = r * L[i]
        principal[i] = C - interest[i]
        L[i + 1] = L[i] - principal[i]
    return L, interest, principal


def build_vnp_tree(r_tree, delta, n_pay, C):
    vnp = [None] * (n_pay + 1)
    vnp[n_pay] = np.zeros(n_pay + 1)
    for i in range(n_pay - 1, -1, -1):
        vnp[i] = np.zeros(i + 1)
        for j in range(i + 1):
            vnp[i][j] = np.exp(-r_tree[i][j] * delta) * (
                0.5 * vnp[i + 1][j] + 0.5 * vnp[i + 1][j + 1] + C
            )
    return vnp


def build_mortgage_tree(r_tree, delta, n_pay, C, L):
    """
    Mortgage value tree with borrower prepayment option.

    Convention:
        - At t=0, prepayment is not allowed.
        - For i >= 1, borrower may either continue or immediately prepay L[i].
    """
    vm = [None] * (n_pay + 1)
    prepay = [None] * (n_pay + 1)
    vm[n_pay] = np.zeros(n_pay + 1)
    prepay[n_pay] = np.zeros(n_pay + 1, dtype=bool)

    for i in range(n_pay - 1, -1, -1):
        vm[i] = np.zeros(i + 1)
        prepay[i] = np.zeros(i + 1, dtype=bool)
        for j in range(i + 1):
            cont = np.exp(-r_tree[i][j] * delta) * (
                0.5 * vm[i + 1][j] + 0.5 * vm[i + 1][j + 1] + C
            )
            if i == 0:
                vm[i][j] = cont
            else:
                vm[i][j] = min(cont, L[i])
                prepay[i][j] = L[i] < cont
    return vm, prepay


def value_mortgage(r_tree, delta, n_pay, F0, rm_sa):
    C = mortgage_coupon(F0, rm_sa, n_pay)
    L, interest, principal = amort_schedule(F0, C, rm_sa, n_pay)
    vnp = build_vnp_tree(r_tree, delta, n_pay, C)
    vm, prepay = build_mortgage_tree(r_tree, delta, n_pay, C, L)
    option = vnp[0][0] - vm[0][0]
    return vm[0][0], vnp[0][0], option, C, L, interest, principal, vnp, vm, prepay


def find_par_rate(r_tree, delta, n_pay, F0):
    def err(rm):
        return value_mortgage(r_tree, delta, n_pay, F0, rm)[0] - F0

    return brentq(err, 0.01, 0.20, xtol=1e-10)


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  PART (b): PAR MORTGAGE RATE                                            ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

print("=" * 70)
print("  PART (b): PAR MORTGAGE RATE")
print("=" * 70)
print(f"  Principal={F0:,.0f}, T={T:.0f}y, {n_steps} payments")

rm_HL = find_par_rate(hl_tree, DELTA, n_steps, F0)
rm_BDT = find_par_rate(bdt_r_tree, DELTA, n_steps, F0)
res_HL = value_mortgage(hl_tree, DELTA, n_steps, F0, rm_HL)
res_BDT = value_mortgage(bdt_r_tree, DELTA, n_steps, F0, rm_BDT)

print(f"\n  {'':>30} {'Ho-Lee':>12} {'BDT':>12}")
print(f"  {'-' * 55}")
print(f"  {'Par mortgage rate (%)':>30} {rm_HL * 100:>12.4f} {rm_BDT * 100:>12.4f}")
print(f"  {'Semi-annual coupon C':>30} {res_HL[3]:>12,.2f} {res_BDT[3]:>12,.2f}")
print(f"  {'V^np (no prepay)':>30} {res_HL[1]:>12,.2f} {res_BDT[1]:>12,.2f}")
print(f"  {'Prepayment option':>30} {res_HL[2]:>12,.2f} {res_BDT[2]:>12,.2f}")
print(f"  {'Mortgage value V_0':>30} {res_HL[0]:>12,.2f} {res_BDT[0]:>12,.2f}")


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  PART (c): MBS CONSTRUCTION                                             ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

def value_mbs(r_tree, delta, n_pay, F0, rm_mort, rm_pt):
    """
    Pass-through convention:
        - Scheduled principal follows the mortgage amortization schedule.
        - Pass-through interest uses the pass-through coupon rate rm_pt.
        - If prepayment occurs at node i >= 1, the deal terminates immediately:
          PT = L[i], IO = 0, PO = L[i].
    """
    C_m = mortgage_coupon(F0, rm_mort, n_pay)
    L, _, principal_m = amort_schedule(F0, C_m, rm_mort, n_pay)
    interest_pt = (rm_pt / 2.0) * L[:-1]

    _, _, _, _, _, _, _, _, _, prepay = value_mortgage(r_tree, delta, n_pay, F0, rm_mort)

    PT = [None] * (n_pay + 1)
    IO = [None] * (n_pay + 1)
    PO = [None] * (n_pay + 1)
    PT[n_pay] = np.zeros(n_pay + 1)
    IO[n_pay] = np.zeros(n_pay + 1)
    PO[n_pay] = np.zeros(n_pay + 1)

    for i in range(n_pay - 1, -1, -1):
        PT[i] = np.zeros(i + 1)
        IO[i] = np.zeros(i + 1)
        PO[i] = np.zeros(i + 1)
        for j in range(i + 1):
            if i > 0 and prepay[i][j]:
                PT[i][j] = L[i]
                IO[i][j] = 0.0
                PO[i][j] = L[i]
            else:
                d = np.exp(-r_tree[i][j] * delta)
                PT[i][j] = d * (
                    0.5 * PT[i + 1][j] + 0.5 * PT[i + 1][j + 1]
                    + interest_pt[i] + principal_m[i]
                )
                IO[i][j] = d * (
                    0.5 * IO[i + 1][j] + 0.5 * IO[i + 1][j + 1] + interest_pt[i]
                )
                PO[i][j] = d * (
                    0.5 * PO[i + 1][j] + 0.5 * PO[i + 1][j + 1] + principal_m[i]
                )

    return PT[0][0], IO[0][0], PO[0][0]


print("\n\n" + "=" * 70)
print("  PART (c): MORTGAGE-BACKED SECURITIES")
print("=" * 70)
print(f"  MBS coupon = mortgage rate - {SPREAD_BP} bp")

sp = SPREAD_BP / 10000.0
rm_pt_HL = rm_HL - sp
rm_pt_BDT = rm_BDT - sp

PT_HL, IO_HL, PO_HL = value_mbs(hl_tree, DELTA, n_steps, F0, rm_HL, rm_pt_HL)
PT_BDT, IO_BDT, PO_BDT = value_mbs(bdt_r_tree, DELTA, n_steps, F0, rm_BDT, rm_pt_BDT)

print(f"\n  {'':>30} {'Ho-Lee':>12} {'BDT':>12}")
print(f"  {'-' * 55}")
print(f"  {'Mortgage rate (%)':>30} {rm_HL * 100:>12.4f} {rm_BDT * 100:>12.4f}")
print(f"  {'PT rate (%)':>30} {rm_pt_HL * 100:>12.4f} {rm_pt_BDT * 100:>12.4f}")
print(f"  {'Pass-Through':>30} {PT_HL:>12,.2f} {PT_BDT:>12,.2f}")
print(f"  {'IO strip':>30} {IO_HL:>12,.2f} {IO_BDT:>12,.2f}")
print(f"  {'PO strip':>30} {PO_HL:>12,.2f} {PO_BDT:>12,.2f}")
print(f"  {'IO + PO':>30} {IO_HL + PO_HL:>12,.2f} {IO_BDT + PO_BDT:>12,.2f}")


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  PART (d): MONTE CARLO RATE SIMULATION                                  ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

def simulate_rates(model, r0, theta, sigma, delta, n_steps, n_sim, seed=42):
    rng = np.random.RandomState(seed)
    sd = np.sqrt(delta)

    if model == "HL":
        r = np.zeros((n_sim, n_steps))
        r[:, 0] = r0
        for i in range(n_steps - 1):
            z = rng.choice([-1, 1], size=n_sim)
            r[:, i + 1] = r[:, i] + theta[i] * delta + sigma * sd * z
        return r

    if model == "BDT":
        x = np.zeros((n_sim, n_steps))
        x[:, 0] = np.log(r0)
        for i in range(n_steps - 1):
            z = rng.choice([-1, 1], size=n_sim)
            x[:, i + 1] = x[:, i] + theta[i] * delta + sigma * sd * z
        return np.exp(x)

    raise ValueError(model)


print("\n\n" + "=" * 70)
print(f"  PART (d): MC RATE SIMULATION (N={N_SIM:,})")
print("=" * 70)
print(f"  Note: the simulated terminal node is the short rate at t={FINAL_RATE_TIME:.1f},")
print(f"  i.e. the rate applying over the final half-year interval [{FINAL_RATE_TIME:.1f}, {T:.1f}].")

rates_HL = simulate_rates("HL", r0, hl_theta, SIGMA_HL, DELTA, n_steps, N_SIM)
rates_BDT = simulate_rates("BDT", r0, bdt_theta, SIGMA_BDT, DELTA, n_steps, N_SIM)
rT_HL = rates_HL[:, -1] * 100
rT_BDT = rates_BDT[:, -1] * 100

for lbl, rT in [("Ho-Lee", rT_HL), ("BDT", rT_BDT)]:
    print(f"\n  {lbl} at t={FINAL_RATE_TIME:.1f}: mean={np.mean(rT):.2f}%, std={np.std(rT):.2f}%")
    print(f"    [{np.min(rT):.2f}%, {np.max(rT):.2f}%]")
    if lbl == "Ho-Lee":
        print(f"    Neg: {np.sum(rT < 0)} ({100 * np.mean(rT < 0):.1f}%)")
    else:
        print("    Neg: 0 (always positive)")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].hist(rT_HL, bins=80, density=True, color="steelblue", alpha=0.7, edgecolor="white")
axes[0].axvline(0, color="red", ls="--", lw=1, label="r=0")
axes[0].set_xlabel("Rate (%)")
axes[0].set_ylabel("Density")
axes[0].set_title(f"Ho-Lee at t={FINAL_RATE_TIME:.1f}")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].hist(rT_BDT, bins=80, density=True, color="darkorange", alpha=0.7, edgecolor="white")
axes[1].set_xlabel("Rate (%)")
axes[1].set_ylabel("Density")
axes[1].set_title(f"BDT at t={FINAL_RATE_TIME:.1f}")
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("fig_d_mc_histograms_fixed.png", dpi=150, bbox_inches="tight")
plt.show()


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  PART (e): MC MORTGAGE VALUATION (BDT)                                  ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

def mc_mortgage_bdt(r0, theta, sigma, delta, n_pay, F0, rm_sa, r_tree, n_sim, seed=42):
    rng = np.random.RandomState(seed)
    sd = np.sqrt(delta)

    C = mortgage_coupon(F0, rm_sa, n_pay)
    L, _, _ = amort_schedule(F0, C, rm_sa, n_pay)
    _, _, _, _, _, _, _, _, _, prepay = value_mortgage(r_tree, delta, n_pay, F0, rm_sa)

    vals = np.zeros(n_sim)

    for s in range(n_sim):
        z = np.log(r0)
        j = 0
        disc = 1.0
        v = 0.0

        for i in range(n_pay):
            if i > 0 and prepay[i][j]:
                v += disc * L[i]
                break

            r_i = np.exp(z)
            disc *= np.exp(-r_i * delta)
            v += disc * C

            if i < n_pay - 1:
                down = rng.randint(0, 2)
                z = z + theta[i] * delta + sigma * sd * (1 - 2 * down)
                j += down

        vals[s] = v

    mn = np.mean(vals)
    se = np.std(vals, ddof=1) / np.sqrt(n_sim)

    # FM405 lecture-note convention: 95% CI ≈ estimate ± 2 × SE
    return mn, se, mn - 2.0 * se, mn + 2.0 * se, vals


print("\n\n" + "=" * 70)
print(f"  PART (e): MC MORTGAGE VALUATION - BDT (N={N_SIM:,})")
print("=" * 70)

mc_val, mc_se, mc_lo, mc_hi, mc_vals = mc_mortgage_bdt(
    r0, bdt_theta, SIGMA_BDT, DELTA, n_steps, F0, rm_BDT, bdt_r_tree, N_SIM
)

print(f"\n  BDT rate:   {rm_BDT * 100:.4f}%")
print(f"  Tree value: {F0:,.2f}")
print(f"  MC value:   {mc_val:,.2f} ± {mc_se:,.2f}")
print(f"  95% CI:     [{mc_lo:,.2f}, {mc_hi:,.2f}]")
print(f"  Consistent? {'✓ YES' if mc_lo <= F0 <= mc_hi else '✗ NO'}")

fig, ax = plt.subplots(figsize=(10, 5))
ax.hist(mc_vals, bins=100, density=True, color="darkorange", alpha=0.7, edgecolor="white")
ax.axvline(F0, color="red", ls="--", lw=2, label=f"Tree={F0:,.0f}")
ax.axvline(mc_val, color="blue", lw=2, label=f"MC={mc_val:,.0f}")
ax.axvspan(mc_lo, mc_hi, alpha=0.15, color="blue", label="95% CI")
ax.set_xlabel("Mortgage Value")
ax.set_ylabel("Density")
ax.set_title(f"MC Mortgage Value (BDT, N={N_SIM:,})")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("fig_e_mc_mortgage_fixed.png", dpi=150, bbox_inches="tight")
plt.show()


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  SUMMARY                                                                ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

print("\n\n" + "=" * 70)
print("  SUMMARY - PARTS (b)-(e)")
print("=" * 70)
print(f"\n  {'':>30} {'Ho-Lee':>12} {'BDT':>12}")
print(f"  {'-' * 55}")
print(f"  {'Par rate (%)':>30} {rm_HL * 100:>12.4f} {rm_BDT * 100:>12.4f}")
print(f"  {'Coupon C':>30} {res_HL[3]:>12,.2f} {res_BDT[3]:>12,.2f}")
print(f"  {'V^np':>30} {res_HL[1]:>12,.2f} {res_BDT[1]:>12,.2f}")
print(f"  {'Prepay option':>30} {res_HL[2]:>12,.2f} {res_BDT[2]:>12,.2f}")
print(f"  {'V_0':>30} {res_HL[0]:>12,.2f} {res_BDT[0]:>12,.2f}")
print(f"  {'PT value':>30} {PT_HL:>12,.2f} {PT_BDT:>12,.2f}")
print(f"  {'IO value':>30} {IO_HL:>12,.2f} {IO_BDT:>12,.2f}")
print(f"  {'PO value':>30} {PO_HL:>12,.2f} {PO_BDT:>12,.2f}")
print(f"  {'MC value (BDT)':>30} {'-':>12} {mc_val:>12,.2f}")
print(f"  {'MC 95% CI':>30} {'-':>12} [{mc_lo:,.0f}, {mc_hi:,.0f}]")
print("\n  DONE ✓")
