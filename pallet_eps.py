"""
Circular pallet recovery network (GOC / POC) - bi-objective, epsilon-constraint
GAMSPy implementation, solved with CPLEX.

   Z1 = total cost                (min)
   Z2 = total CO2 emissions, kg   (min)

Solution method: augmented epsilon-constraint (AUGMECON, Mavrotas 2009).
   1. Payoff table by lexicographic optimisation:
        min Z1, then min Z2 with Z1 held at its optimum
        min Z2, then min Z1 with Z2 held at its optimum
   2. eps_k = Z2min + (k-1)/(N_GRID-1) * rangeZ2,   k = 1..N_GRID
   3. For each eps_k:  min Z1 - rho*(rangeZ1/rangeZ2)*s
                       s.t. Z2 + s = eps_k,  s >= 0,  all model constraints
The Pareto points are the (Z1, Z2) values actually reached, not (eps, Z1).
"""

import matplotlib.pyplot as plt
import pandas as pd
from gamspy import (
    Alias, Container, Domain, Equation, Model, ModelStatus, Options,
    Parameter, Problem, Sense, Set, Sum, Variable,
)
from gamspy.math import uniformInt

SOLVER = "CPLEX"
N_GRID = 10          # number of epsilon grid points
RHO    = 1e-3        # AUGMECON augmentation weight
OPTS   = Options(relative_optimality_gap=0, time_limit=3600)

# =============================================================================
# 1. DATA  (illustrative instance)
# =============================================================================
GOCS     = ["r1", "r2"]                    # R  : general operation centres
POCS     = ["k1"]                          # K  : pivotal operation centres
CENTRES  = GOCS + POCS                     # J = R u K
DISPATCH = ["i1", "i2", "i3"]              # I^s: dispatch points
RETURNS  = ["l1", "l2", "l3"]              # I^c: return points

T_MAX  = 10                                # t_max
N_MAX  = 3                                 # n_max: retirement cohort
PERIODS = [f"t{x}" for x in range(1, T_MAX + 1)]
COHORTS = [f"n{x}" for x in range(N_MAX + 1)]

# ---- scalar parameters ----------------------------------------------------------
c     = 12       # c    : purchase cost of a new pallet
h     = 0.5      # h    : holding cost per pallet and period
cD    = 3        # c^D  : disposal cost per scrapped pallet
sal   = 1.5      # s    : salvage value per scrapped pallet
cKm   = 0.02     #        transport cost per pallet-km
v0    = 0.20     # v_0  : repairable share at cohort 0
e0    = 0.05     # e_0  : scrap share at cohort 0
dr    = 0.15     # dr   : degradation rate per cycle
gS    = 100      # g^s  : forward assignment cost
gC    = 100      # g^c  : reverse assignment cost
RRbar = 0.50     # RR_t : reuse floor
BIGM  = 100000   # M    : constant big-M

# ---- emission factors (illustrative) --------------------------------------------
emA = 25         # kg CO2 per new pallet produced
emM = 2          # kg CO2 per repaired pallet
emD = 1          # kg CO2 per scrapped pallet
eKm = 0.006      # kg CO2 per pallet-km transported

# ---- indexed parameters ---------------------------------------------------------
fixed  = {"r1": 20000, "r2": 22000, "k1": 45000}   # f_j
repair = {"r1": 10,    "r2": 10,    "k1": 6}       # m_j
WR     = {"r1": 60,    "r2": 60}                   # W_r
cap    = {"r1": 3000,  "r2": 3000,  "k1": 6000}    # Cap_j
RF     = {"r1": 0.30,  "r2": 0.30,  "k1": 0.30}    # RF_j
LT     = {"i1": 1,     "i2": 2,     "i3": 1}       # LT_i

rr = {("i1", "l1"): .50, ("i1", "l2"): .30, ("i1", "l3"): .10,
      ("i2", "l1"): .20, ("i2", "l2"): .50, ("i2", "l3"): .20,
      ("i3", "l1"): .10, ("i3", "l2"): .30, ("i3", "l3"): .50}

dS = {("r1", "i1"): 50,  ("r1", "i2"): 90,  ("r1", "i3"): 150,
      ("r2", "i1"): 140, ("r2", "i2"): 80,  ("r2", "i3"): 60,
      ("k1", "i1"): 320, ("k1", "i2"): 300, ("k1", "i3"): 310}
dC = {("l1", "r1"): 45,  ("l1", "r2"): 130, ("l1", "k1"): 330,
      ("l2", "r1"): 85,  ("l2", "r2"): 75,  ("l2", "k1"): 300,
      ("l3", "r1"): 145, ("l3", "r2"): 55,  ("l3", "k1"): 310}
dT = {("r1", "k1"): 260, ("r2", "k1"): 240}


def shares(nn):
    """Cohort law u(n), v(n), e(n) with forced retirement at n_max."""
    if nn == N_MAX:
        return 0.0, 0.0, 1.0
    q = (1 - dr) ** nn
    return (1 - v0 - e0) * q, v0 * q, 1 - (1 - e0) * q


u = {f"n{x}": shares(x)[0] for x in range(N_MAX + 1)}
v = {f"n{x}": shares(x)[1] for x in range(N_MAX + 1)}
e = {f"n{x}": shares(x)[2] for x in range(N_MAX + 1)}

# (i, t, t - LT_i): dispatch period of pallets returning in t
lag = [(i, PERIODS[s], PERIODS[s - LT[i]])
       for i in DISPATCH for s in range(T_MAX) if s - LT[i] >= 0]


def rec(d, scale=1.0):
    """dict -> GAMSPy records; keys may be tuples."""
    return [(*k, val * scale) if isinstance(k, tuple) else (k, val * scale)
            for k, val in d.items()]


# =============================================================================
# 2. SETS AND PARAMETERS
# =============================================================================
m = Container()

j  = Set(m, "j", records=CENTRES,  description="operation centres")
r  = Set(m, "r", domain=j, records=GOCS, description="GOCs")
k  = Set(m, "k", domain=j, records=POCS, description="POCs")
i  = Set(m, "i", records=DISPATCH, description="dispatch points")
l  = Set(m, "l", records=RETURNS,  description="return points")
t  = Set(m, "t", records=PERIODS,  description="periods 1..t_max")
n  = Set(m, "n", records=COHORTS,  description="cohorts 0..n_max")
tt = Alias(m, "tt", t)

tR   = Set(m, "tR", domain=t, records=PERIODS[max(LT.values()):],
           description="periods with circularity constraints")
nd   = Set(m, "nd", domain=n, records=COHORTS[:-1], description="dispatchable cohorts")
nr   = Set(m, "nr", domain=n, records=COHORTS[1:-1], description="repairable cohorts")
np_  = Set(m, "np", domain=n, records=COHORTS[1:], description="collected cohorts")
tlag = Set(m, "tlag", domain=[i, t, tt], records=lag, description="tt = t - LT_i")

f_   = Parameter(m, "f",   domain=j, records=rec(fixed),  description="fixed opening cost")
mrep = Parameter(m, "m",   domain=j, records=rec(repair), description="unit repair cost")
WR_  = Parameter(m, "WR",  domain=r, records=rec(WR),     description="GOC repair capacity")
Cap  = Parameter(m, "Cap", domain=j, records=rec(cap),    description="storage capacity")
RF_  = Parameter(m, "RF",  domain=j, records=rec(RF),     description="recovery floor")
rr_  = Parameter(m, "rr",  domain=[i, l], records=rec(rr), description="return shares")
u_   = Parameter(m, "u",   domain=n, records=rec(u), description="reusable share")
v_   = Parameter(m, "v",   domain=n, records=rec(v), description="repairable share")
e_   = Parameter(m, "e",   domain=n, records=rec(e), description="scrap share")
tcS  = Parameter(m, "tcS", domain=[j, i], records=rec(dS, cKm), description="unit dispatch cost")
tcC  = Parameter(m, "tcC", domain=[l, j], records=rec(dC, cKm), description="unit collection cost")
tcT  = Parameter(m, "tcT", domain=[r, k], records=rec(dT, cKm), description="unit transshipment cost")
emS  = Parameter(m, "emS", domain=[j, i], records=rec(dS, eKm), description="kg CO2 per dispatched pallet")
emC  = Parameter(m, "emC", domain=[l, j], records=rec(dC, eKm), description="kg CO2 per collected pallet")
emT  = Parameter(m, "emT", domain=[r, k], records=rec(dT, eKm), description="kg CO2 per transshipped pallet")
cEol = cD - sal

# demand D(i,t) = uniformInt(250,350), same random stream as the GAMS code
D_ = Parameter(m, "D", domain=[i, t], description="demand")
D_[i, t] = uniformInt(250, 350)

# epsilon-constraint data (changed between solves)
epsLev = Parameter(m, "epsLev", description="current epsilon level on Z2")
rngZ1  = Parameter(m, "rngZ1", records=1, description="range of Z1 on the front")
rngZ2  = Parameter(m, "rngZ2", records=1, description="range of Z2 on the front")

# =============================================================================
# 3. VARIABLES
# =============================================================================
X   = Variable(m, "X",   type="binary",   domain=j, description="open centre j")
Y   = Variable(m, "Y",   type="binary",   domain=[j, i], description="i served by j")
Z   = Variable(m, "Z",   type="binary",   domain=[l, j], description="l collected by j")
A   = Variable(m, "A",   type="positive", domain=[j, t], description="new pallets bought")
Fs  = Variable(m, "Fs",  type="positive", domain=[j, i, t, n], description="dispatched j -> i")
Fc  = Variable(m, "Fc",  type="positive", domain=[l, j, t, n], description="collected l -> j")
G   = Variable(m, "G",   type="positive", domain=[r, k, t, n], description="transshipped r -> k")
Inv = Variable(m, "Inv", type="positive", domain=[j, t, n], description="inventory")
S   = Variable(m, "S",   type="positive", domain=[j, t], description="pallets scrapped")
Slk = Variable(m, "Slk", type="positive", description="slack of the epsilon-constraint")
Z1     = Variable(m, "Z1",     description="total cost")
Z2     = Variable(m, "Z2",     description="total CO2 emissions (kg)")
CostT  = Variable(m, "CostT",  domain=t, description="variable cost in period t")
EmisT  = Variable(m, "EmisT",  domain=t, description="emissions in period t")
AugObj = Variable(m, "AugObj", description="augmented objective")

# =============================================================================
# 4. EQUATIONS
# =============================================================================
# ---- objective 1: cost ----------------------------------------------------------
eZ1 = Equation(m, "eZ1", description="total cost")
eZ1[...] = Z1 == (Sum(j, f_[j] * X[j]) + Sum((j, i), gS * Y[j, i])
                  + Sum((l, j), gC * Z[l, j]) + Sum(t, CostT[t]))

eCostT = Equation(m, "eCostT", domain=t, description="variable cost of period t")
eCostT[t] = CostT[t] == (
      Sum(j, c * A[j, t])
    + Sum((j, nd), h * Inv[j, t, nd])
    + Sum((j, i, nd), tcS[j, i] * Fs[j, i, t, nd])
    + Sum((l, j, np_), tcC[l, j] * Fc[l, j, t, np_])
    + Sum((j, l, np_), mrep[j] * v_[np_] * Fc[l, j, t, np_])
    + Sum((r, k, nr), (tcT[r, k] + mrep[k] - mrep[r]) * G[r, k, t, nr])
    + Sum(j, cEol * S[j, t]))

# ---- objective 2: CO2 emissions -------------------------------------------------
eZ2 = Equation(m, "eZ2", description="total emissions")
eZ2[...] = Z2 == Sum(t, EmisT[t])

eEmisT = Equation(m, "eEmisT", domain=t, description="emissions of period t")
eEmisT[t] = EmisT[t] == (
      Sum(j, emA * A[j, t])
    + Sum((j, i, nd), emS[j, i] * Fs[j, i, t, nd])
    + Sum((l, j, np_), emC[l, j] * Fc[l, j, t, np_])
    + Sum((r, k, nr), emT[r, k] * G[r, k, t, nr])
    + Sum((j, l, np_), emM * v_[np_] * Fc[l, j, t, np_])
    + Sum(j, emD * S[j, t]))

# ---- assignment -----------------------------------------------------------------
eAsgS = Equation(m, "eAsgS", domain=i, description="one centre per dispatch point")
eAsgS[i] = Sum(j, Y[j, i]) == 1
eAsgC = Equation(m, "eAsgC", domain=l, description="one centre per return point")
eAsgC[l] = Sum(j, Z[l, j]) == 1
eOpenS = Equation(m, "eOpenS", domain=[j, i], description="dispatch only from open centre")
eOpenS[j, i] = Y[j, i] <= X[j]
eOpenC = Equation(m, "eOpenC", domain=[l, j], description="collection only by open centre")
eOpenC[l, j] = Z[l, j] <= X[j]

# ---- demand ---------------------------------------------------------------------
eDemand = Equation(m, "eDemand", domain=[i, t], description="demand satisfaction")
eDemand[i, t] = Sum((j, nd), Fs[j, i, t, nd]) == D_[i, t]
eDispLink = Equation(m, "eDispLink", domain=[j, i, t], description="dispatch on assigned arcs")
eDispLink[j, i, t] = Sum(nd, Fs[j, i, t, nd]) <= D_[i, t] * Y[j, i]

# ---- returns: collected = what actually comes back ------------------------------
eReturn = Equation(m, "eReturn", domain=[l, t, n], description="return balance by cohort")
eReturn[l, t, n].where[np_[n]] = Sum(j, Fc[l, j, t, n]) == Sum(
    Domain(i, tt, j).where[tlag[i, t, tt]], rr_[i, l] * Fs[j, i, tt, n.lag(1)])

# assignment link only (constant big-M)
eCollLink = Equation(m, "eCollLink", domain=[l, j, t], description="collection on assigned arcs")
eCollLink[l, j, t] = Sum(np_, Fc[l, j, t, np_]) <= BIGM * Z[l, j]

# ---- inventory (empty initial stock) --------------------------------------------
eInv0 = Equation(m, "eInv0", domain=[j, t], description="inventory of new pallets")
eInv0[j, t] = Inv[j, t, "n0"] == Inv[j, t.lag(1), "n0"] + A[j, t] - Sum(i, Fs[j, i, t, "n0"])

eInvGOC = Equation(m, "eInvGOC", domain=[r, t, n], description="GOC inventory of used pallets")
eInvGOC[r, t, nr] = Inv[r, t, nr] == (
      Inv[r, t.lag(1), nr]
    + u_[nr] * Sum(l, Fc[l, r, t, nr])
    + v_[nr] * Sum(l, Fc[l, r, t.lag(1), nr])
    - Sum(k, G[r, k, t.lag(1), nr])
    - Sum(i, Fs[r, i, t, nr]))

eInvPOC = Equation(m, "eInvPOC", domain=[k, t, n], description="POC inventory of used pallets")
eInvPOC[k, t, nr] = Inv[k, t, nr] == (
      Inv[k, t.lag(1), nr]
    + u_[nr] * Sum(l, Fc[l, k, t, nr])
    + v_[nr] * Sum(l, Fc[l, k, t.lag(1), nr])
    + Sum(r, G[r, k, t.lag(1), nr])
    - Sum(i, Fs[k, i, t, nr]))

# ---- repair ---------------------------------------------------------------------
eTrans = Equation(m, "eTrans", domain=[r, t, n], description="transship only repairables")
eTrans[r, t, nr] = Sum(k, G[r, k, t, nr]) <= v_[nr] * Sum(l, Fc[l, r, t, nr])

eRepGOC = Equation(m, "eRepGOC", domain=[r, t], description="GOC repair capacity")
eRepGOC[r, t] = (Sum((l, nr), v_[nr] * Fc[l, r, t, nr])
                 - Sum((k, nr), G[r, k, t, nr])) <= WR_[r] * X[r]

eLinkPOC = Equation(m, "eLinkPOC", domain=[k, t], description="repair at POC only if open")
eLinkPOC[k, t] = (Sum((l, nr), v_[nr] * Fc[l, k, t, nr])
                  + Sum((r, nr), G[r, k, t, nr])) <= BIGM * X[k]

# ---- storage and scrap ----------------------------------------------------------
eStore = Equation(m, "eStore", domain=[j, t], description="storage capacity")
eStore[j, t] = Sum(nd, Inv[j, t, nd]) <= Cap[j] * X[j]
eScrap = Equation(m, "eScrap", domain=[j, t], description="scrapped pallets")
eScrap[j, t] = S[j, t] == Sum((l, np_), e_[np_] * Fc[l, j, t, np_])

# ---- circularity ----------------------------------------------------------------
eReuse = Equation(m, "eReuse", domain=t, description="reuse floor")
eReuse[tR] = Sum((j, i, nr), Fs[j, i, tR, nr]) >= RRbar * Sum(i, D_[i, tR])
eRecover = Equation(m, "eRecover", domain=[j, t], description="recovery floor")
eRecover[j, tR] = (Sum((l, np_), (1 - e_[np_]) * Fc[l, j, tR, np_])
                   >= RF_[j] * Sum((i, nd), Fs[j, i, tR, nd]))

# ---- augmented epsilon-constraint -----------------------------------------------
eEps = Equation(m, "eEps", description="epsilon-constraint on Z2")
eEps[...] = Z2 + Slk == epsLev
eAug = Equation(m, "eAug", description="augmented objective")
eAug[...] = AugObj == Z1 - RHO * (rngZ1 / rngZ2) * Slk

# =============================================================================
# 5. MODELS
# =============================================================================
base = [eZ1, eCostT, eZ2, eEmisT, eAsgS, eAsgC, eOpenS, eOpenC, eDemand, eDispLink,
        eReturn, eCollLink, eInv0, eInvGOC, eInvPOC, eTrans, eRepGOC, eLinkPOC,
        eStore, eScrap, eReuse, eRecover]

minZ1 = Model(m, "minZ1", equations=base, problem=Problem.MIP, sense=Sense.MIN, objective=Z1)
minZ2 = Model(m, "minZ2", equations=base, problem=Problem.MIP, sense=Sense.MIN, objective=Z2)
epsModel = Model(m, "palletEps", equations=base + [eEps, eAug], problem=Problem.MIP,
                 sense=Sense.MIN, objective=AugObj)


def solve(model, label):
    model.solve(solver=SOLVER, options=OPTS)
    if model.status != ModelStatus.OptimalGlobal:
        raise RuntimeError(f"{label}: not optimal ({model.status})")


# =============================================================================
# 6. STEP 1 - PAYOFF TABLE (lexicographic)
# =============================================================================
solve(minZ1, "min Z1")
Z1min = Z1.toValue()
Z1.up[...] = Z1min * (1 + 1e-6)          # hold Z1 at its optimum
solve(minZ2, "lex min Z2 | Z1")
Z2atZ1min = Z2.toValue()
Z1.up[...] = float("inf")                # release

solve(minZ2, "min Z2")
Z2min = Z2.toValue()
Z2.up[...] = Z2min * (1 + 1e-6)          # hold Z2 at its optimum
solve(minZ1, "lex min Z1 | Z2")
Z1atZ2min = Z1.toValue()
Z2.up[...] = float("inf")                # release

rangeZ1 = Z1atZ2min - Z1min
rangeZ2 = Z2atZ1min - Z2min
rngZ1.setRecords(rangeZ1)
rngZ2.setRecords(rangeZ2)

payoff = pd.DataFrame({"Z1": [Z1min, Z1atZ2min], "Z2": [Z2atZ1min, Z2min]},
                      index=["min Z1", "min Z2"])
print("\nPayoff table\n", payoff.round(1))

# =============================================================================
# 7. STEPS 2-3 - EPSILON GRID
# =============================================================================
rows = []
for kp in range(1, N_GRID + 1):
    eps = Z2min + (kp - 1) / (N_GRID - 1) * rangeZ2
    epsLev.setRecords(eps)
    solve(epsModel, f"grid point {kp}")
    x = X.records.set_index("j")["level"].round().astype(int)
    rows.append({
        "point":     f"kp{kp}",
        "eps":       eps,
        "Z1":        Z1.toValue(),
        "Z2":        Z2.toValue(),
        "slack":     Slk.toValue(),
        "design":    "+".join(x[x > 0].index),
        "new":       A.records["level"].sum(),
        "transship": G.records["level"].sum() if G.records is not None else 0.0,
    })

front = pd.DataFrame(rows)
front.to_csv("pareto_front.csv", index=False)
print("\nAll grid points\n", front.round(2).to_string(index=False))

# distinct Pareto solutions = unique (Z1, Z2) pairs actually reached
pareto = (front.round({"Z1": 1, "Z2": 1})
               .drop_duplicates(subset=["Z1", "Z2"])
               .sort_values("Z1").reset_index(drop=True))
pareto.insert(0, "P", [f"P{x + 1}" for x in range(len(pareto))])
print("\nPareto solutions\n", pareto[["P", "Z1", "Z2", "design"]].to_string(index=False))

# =============================================================================
# 8. PLOT THE PARETO FRONT
# =============================================================================
INK, INK2, GRID, BLUE = "#0b0b0b", "#52514e", "#e4e3df", "#2a78d6"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": INK2, "axes.labelcolor": INK,
                     "xtick.color": INK2, "ytick.color": INK2})

fig, ax = plt.subplots(figsize=(7.2, 4.8), dpi=200)

# epsilon levels as faint horizontal lines (the control knob, not Pareto points)
for eps in front["eps"]:
    ax.axhline(eps / 1000, color=GRID, lw=0.8, zorder=0)

x_pf, y_pf = pareto["Z1"] / 1000, pareto["Z2"] / 1000
ax.plot(x_pf, y_pf, color=BLUE, lw=1, ls=(0, (3, 3)), alpha=0.6, zorder=1)
ax.plot(x_pf, y_pf, ls="none", marker="o", ms=8, mfc=BLUE, mec="white", mew=2, zorder=3)

# labels: right of the point, left in the right-most third of the axis;
# a point within 5% of the cost range of its neighbour takes the other side
span = x_pf.max() - x_pf.min()
prev_left = True
for idx, row in pareto.iterrows():
    left = x_pf[idx] > x_pf.min() + 0.66 * span
    if idx > 0 and x_pf[idx] - x_pf[idx - 1] < 0.05 * span:
        left = not prev_left
    ax.annotate(f"{row.P}  {row.design}\n({row.Z1:,.0f}; {row.Z2:,.0f})",
                (x_pf[idx], y_pf[idx]), xytext=(-10, -18) if left else (10, 2),
                textcoords="offset points", ha="right" if left else "left",
                fontsize=8, color=INK)
    prev_left = left

pad_x = 0.08 * (x_pf.max() - x_pf.min())
pad_y = 0.08 * (y_pf.max() - y_pf.min())
ax.set_xlim(x_pf.min() - pad_x, x_pf.max() + pad_x)
ax.set_ylim(y_pf.min() - 1.5 * pad_y, y_pf.max() + pad_y)
ax.text(ax.get_xlim()[1], front["eps"].iloc[N_GRID // 2] / 1000 + 0.1,
        f"grey lines: {N_GRID} ε levels on Z2", color=INK2, fontsize=8, ha="right")
ax.set_xlabel("Z1  total cost (thousand)")
ax.set_ylabel("Z2  CO$_2$ emissions (tonnes)")
ax.set_title(f"Pareto front — augmented ε-constraint, {N_GRID} grid points",
             loc="left", fontsize=11, color=INK)
for side in ["top", "right"]:
    ax.spines[side].set_visible(False)

fig.tight_layout()
fig.savefig("pareto_front_py.png")
fig.savefig("pareto_front_py.pdf")
# plt.show()   # uncomment to open the figure window in VS Code
