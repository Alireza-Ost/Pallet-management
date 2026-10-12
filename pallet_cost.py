"""
Circular pallet recovery network (GOC / POC) - single objective (minimum cost)
Deterministic model, GAMSPy implementation, solved with CPLEX.

   Z1 = total cost (min)
"""

import pandas as pd
from gamspy import (
    Alias, Container, Domain, Equation, Model, ModelStatus, Options,
    Parameter, Problem, Sense, Set, Sum, Variable,
)
from gamspy.math import uniformInt

SOLVER = "CPLEX"
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
cEol = cD - sal

# demand D(i,t) = uniformInt(250,350), same random stream as the GAMS code
D_ = Parameter(m, "D", domain=[i, t], description="demand")
D_[i, t] = uniformInt(250, 350)

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
Z1     = Variable(m, "Z1",     description="total cost")
CostT  = Variable(m, "CostT",  domain=t, description="variable cost in period t")

# =============================================================================
# 4. EQUATIONS
# =============================================================================
# ---- objective: total cost ------------------------------------------------------
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

# =============================================================================
# 5. MODEL AND SOLVE
# =============================================================================
pallet = Model(m, "pallet", equations=m.getEquations(), problem=Problem.MIP,
               sense=Sense.MIN, objective=Z1)
pallet.solve(solver=SOLVER, options=OPTS)
if pallet.status != ModelStatus.OptimalGlobal:
    raise RuntimeError(f"Model not solved to optimality: {pallet.status}")

# =============================================================================
# 6. REPORT
# =============================================================================
def level(var, by):
    """Sum of a variable's levels grouped by the given index columns."""
    df = var.records
    if df is None:
        return pd.Series(dtype=float)
    return df.groupby(by, observed=True)["level"].sum()


x = X.records.set_index("j")["level"].round().astype(int)
y = Y.records.query("level > 0.5")[["i", "j"]]
z = Z.records.query("level > 0.5")[["l", "j"]]

fixed_cost  = sum(fixed[c_] * x[c_] for c_ in CENTRES)
assign_cost = gS * len(y) + gC * len(z)

print(f"\nZ1 total cost        {Z1.toValue():>12,.1f}")
print(f"  opening cost       {fixed_cost:>12,.1f}")
print(f"  assignment cost    {assign_cost:>12,.1f}")
print(f"  variable cost      {CostT.records['level'].sum():>12,.1f}")
print(f"open centres         {', '.join(x[x > 0].index)}")
print("dispatch points      " + ", ".join(f"{a}->{b}" for b, a in y.values))
print("return points        " + ", ".join(f"{a}->{b}" for a, b in z.values))

# per-period summary
fs = Fs.records
used = fs[fs["n"].isin(COHORTS[1:-1])].groupby("t", observed=True)["level"].sum()
demand = D_.records.groupby("t", observed=True)["value"].sum()
per = pd.DataFrame({
    "demand":    demand,
    "new":       level(A, "t"),
    "reused":    used,
    "collected": level(Fc, "t"),
    "transship": level(G, "t"),
    "scrapped":  level(S, "t"),
    "cost":      CostT.records.set_index("t")["level"],
}).reindex(PERIODS).fillna(0.0)
per["reuse share"] = per["reused"] / per["demand"]
print("\nPer period\n", per.round(2).to_string())
print(f"\nnew pallets bought   {per['new'].sum():>12,.1f}")
print(f"pallets transshipped {per['transship'].sum():>12,.1f}")