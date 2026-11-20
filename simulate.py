"""The price of memorylessness: a simulation.

An agent repeatedly chooses among K options. Its choices depend on

* an accessible context (which option is on promotion today), and
* the latent activation of each option, raised by masked primes that the
  agent never perceives and that decays over trials.

We then fit three families of state-based (Markovian) predictors of the
agent's choices and measure, on held-out data, how far each is from the
true choice probabilities (excess cross-entropy, in nats per choice):

F1  accessible states: today's context plus the last k contexts and the
    agent's own last k choices -- everything the agent can access;
F2  augmented states: today's context plus the last k prime events --
    the full history, truncated at depth k;
F3  latent states: today's context plus the (discretised) activation of
    m of the K contents -- a bounded variable posited by the modeller.

These are three ways of making a primed agent Markovian, discussed in the
paper "Markov, Memory and Decision-Making: An Impossibility Result":
F1 stays accessible and is not sufficient; F2 becomes
sufficient only as its state space grows without bound; F3 is bounded and
sufficient only if it tracks every content, and its state is not
accessible to the agent.

Usage:  python simulate.py            (about one minute)
"""

import csv
import json
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
FIGURES = HERE / "figures"

# ----------------------------------------------------------------------------
# Parameters
# ----------------------------------------------------------------------------
K = 3              # options; content j primes option j
Q_PRIME = 0.4      # probability that a trial carries a masked prime
DECAY = 0.7        # activation retained from one trial to the next
BETA = 1.0         # weight of the accessible context (promotion)
GAMMA = 1.0        # weight of the latent activation
BURN_IN = 30       # trials discarded at the start of each agent's sequence

TRAIN = dict(agents=10_000, trials=1_000, seed=1)
TEST = dict(agents=2_000, trials=1_000, seed=2)

ACCESSIBLE_K = [0, 1, 2, 3, 4]
AUGMENTED_K = [0, 1, 2, 3, 4, 5, 6, 7]
LATENT_M = [1, 2, 3]
ACT_BINS = np.array([0.15, 0.4, 0.7, 1.0, 1.4])   # 6 activation levels
SMOOTH = 0.5        # additive smoothing of the count estimates


# ----------------------------------------------------------------------------
# Simulation
# ----------------------------------------------------------------------------
def simulate(agents, trials, seed):
    """Return arrays of shape (agents, trials)."""
    rng = np.random.default_rng(seed)
    context = rng.integers(0, K, size=(agents, trials), dtype=np.int8)
    has_prime = rng.random((agents, trials)) < Q_PRIME
    prime_target = rng.integers(0, K, size=(agents, trials), dtype=np.int8)
    # prime event: 0 = none, 1..K = masked prime for option j-1
    prime = np.where(has_prime, prime_target + 1, 0).astype(np.int8)

    act = np.zeros((agents, K))
    activation = np.zeros((agents, trials, K), dtype=np.float32)
    choice = np.zeros((agents, trials), dtype=np.int8)
    prob = np.zeros((agents, trials, K), dtype=np.float32)
    rows = np.arange(agents)

    for t in range(trials):
        # a prime at trial t-1 acts from trial t onwards (t_i < t_k)
        act *= DECAY
        if t > 0:
            p = prime[:, t - 1]
            hit = p > 0
            act[rows[hit], p[hit] - 1] += 1.0
        activation[:, t] = act
        util = GAMMA * act
        util[rows, context[:, t]] += BETA
        util -= util.max(axis=1, keepdims=True)
        w = np.exp(util)
        pr = w / w.sum(axis=1, keepdims=True)
        prob[:, t] = pr
        u = rng.random(agents)
        choice[:, t] = (u[:, None] > np.cumsum(pr, axis=1)).sum(axis=1)

    return dict(context=context, prime=prime, activation=activation,
                choice=choice, prob=prob)


# ----------------------------------------------------------------------------
# State encodings (each returns an integer state per trial and the number of
# possible states); trials before BURN_IN are dropped later.
# ----------------------------------------------------------------------------
def lagged(x, lag):
    out = np.zeros_like(x)
    out[:, lag:] = x[:, :-lag]
    return out


def accessible_state(d, k):
    s = d["context"].astype(np.int64)
    n = K
    for lag in range(1, k + 1):
        s = s * K + lagged(d["context"], lag)
        s = s * K + lagged(d["choice"], lag)
        n *= K * K
    return s, n


def augmented_state(d, k):
    s = d["context"].astype(np.int64)
    n = K
    for lag in range(1, k + 1):
        s = s * (K + 1) + lagged(d["prime"], lag)
        n *= K + 1
    return s, n


def latent_state(d, m):
    s = d["context"].astype(np.int64)
    n = K
    levels = len(ACT_BINS) + 1
    for j in range(m):
        s = s * levels + np.digitize(d["activation"][:, :, j], ACT_BINS)
        n *= levels
    return s, n


# ----------------------------------------------------------------------------
# Fitting and evaluation
# ----------------------------------------------------------------------------
def fit(states, choices, n_states):
    counts = np.bincount(states * K + choices, minlength=n_states * K)
    counts = counts.reshape(n_states, K).astype(float) + SMOOTH
    return counts / counts.sum(axis=1, keepdims=True)


def log_loss(table, states, choices):
    return float(-np.mean(np.log(table[states, choices])))


def evaluate(encoder, arg, train, test):
    s_tr, n = encoder(train, arg)
    s_te, _ = encoder(test, arg)
    keep = slice(BURN_IN, None)
    table = fit(s_tr[:, keep].ravel(), train["choice"][:, keep].ravel().astype(np.int64), n)
    loss = log_loss(table, s_te[:, keep].ravel(), test["choice"][:, keep].ravel().astype(np.int64))
    return n, loss


def main():
    RESULTS.mkdir(exist_ok=True)
    FIGURES.mkdir(exist_ok=True)

    train = simulate(**TRAIN)
    test = simulate(**TEST)

    keep = slice(BURN_IN, None)
    c_te = test["choice"][:, keep].ravel().astype(np.int64)
    p_te = test["prob"][:, keep].reshape(-1, K)
    oracle = float(-np.mean(np.log(p_te[np.arange(len(c_te)), c_te])))

    rows = []
    for k in ACCESSIBLE_K:
        n, loss = evaluate(accessible_state, k, train, test)
        rows.append(dict(family="F1 accessible", param=f"k={k}", states=n, loss=loss))
    for k in AUGMENTED_K:
        n, loss = evaluate(augmented_state, k, train, test)
        rows.append(dict(family="F2 augmented", param=f"k={k}", states=n, loss=loss))
    for m in LATENT_M:
        n, loss = evaluate(latent_state, m, train, test)
        rows.append(dict(family="F3 latent", param=f"m={m}", states=n, loss=loss))

    base = rows[0]["loss"]          # present accessible state only (k = 0)
    gap = base - oracle             # what memory can explain at most
    for r in rows:
        r["excess"] = r["loss"] - oracle
        r["recovered"] = (base - r["loss"]) / gap

    with open(RESULTS / "results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["family", "param", "states", "loss", "excess", "recovered"])
        w.writeheader()
        for r in rows:
            w.writerow({**r, "loss": f"{r['loss']:.5f}", "excess": f"{r['excess']:.5f}",
                        "recovered": f"{r['recovered']:.3f}"})
    summary = dict(
        parameters=dict(K=K, q_prime=Q_PRIME, decay=DECAY, beta=BETA, gamma=GAMMA,
                        burn_in=BURN_IN, train=TRAIN, test=TEST,
                        activation_bins=ACT_BINS.tolist(), smoothing=SMOOTH),
        oracle_loss=oracle,
        present_state_loss=base,
        rows=rows,
    )
    (RESULTS / "summary.json").write_text(json.dumps(summary, indent=2))

    # ------------------------------------------------------------------ plot
    styles = {
        "F1 accessible": dict(color="#2563EB", marker="o"),
        "F2 augmented": dict(color="#D97706", marker="s"),
        "F3 latent": dict(color="#059669", marker="^"),
    }
    labels = {
        "F1 accessible": "accessible state (context + own past choices)",
        "F2 augmented": "augmented state (history of primes)",
        "F3 latent": "latent state (activation of m contents)",
    }
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    for fam, st in styles.items():
        rs = [r for r in rows if r["family"] == fam]
        ax.plot([r["states"] for r in rs], [r["excess"] for r in rs],
                label=labels[fam], linewidth=1.6, markersize=5, **st)
        for r in rs:
            ax.annotate(r["param"], (r["states"], r["excess"]), textcoords="offset points",
                        xytext=(4, 4), fontsize=7, color=st["color"])
    ax.set_xscale("log")
    ax.set_xlabel("number of states (log scale)")
    ax.set_ylabel("excess cross-entropy (nats per choice)")
    ax.axhline(0, color="#64748B", linewidth=0.8)
    ax.legend(fontsize=7.5, frameon=False, loc="center right", bbox_to_anchor=(1.0, 0.62))
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIGURES / "price_of_memorylessness.png", dpi=200)
    fig.savefig(FIGURES / "price_of_memorylessness.pdf")

    print(f"oracle loss {oracle:.4f}   present-state loss {base:.4f}   gap {gap:.4f}")
    for r in rows:
        print(f"{r['family']:14s} {r['param']:5s} states={r['states']:>7d} "
              f"excess={r['excess']:.4f} recovered={r['recovered']:.2f}")


if __name__ == "__main__":
    main()
