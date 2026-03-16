"""
Word2Vec — Skip-gram with Negative Sampling
Implemented in pure NumPy (no ML frameworks).

Architecture:
  - Two embedding matrices: W_in (center words) and W_out (context/negative words)
  - Objective: maximise log-sigmoid(v_context · v_center)
               + sum_k log-sigmoid(−v_neg_k · v_center)
  - Updates via SGD with analytically derived gradients

Author: <your name>
"""

import numpy as np
import re
import random
import time
import pickle
from collections import Counter
from pathlib import Path


# ─────────────────────────────────────────────
#  1.  TEXT PRE-PROCESSING
# ─────────────────────────────────────────────

def tokenize(text: str) -> list[str]:
    """Lowercase and split on non-alpha characters."""
    return re.findall(r"[a-z]+", text.lower())


def build_vocab(tokens: list[str],
                min_count: int = 5
                ) -> tuple[dict, dict, np.ndarray]:
    """
    Build word↔index mappings and a unigram frequency table.

    The unigram table is raised to the 3/4 power (as in the original paper)
    to smooth the distribution and favour rare words slightly.

    Returns
    -------
    word2idx : dict[str, int]
    idx2word : dict[int, str]
    neg_probs : np.ndarray  — smoothed sampling probabilities (len = vocab_size)
    """
    counts = Counter(tokens)
    vocab = [w for w, c in counts.items() if c >= min_count]
    vocab.sort()                               # deterministic ordering

    word2idx = {w: i for i, w in enumerate(vocab)}
    idx2word = {i: w for w, i in word2idx.items()}

    # Smoothed unigram distribution for negative sampling
    freqs = np.array([counts[w] ** 0.75 for w in vocab], dtype=np.float64)
    neg_probs = freqs / freqs.sum()

    return word2idx, idx2word, neg_probs


def subsample(tokens: list[str],
              word2idx: dict,
              counts: Counter,
              t: float = 1e-4) -> list[int]:
    """
    Frequent-word subsampling (Mikolov et al., 2013 §2.3).

    Each token w is discarded with probability:
        P(discard) = 1 − sqrt(t / f(w))
    where f(w) is the relative frequency of w.

    This speeds up training and improves quality for rare words.
    """
    total = sum(counts[w] for w in word2idx)
    keep = []
    for tok in tokens:
        if tok not in word2idx:
            continue
        f = counts[tok] / total
        p_keep = min(1.0, (np.sqrt(f / t) + 1) * (t / f))
        if random.random() < p_keep:
            keep.append(word2idx[tok])
    return keep


# ─────────────────────────────────────────────
#  2.  MODEL PARAMETERS
# ─────────────────────────────────────────────

class Word2Vec:
    """
    Parameters
    ----------
    vocab_size : int
    embed_dim  : int      — dimensionality of word vectors
    seed       : int      — for reproducibility

    Attributes
    ----------
    W_in  : (vocab_size, embed_dim)  — centre-word embeddings
    W_out : (vocab_size, embed_dim)  — context-word embeddings
    """

    def __init__(self, vocab_size: int, embed_dim: int = 100, seed: int = 42):
        rng = np.random.default_rng(seed)
        # Initialise with small uniform noise (common heuristic)
        limit = 0.5 / embed_dim
        self.W_in  = rng.uniform(-limit, limit, (vocab_size, embed_dim))
        self.W_out = np.zeros((vocab_size, embed_dim))

    # ── forward + loss ──────────────────────────────────────────────────

    @staticmethod
    def _sigmoid(x: np.ndarray) -> np.ndarray:
        """
        Numerically stable sigmoid.

        For x >= 0:  σ(x) = 1 / (1 + e^{-x})
        For x <  0:  σ(x) = e^x / (1 + e^x)

        We avoid computing both branches simultaneously to prevent NumPy
        from raising overflow warnings on the branch that won't be used.
        """
        out = np.empty_like(x, dtype=np.float64)
        pos = x >= 0
        out[ pos] = 1.0 / (1.0 + np.exp(-x[ pos]))
        exp_x = np.exp(x[~pos])
        out[~pos] = exp_x / (1.0 + exp_x)
        return out

    def forward_and_loss(self,
                         center_idx:  int,
                         context_idx: int,
                         neg_indices: np.ndarray
                         ) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute the negative-sampling loss for one (center, context) pair.

        Loss (to *minimise*) — negative of the SGNS objective:
            L = −log σ(v_o · v_c) − Σ_k log σ(−v_k · v_c)

        where:
            v_c = W_in[center_idx]          — centre-word vector
            v_o = W_out[context_idx]         — positive context vector
            v_k = W_out[neg_indices[k]]      — negative sample vectors

        Returns
        -------
        loss    : scalar
        grad_vc : gradient w.r.t. v_c  (shape: embed_dim,)
        grad_vo : gradient w.r.t. v_o  (shape: embed_dim,)
        grad_vk : gradients w.r.t. each v_k  (shape: K × embed_dim)
        """
        v_c = self.W_in[center_idx]          # (D,)
        v_o = self.W_out[context_idx]         # (D,)
        V_k = self.W_out[neg_indices]         # (K, D)

        # ── scores ───────────────────────────────────────────────────────
        pos_score = v_o @ v_c                 # scalar
        neg_scores = V_k @ v_c               # (K,)

        # ── sigmoid activations ──────────────────────────────────────────
        sig_pos  = self._sigmoid( pos_score)  # σ( v_o · v_c)
        sig_neg  = self._sigmoid(-neg_scores) # σ(−v_k · v_c),  shape (K,)

        # ── loss  (negated because we minimise) ──────────────────────────
        eps = 1e-10   # numerical safety
        loss = -(np.log(sig_pos + eps) + np.sum(np.log(sig_neg + eps)))

        # ── gradients ────────────────────────────────────────────────────
        # ∂L/∂v_o = (σ(v_o·v_c) − 1) · v_c
        grad_vo = (sig_pos - 1.0) * v_c      # (D,)

        # ∂L/∂v_k = (1 − σ(−v_k·v_c)) · v_c  =  σ(v_k·v_c) · v_c
        # shape: (K, D)
        grad_vk = (1.0 - sig_neg)[:, None] * v_c[None, :]

        # ∂L/∂v_c = (σ(v_o·v_c)−1)·v_o + Σ_k (1−σ(−v_k·v_c))·v_k
        grad_vc = (sig_pos - 1.0) * v_o + (1.0 - sig_neg) @ V_k  # (D,)

        return loss, grad_vc, grad_vo, grad_vk

    # ── parameter update (SGD) ──────────────────────────────────────────

    def update(self,
               center_idx:  int,
               context_idx: int,
               neg_indices: np.ndarray,
               lr: float
               ) -> float:
        """Run one forward pass and update embeddings in-place via SGD."""
        loss, grad_vc, grad_vo, grad_vk = self.forward_and_loss(
            center_idx, context_idx, neg_indices
        )

        self.W_in[center_idx]        -= lr * grad_vc
        self.W_out[context_idx]      -= lr * grad_vo
        self.W_out[neg_indices]      -= lr * grad_vk

        return float(loss)

    def update_batch(self,
                     center_ids:  np.ndarray,
                     context_ids: np.ndarray,
                     neg_ids:     np.ndarray,
                     lr: float
                     ) -> float:
        """
        Vectorised update for a batch of (center, context) pairs.

        Parameters
        ----------
        center_ids  : (B,)    — center word indices
        context_ids : (B,)    — positive context indices
        neg_ids     : (B, K)  — negative sample indices
        lr          : float

        Gradients are computed in parallel across the batch, then
        accumulated with np.add.at (handles repeated indices correctly).

        Returns
        -------
        mean loss over the batch
        """
        B = len(center_ids)

        V_c = self.W_in[center_ids]          # (B, D)
        V_o = self.W_out[context_ids]         # (B, D)
        V_k = self.W_out[neg_ids]             # (B, K, D)

        pos_scores = np.einsum('bd,bd->b', V_o, V_c)        # (B,)
        neg_scores = np.einsum('bd,bkd->bk', V_c, V_k)      # (B, K)

        sig_pos = self._sigmoid( pos_scores)                 # (B,)
        sig_neg = self._sigmoid(-neg_scores)                 # (B, K)

        eps = 1e-10
        loss = -(np.sum(np.log(sig_pos + eps)) +
                 np.sum(np.log(sig_neg + eps))) / B

        # gradients
        d_vo = (sig_pos - 1.0)[:, None] * V_c               # (B, D)
        d_vk = (1.0 - sig_neg)[:, :, None] * V_c[:, None, :]  # (B, K, D)
        d_vc = ((sig_pos - 1.0)[:, None] * V_o
                + np.einsum('bk,bkd->bd', 1.0 - sig_neg, V_k))  # (B, D)

        # SGD: np.add.at handles repeated indices safely
        np.add.at(self.W_in,  center_ids,  -lr * d_vc)
        np.add.at(self.W_out, context_ids, -lr * d_vo)
        np.add.at(self.W_out, neg_ids.ravel(),
                  -lr * d_vk.reshape(-1, self.W_out.shape[1]))

        return float(loss)

    # ── cosine similarity helpers ────────────────────────────────────────

    def most_similar(self,
                     word: str,
                     word2idx: dict,
                     idx2word: dict,
                     top_k: int = 10) -> list[tuple[str, float]]:
        """Return the top-k most similar words by cosine similarity."""
        if word not in word2idx:
            raise KeyError(f"'{word}' not in vocabulary.")
        idx  = word2idx[word]
        vec  = self.W_in[idx]
        # Cosine similarity against all W_in rows
        norms = np.linalg.norm(self.W_in, axis=1) + 1e-10
        sims  = self.W_in @ vec / (norms * np.linalg.norm(vec) + 1e-10)
        sims[idx] = -np.inf                   # exclude the word itself
        top  = np.argsort(sims)[::-1][:top_k]
        return [(idx2word[i], float(sims[i])) for i in top]

    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump({"W_in": self.W_in, "W_out": self.W_out}, f)

    @classmethod
    def load(cls, path: str) -> "Word2Vec":
        with open(path, "rb") as f:
            d = pickle.load(f)
        m = cls.__new__(cls)
        m.W_in, m.W_out = d["W_in"], d["W_out"]
        return m


# ─────────────────────────────────────────────
#  3.  TRAINING
# ─────────────────────────────────────────────

def generate_skipgram_pairs(token_ids: list[int],
                             window: int = 5
                             ) -> list[tuple[int, int]]:
    """
    Yield (center, context) index pairs for Skip-gram.

    The window radius is sampled uniformly in [1, window] for each center,
    following the original paper's practice.
    """
    pairs = []
    n = len(token_ids)
    for i, center in enumerate(token_ids):
        radius = random.randint(1, window)
        lo = max(0, i - radius)
        hi = min(n, i + radius + 1)
        for j in range(lo, hi):
            if j != i:
                pairs.append((center, token_ids[j]))
    return pairs


def train(model:       Word2Vec,
          pairs:       list[tuple[int, int]],
          neg_probs:   np.ndarray,
          n_epochs:    int   = 5,
          lr:          float = 0.025,
          min_lr:      float = 0.0001,
          k_neg:       int   = 5,
          batch_size:  int   = 512,
          log_every:   int   = 100_000,
          seed:        int   = 0
          ) -> list[float]:
    """
    Train word2vec with batched vectorised updates and linear LR decay.

    Instead of updating one pair at a time (slow Python loop), pairs are
    grouped into mini-batches and processed with NumPy array ops, giving
    ~20-30x speedup over the naive loop while keeping pure-NumPy semantics.

    Parameters
    ----------
    model      : Word2Vec instance (modified in-place)
    pairs      : list of (center_idx, context_idx) training pairs
    neg_probs  : smoothed unigram sampling distribution
    n_epochs   : number of full passes over pairs
    lr         : initial learning rate
    min_lr     : minimum learning rate (floor)
    k_neg      : number of negative samples per positive pair
    batch_size : pairs processed per gradient step
    log_every  : print status every N *pairs* processed
    seed       : random seed for negative sampling

    Returns
    -------
    epoch_losses : average loss per epoch
    """
    rng = np.random.default_rng(seed)
    vocab_size  = len(neg_probs)
    pairs_arr   = np.array(pairs, dtype=np.int32)   # (N, 2) — faster indexing
    total_pairs = n_epochs * len(pairs)
    epoch_losses = []

    processed = 0
    for epoch in range(1, n_epochs + 1):
        rng.shuffle(pairs_arr)
        epoch_loss  = 0.0
        n_batches   = 0
        t0          = time.time()
        last_log    = 0

        for start in range(0, len(pairs_arr), batch_size):
            batch       = pairs_arr[start:start + batch_size]
            center_ids  = batch[:, 0]
            context_ids = batch[:, 1]

            # draw all negative samples for the batch at once
            neg_ids = rng.choice(vocab_size,
                                 size=(len(batch), k_neg),
                                 replace=True, p=neg_probs)

            # linear LR decay based on pairs processed so far
            lr_t = max(min_lr, lr * (1.0 - processed / total_pairs))

            loss = model.update_batch(center_ids, context_ids, neg_ids, lr_t)
            epoch_loss += loss
            n_batches  += 1
            processed  += len(batch)

            if processed - last_log >= log_every:
                avg     = epoch_loss / n_batches
                elapsed = time.time() - t0
                pct     = 100 * processed / total_pairs
                print(f"  pairs {processed:>9,}  ({pct:5.1f}%)  "
                      f"loss={avg:.4f}  lr={lr_t:.5f}  "
                      f"elapsed={elapsed:.0f}s")
                last_log = processed

        avg_epoch_loss = epoch_loss / n_batches
        epoch_losses.append(avg_epoch_loss)
        print(f"Epoch {epoch}/{n_epochs}  avg_loss={avg_epoch_loss:.4f}  "
              f"time={time.time()-t0:.1f}s")

    return epoch_losses


# ─────────────────────────────────────────────
#  4.  DEMO / ENTRY POINT
# ─────────────────────────────────────────────

def load_text(path: str | None = None) -> str:
    """
    Load text corpus.  Falls back to a short built-in excerpt if no path
    is given (so the file runs out-of-the-box for quick testing).
    """
    if path and Path(path).exists():
        return Path(path).read_text(encoding="utf-8", errors="ignore")

    # Minimal fallback — replace with WikiText-103 or PubMed for real runs
    print("[INFO] No corpus file found.  Using built-in toy corpus.")
    return (
        "the quick brown fox jumps over the lazy dog "
        "a fox is a quick and cunning animal "
        "dogs are loyal animals and very friendly "
        "the dog chased the fox across the field "
        "natural language processing studies how computers understand human language "
        "word embeddings represent words as dense vectors in a continuous space "
        "similar words tend to have similar vector representations "
        "the king and queen ruled the kingdom together "
        "man and woman are both human beings "
        "paris is the capital of france and london is the capital of england "
    ) * 500   # repeat to get a non-trivial number of tokens


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Word2Vec – Skip-gram + Negative Sampling (NumPy)")
    parser.add_argument("--corpus",     default=None,  help="Path to plain-text corpus file")
    parser.add_argument("--embed_dim",  type=int, default=100)
    parser.add_argument("--window",     type=int, default=5)
    parser.add_argument("--min_count",  type=int, default=5)
    parser.add_argument("--epochs",     type=int, default=5)
    parser.add_argument("--lr",         type=float, default=0.025)
    parser.add_argument("--k_neg",      type=int, default=5)
    parser.add_argument("--save",       default="results/word2vec.pkl")
    args = parser.parse_args()

    # ── 1. load & tokenise ──────────────────────────────────────────────
    print("Loading corpus …")
    text   = load_text(args.corpus)
    tokens = tokenize(text)
    print(f"  {len(tokens):,} tokens before subsampling")

    # ── 2. build vocabulary ─────────────────────────────────────────────
    counts   = Counter(tokens)
    word2idx, idx2word, neg_probs = build_vocab(tokens, min_count=args.min_count)
    print(f"  vocab size = {len(word2idx):,}")

    # ── 3. subsample frequent words ─────────────────────────────────────
    token_ids = subsample(tokens, word2idx, counts)
    print(f"  {len(token_ids):,} tokens after subsampling")

    # ── 4. generate skip-gram pairs ─────────────────────────────────────
    print("Generating skip-gram pairs …")
    pairs = generate_skipgram_pairs(token_ids, window=args.window)
    print(f"  {len(pairs):,} (center, context) pairs")

    # ── 5. initialise model ─────────────────────────────────────────────
    model = Word2Vec(vocab_size=len(word2idx), embed_dim=args.embed_dim)

    # ── 6. train ────────────────────────────────────────────────────────
    print(f"\nTraining  (embed_dim={args.embed_dim}, "
          f"window={args.window}, k_neg={args.k_neg}, "
          f"lr={args.lr}, epochs={args.epochs}) …\n")
    epoch_losses = train(
        model, pairs, neg_probs,
        n_epochs=args.epochs,
        lr=args.lr,
        k_neg=args.k_neg,
        batch_size=512,
    )

    # ── 7. save ─────────────────────────────────────────────────────────
    Path(args.save).parent.mkdir(parents=True, exist_ok=True)
    model.save(args.save)
    print(f"\nModel saved → {args.save}")

    # ── 8. quick sanity check ───────────────────────────────────────────
    probe_words = [w for w in ("king", "dog", "language", "fox", "word")
                   if w in word2idx]
    for w in probe_words[:3]:
        print(f"\nMost similar to '{w}':")
        for neighbour, sim in model.most_similar(w, word2idx, idx2word, top_k=5):
            print(f"  {neighbour:<20} {sim:.4f}")


if __name__ == "__main__":
    main()
