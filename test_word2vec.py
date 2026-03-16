"""
Tests for the NumPy Word2Vec implementation.

Covers:
  - Sigmoid correctness and numerical stability
  - Loss is positive and finite
  - Gradient correctness via numerical differentiation (finite differences)
  - A short training run reduces loss
  - most_similar returns plausible results
"""

import numpy as np
import pytest
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from word2vec import Word2Vec, tokenize, build_vocab, subsample, \
                     generate_skipgram_pairs, train
from collections import Counter


# ─────────────────────────────────────────────
#  Fixtures
# ─────────────────────────────────────────────

VOCAB_SIZE = 20
EMBED_DIM  = 8
SEED       = 7


@pytest.fixture
def model():
    return Word2Vec(VOCAB_SIZE, EMBED_DIM, seed=SEED)


@pytest.fixture
def simple_indices():
    """Fixed center, context, negatives for deterministic tests."""
    center  = 3
    context = 7
    negs    = np.array([1, 4, 9, 12, 15])
    return center, context, negs


# ─────────────────────────────────────────────
#  Unit tests
# ─────────────────────────────────────────────

class TestSigmoid:
    def test_zero(self):
        assert Word2Vec._sigmoid(np.array([0.0]))[0] == pytest.approx(0.5)

    def test_large_positive(self):
        assert Word2Vec._sigmoid(np.array([1000.0]))[0] == pytest.approx(1.0, abs=1e-6)

    def test_large_negative(self):
        assert Word2Vec._sigmoid(np.array([-1000.0]))[0] == pytest.approx(0.0, abs=1e-6)

    def test_no_nan(self):
        x = np.array([-1000.0, -1.0, 0.0, 1.0, 1000.0])
        assert np.all(np.isfinite(Word2Vec._sigmoid(x)))


class TestForwardAndLoss:
    def test_loss_is_positive(self, model, simple_indices):
        center, context, negs = simple_indices
        loss, *_ = model.forward_and_loss(center, context, negs)
        assert loss > 0

    def test_loss_is_finite(self, model, simple_indices):
        center, context, negs = simple_indices
        loss, *_ = model.forward_and_loss(center, context, negs)
        assert np.isfinite(loss)

    def test_gradient_shapes(self, model, simple_indices):
        center, context, negs = simple_indices
        loss, grad_vc, grad_vo, grad_vk = model.forward_and_loss(
            center, context, negs
        )
        assert grad_vc.shape == (EMBED_DIM,)
        assert grad_vo.shape == (EMBED_DIM,)
        assert grad_vk.shape == (len(negs), EMBED_DIM)


class TestGradients:
    """
    Numerical gradient check (finite differences).
    We perturb each dimension of W_in[center] and W_out[context] by ε and
    verify the analytic gradient matches.
    """
    EPS = 1e-5
    TOL = 1e-4   # relative tolerance for numerical vs analytic

    def _numerical_grad(self, model, center, context, negs, param_name, idx):
        """Compute ∂L/∂param[idx] numerically."""
        W = getattr(model, param_name)
        results = []
        for sign in (+1, -1):
            W[idx] += sign * self.EPS
            loss, *_ = model.forward_and_loss(center, context, negs)
            results.append(loss)
            W[idx] -= sign * self.EPS
        return (results[0] - results[1]) / (2 * self.EPS)

    def test_grad_W_in(self, model, simple_indices):
        center, context, negs = simple_indices
        _, grad_vc, _, _ = model.forward_and_loss(center, context, negs)
        for dim in range(EMBED_DIM):
            num = self._numerical_grad(model, center, context, negs,
                                       "W_in", (center, dim))
            assert abs(grad_vc[dim] - num) < self.TOL, \
                f"W_in grad mismatch at dim {dim}: analytic={grad_vc[dim]:.6f}  numerical={num:.6f}"

    def test_grad_W_out_context(self, model, simple_indices):
        center, context, negs = simple_indices
        _, _, grad_vo, _ = model.forward_and_loss(center, context, negs)
        for dim in range(EMBED_DIM):
            num = self._numerical_grad(model, center, context, negs,
                                       "W_out", (context, dim))
            assert abs(grad_vo[dim] - num) < self.TOL, \
                f"W_out context grad mismatch at dim {dim}"

    def test_grad_W_out_neg(self, model, simple_indices):
        center, context, negs = simple_indices
        _, _, _, grad_vk = model.forward_and_loss(center, context, negs)
        for ki, neg_idx in enumerate(negs):
            for dim in range(EMBED_DIM):
                num = self._numerical_grad(model, center, context, negs,
                                           "W_out", (neg_idx, dim))
                assert abs(grad_vk[ki, dim] - num) < self.TOL, \
                    f"W_out neg[{ki}] grad mismatch at dim {dim}"


class TestTraining:
    def test_loss_decreases(self):
        """A short training run on a toy corpus should reduce average loss."""
        corpus = (
            "the cat sat on the mat the cat is fat "
            "a dog ran in the park the dog is big "
            "cats and dogs are different animals "
        ) * 200

        tokens = tokenize(corpus)
        counts = Counter(tokens)
        word2idx, idx2word, neg_probs = build_vocab(tokens, min_count=3)
        token_ids = subsample(tokens, word2idx, counts)
        pairs     = generate_skipgram_pairs(token_ids, window=3)

        model = Word2Vec(len(word2idx), embed_dim=20, seed=1)
        losses = train(model, pairs, neg_probs,
                       n_epochs=3, lr=0.05, k_neg=5, log_every=999999)

        assert losses[-1] < losses[0], \
            f"Loss did not decrease: {losses[0]:.4f} → {losses[-1]:.4f}"

    def test_most_similar_runs(self):
        """most_similar should return the right number of results."""
        corpus = ("king queen man woman " * 300 +
                  "paris france london england " * 300)
        tokens = tokenize(corpus)
        counts = Counter(tokens)
        word2idx, idx2word, neg_probs = build_vocab(tokens, min_count=2)
        token_ids = subsample(tokens, word2idx, counts)
        pairs     = generate_skipgram_pairs(token_ids, window=2)

        model = Word2Vec(len(word2idx), embed_dim=10, seed=2)
        train(model, pairs, neg_probs,
              n_epochs=5, lr=0.05, k_neg=3, log_every=999999)

        results = model.most_similar("king", word2idx, idx2word, top_k=3)
        assert len(results) == 3
        for word, score in results:
            assert isinstance(word, str)
            assert -1.0 <= score <= 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
