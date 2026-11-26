"""A tiny autoregressive language model implemented purely with NumPy.

The model runs on CPU, has no runtime dependencies beyond NumPy and no
external weights: every parameter is derived deterministically from fixed
seeds so inference is fully reproducible.

Architecture
------------
* Character-level vocabulary (letters, digits, punctuation, space) plus an
  ``<unk>`` sentinel.
* The last ``CONTEXT`` characters are embedded with a *hash-projection*
  embedding (deterministic pseudo-random vectors keyed by token id), summed
  with a positional embedding and fed through a fixed tanh MLP.
* Final logits combine the MLP score with a deterministic hash-noise term
  (keyed by the current context window) and two hand-built structural
  priors that keep the greedy output text-like:
    - a word-length prior favouring spaces after several non-space tokens;
    - a vowel/consonant alternation prior conditioned on the previous token;
    - a repetition penalty against tokens already in the context window.

Using hash noise rather than learned random weights avoids the classic greedy
attractor (an MLP with tiny fixed weights collapses into a 2-token cycle),
while remaining deterministic: the same prefix always yields the same
continuation, which lets the annotation UI re-fetch an unacknowledged segment
and get byte-identical text back.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

VOCAB = (
    " "
    "abcdefghijklmnopqrstuvwxyz"
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "0123456789"
    ".,!?;:'\"-()\n"
)
UNK = "<unk>"
TOKENS = [UNK] + list(VOCAB)
TOKEN_TO_ID = {t: i for i, t in enumerate(TOKENS)}
VOCAB_SIZE = len(TOKENS)
SPACE_ID = TOKEN_TO_ID[" "]
VOWEL_IDS = np.array([TOKEN_TO_ID[c] for c in "aeiouAEIOU"], dtype=np.int64)
CONSONANT_IDS = np.array(
    [TOKEN_TO_ID[c] for c in "bcdfghjklmnpqrstvwxyzBCDFGHJKLMNPQRSTVWXYZ"],
    dtype=np.int64,
)
_LETTER_IDS = np.concatenate([VOWEL_IDS, CONSONANT_IDS])
PUNCT_IDS = np.array(
    [TOKEN_TO_ID[c] for c in ".,!?;:"], dtype=np.int64)

CONTEXT = 4
EMBED_DIM = 48
HIDDEN_DIM = 96


def _hash_matrix(rows: int, cols: int, seed: int) -> np.ndarray:
    """Deterministic pseudo-random matrix in [-1, 1) keyed by integer ids."""
    vec = np.arange(rows * cols, dtype=np.int64)
    h = (vec * np.int64(0x9E3779B1) + np.int64(seed)) & np.int64(0xFFFFFFFF)
    h = (h ^ (h >> 13)) * np.int64(0xC2B2AE35) & np.int64(0xFFFFFFFF)
    h = h ^ (h >> 16)
    return (h.astype(np.float64) / np.float64(0x80000000) - 1.0).reshape(
        rows, cols).astype(np.float32)


@dataclass
class ModelState:
    """Mutable decoding state for one generation stream (one order)."""

    context: list[int]
    non_space_run: int = 0

    @classmethod
    def initial(cls, prefix_ids: list[int]) -> "ModelState":
        return cls(context=list(prefix_ids[-CONTEXT:]))


class TinyARModel:
    """Deterministic character-level autoregressive model (NumPy, CPU)."""

    def __init__(self, seed: int = 20261126) -> None:
        # Hash-projection token embedding: row i is derived from id i.
        self.token_emb = _hash_matrix(VOCAB_SIZE, EMBED_DIM, seed)
        self.pos_emb = _hash_matrix(CONTEXT, EMBED_DIM, seed + 1) * 0.5
        self.w1 = _hash_matrix(EMBED_DIM, HIDDEN_DIM, seed + 2) / np.sqrt(
            EMBED_DIM).astype(np.float32)
        self.w2 = _hash_matrix(HIDDEN_DIM, VOCAB_SIZE, seed + 3) / np.sqrt(
            HIDDEN_DIM).astype(np.float32)
        self.b1 = np.zeros(HIDDEN_DIM, dtype=np.float32)
        self.b2 = np.zeros(VOCAB_SIZE, dtype=np.float32)
        # Per-context hash noise table, indexed lazily from a rolling hash;
        # shape (NOISE_SLOTS, VOCAB_SIZE).
        self.noise = _hash_matrix(4096, VOCAB_SIZE, seed + 4) * 0.9

    # ------------------------------------------------------------------ #
    # Encoding helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def encode(text: str) -> list[int]:
        return [TOKEN_TO_ID.get(ch, 0) for ch in text]

    @staticmethod
    def decode(ids) -> str:
        return "".join(TOKENS[int(i)] for i in ids)

    def new_state(self, prefix: str) -> ModelState:
        return ModelState.initial(self.encode(prefix))

    # ------------------------------------------------------------------ #
    # Forward pass
    # ------------------------------------------------------------------ #
    def _mlp_logits(self, contexts: np.ndarray) -> np.ndarray:
        """ contexts: (B, CONTEXT) int token ids -> (B, VOCAB_SIZE) logits. """
        x = self.token_emb[contexts] + self.pos_emb[None, :, :]
        h = x.sum(axis=1)  # (B, EMBED_DIM)
        hidden = np.tanh(h @ self.w1 + self.b1)
        return hidden @ self.w2 + self.b2

    def _noise_logits(self, contexts: np.ndarray) -> np.ndarray:
        """Context-keyed deterministic hash noise: breaks greedy attractors."""
        # Rolling-ish integer hash of the whole context window.
        keys = np.zeros(contexts.shape[0], dtype=np.int64)
        for pos in range(CONTEXT):
            keys = (keys * 131 + contexts[:, pos] + 17) & 0xFFF
        return self.noise[keys]

    def _structural_bias(self, states: list[ModelState]) -> np.ndarray:
        bias = np.zeros((len(states), VOCAB_SIZE), dtype=np.float32)
        for row, state in enumerate(states):
            run = state.non_space_run
            # Word-length prior: pressure for a space grows with the run,
            # hard-capped so punctuation-heavy text stays possible.
            if run >= 4:
                bias[row, SPACE_ID] += min(2.5, 0.45 * (run - 3))
            if state.context:
                last = state.context[-1]
                if last in CONSONANT_IDS:
                    bias[row, VOWEL_IDS] += 0.8
                elif last in VOWEL_IDS:
                    bias[row, CONSONANT_IDS] += 0.5
                    bias[row, SPACE_ID] += 0.25
                elif last == SPACE_ID:
                    bias[row, _LETTER_IDS] += 0.6
                    bias[row, PUNCT_IDS] -= 1.5
                elif last in PUNCT_IDS:
                    bias[row, SPACE_ID] += 1.2
                # Repetition penalty: the more recently a token appeared,
                # the stronger it is pushed down.
                for age, tok in enumerate(reversed(state.context)):
                    bias[row, tok] -= 1.1 / (age + 1)
            # Newline stays rare.
            bias[row, TOKEN_TO_ID["\n"]] -= 2.0
            # <unk> should practically never be emitted.
            bias[row, 0] -= 5.0
        return bias

    def _padded_contexts(self, states: list[ModelState]) -> np.ndarray:
        """Left-pad every context with zeros so single/batched runs match."""
        contexts = np.zeros((len(states), CONTEXT), dtype=np.int64)
        for i, state in enumerate(states):
            contexts[i, CONTEXT - len(state.context):] = state.context
        return contexts

    def generate_batch(self, states: list[ModelState]) -> list[int]:
        """Run one decoding step for every stream in a single batched pass.

        Returns one argmax token id per state and advances each state in
        place.  A stream decoded alone and the same stream decoded inside a
        wider batch always produce the identical token sequence because
        contexts are left-padded to a fixed width.
        """
        if not states:
            return []
        contexts = self._padded_contexts(states)
        logits = (self._mlp_logits(contexts)
                  + self._noise_logits(contexts)
                  + self._structural_bias(states))
        next_ids = np.argmax(logits, axis=1).tolist()
        for state, nid in zip(states, next_ids):
            state.context = (state.context + [nid])[-CONTEXT:]
            state.non_space_run = 0 if nid == SPACE_ID else state.non_space_run + 1
        return next_ids
