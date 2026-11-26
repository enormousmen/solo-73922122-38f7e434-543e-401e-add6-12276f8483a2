"""Tests for the NumPy model: determinism, batched equivalence, varied output."""

import numpy as np

from app.model import SPACE_ID, TinyARModel, TOKENS


def test_deterministic_same_prefix_same_output():
    m1, m2 = TinyARModel(), TinyARModel()
    s1, s2 = m1.new_state("hello world"), m2.new_state("hello world")
    out1 = [m1.generate_batch([s1])[0] for _ in range(60)]
    out2 = [m2.generate_batch([s2])[0] for _ in range(60)]
    assert out1 == out2
    assert m1.decode(out1) == m2.decode(out2)


def test_batched_matches_single_stream_forward():
    """A stream decoded inside a 2-wide batch must equal decoding it alone."""
    model = TinyARModel()
    solo = model.new_state("prefix alpha")
    paired_a = model.new_state("prefix alpha")
    paired_b = model.new_state("different prefix beta")

    solo_out, pair_a, pair_b = [], [], []
    for _ in range(50):
        solo_out.append(model.generate_batch([solo])[0])
        a, b = model.generate_batch([paired_a, paired_b])
        pair_a.append(a)
        pair_b.append(b)
    assert pair_a == solo_out  # batching must not change per-stream results


def test_output_is_varied_text_not_one_repeated_token():
    model = TinyARModel()
    state = model.new_state("The quick brown fox")
    ids = [model.generate_batch([state])[0] for _ in range(200)]
    text = model.decode(ids)
    assert len(set(ids)) >= 12, text
    # structural prior should produce word-like segments separated by spaces
    assert text.count(" ") >= 8
    assert SPACE_ID in ids
    # no single run of identical chars longer than 3
    longest = cur = 1
    for a, b in zip(ids, ids[1:]):
        cur = cur + 1 if a == b else 1
        longest = max(longest, cur)
    assert longest <= 3


def test_different_prefixes_diverge():
    model = TinyARModel()
    sa, sb = model.new_state("alpha"), model.new_state("zzzzz 999")
    a = [model.generate_batch([sa])[0] for _ in range(40)]
    b = [model.generate_batch([sb])[0] for _ in range(40)]
    assert a != b
    assert all(i in range(len(TOKENS)) for i in a + b)


def test_batch_shapes_and_state_advance():
    model = TinyARModel()
    states = [model.new_state("ab"), model.new_state("cd"), model.new_state("")]
    out = model.generate_batch(states)
    assert len(out) == 3
    assert all(isinstance(x, int) for x in out)
    assert all(len(s.context) <= 4 for s in states)
    # empty prefix is a valid cold start
    assert isinstance(TOKENS[out[2]], str)
