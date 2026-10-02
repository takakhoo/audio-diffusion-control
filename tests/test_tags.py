import numpy as np

from audiosliders import tags as T


def fake_text_embeddings(dim=32, seed=0):
    rng = np.random.default_rng(seed)
    emb = rng.normal(size=(len(T.vocabulary()), dim))
    return emb / np.linalg.norm(emb, axis=1, keepdims=True)


def test_vocabulary_is_unique_and_templated():
    vocab = T.vocabulary()
    assert len(vocab) == sum(len(v) for v in T.VOCAB.values()) == 80
    assert len({t for _, t, _ in vocab}) == 80
    assert ("instrument", "piano", "music featuring piano") in vocab
    assert ("mood", "happy", "happy music") in vocab


def test_label_direction_recovers_the_two_ends():
    text = fake_text_embeddings()
    names = [t for _, t, _ in T.vocabulary()]
    happy, sad = text[names.index("happy")], text[names.index("sad")]
    toward, away = T.label_direction(happy - sad, text, k=1)
    assert toward == ["happy"] and away == ["sad"]


def test_tag_shift_reports_what_moved_between_two_sets():
    text = fake_text_embeddings()
    names = [t for _, t, _ in T.vocabulary()]
    rng = np.random.default_rng(1)
    base = rng.normal(size=(40, text.shape[1])) * 0.05
    high = base + 0.8 * text[names.index("drums")]
    low = base + 0.8 * text[names.index("harp")]
    up, down = T.tag_shift(high, low, text, k=1)
    assert up[0][0] == "drums" and up[0][1] > 0
    assert down[0][0] == "harp" and down[0][1] < 0
