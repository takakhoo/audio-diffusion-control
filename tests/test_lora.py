import pytest

torch = pytest.importorskip("torch")
from torch import nn

from audiosliders.lora import SliderBank


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn1 = nn.ModuleDict(dict(to_q=nn.Linear(8, 8), to_k=nn.Linear(8, 8), to_v=nn.Linear(8, 8)))
        self.attn1["to_out"] = nn.ModuleList([nn.Linear(8, 8)])
        self.attn2 = nn.ModuleDict(dict(to_q=nn.Linear(8, 8), to_k=nn.Linear(6, 8), to_v=nn.Linear(6, 8)))
        self.attn2["to_out"] = nn.ModuleList([nn.Linear(8, 8)])

    def forward(self, x, ctx):
        h = self.attn1["to_out"][0](self.attn1["to_q"](x) + self.attn1["to_v"](x))
        return h + self.attn2["to_out"][0](self.attn2["to_q"](h) + self.attn2["to_v"](ctx))


class Toy(nn.Module):
    def __init__(self):
        super().__init__()
        self.transformer_blocks = nn.ModuleList([Block(), Block()])

    def forward(self, x, ctx):
        for b in self.transformer_blocks:
            x = b(x, ctx)
        return x


def setup():
    torch.manual_seed(0)
    model = Toy()
    bank = SliderBank(model)
    return model, bank, torch.randn(4, 5, 8), torch.randn(4, 5, 6)


def randomize(bank, name):
    for p in bank.parameters(name):
        nn.init.normal_(p, std=0.3)


def test_new_slider_is_a_no_op_until_trained():
    model, bank, x, ctx = setup()
    ref = model(x, ctx)
    bank.add("a", rank=2, targets="attn")
    with bank.at(a=3.0):
        assert torch.allclose(model(x, ctx), ref, atol=1e-6)


def test_scale_zero_restores_base_and_sign_flips_delta():
    model, bank, x, ctx = setup()
    ref = model(x, ctx)
    bank.add("a", rank=2, targets="xattn")
    randomize(bank, "a")
    assert torch.allclose(model(x, ctx), ref, atol=1e-6)
    with bank.at(a=1.0):
        up = model(x, ctx)
    assert not torch.allclose(up, ref, atol=1e-3)
    assert torch.allclose(model(x, ctx), ref, atol=1e-6)


def test_per_sample_scales_match_scalar_runs():
    model, bank, x, ctx = setup()
    bank.add("a", rank=2, targets="attn")
    randomize(bank, "a")
    scales = torch.tensor([-1.0, 0.0, 0.5, 2.0])
    with bank.at(a=scales):
        batched = model(x, ctx)
    for i, s in enumerate(scales.tolist()):
        with bank.at(a=s):
            assert torch.allclose(model(x[i : i + 1], ctx[i : i + 1]), batched[i : i + 1], atol=1e-5)


def test_scales_tile_over_stacked_guidance_batch():
    model, bank, x, ctx = setup()
    bank.add("a", rank=2, targets="attn")
    randomize(bank, "a")
    scales = torch.tensor([1.0, -1.0, 0.5, 2.0])
    with bank.at(a=scales):
        single = model(x, ctx)
        stacked = model(torch.cat([x, x]), torch.cat([ctx, ctx]))
    assert torch.allclose(stacked[:4], single, atol=1e-5) and torch.allclose(stacked[4:], single, atol=1e-5)


def test_two_sliders_compose_and_targets_select_layers():
    model, bank, x, ctx = setup()
    n_self = len(bank.add("a", rank=2, targets="self"))
    n_cross = len(bank.add("b", rank=3, targets="xattn"))
    assert n_self == n_cross == 2 * 4 * 2
    randomize(bank, "a")
    randomize(bank, "b")
    ref = model(x, ctx)
    with bank.at(a=1.0):
        only_a = model(x, ctx)
    with bank.at(a=1.0, b=1.0):
        both = model(x, ctx)
    assert not torch.allclose(only_a, ref, atol=1e-3) and not torch.allclose(both, only_a, atol=1e-3)


def test_save_and_load_round_trip(tmp_path):
    model, bank, x, ctx = setup()
    bank.add("a", rank=2, targets="attn", positive="bright", negative="dark")
    randomize(bank, "a")
    with bank.at(a=1.5):
        want = model(x, ctx)
    bank.save("a", tmp_path / "a.safetensors")

    torch.manual_seed(0)
    fresh = Toy()
    fresh_bank = SliderBank(fresh)
    meta = fresh_bank.load("z", tmp_path / "a.safetensors")
    assert meta["positive"] == "bright" and meta["rank"] == 2
    with fresh_bank.at(z=1.5):
        assert torch.allclose(fresh(x, ctx), want, atol=1e-6)


def test_gated_predictor_only_acts_inside_window():
    model, bank, x, ctx = setup()
    bank.add("a", rank=2, targets="attn")
    randomize(bank, "a")
    base = lambda z, t: model(z, ctx)
    gated = bank.gated(base, {"a": 1.0}, start=0.6)
    ref = model(x, ctx)
    assert torch.allclose(gated(x, torch.tensor(0.9)), ref, atol=1e-6)
    assert not torch.allclose(gated(x, torch.tensor(0.3)), ref, atol=1e-3)
