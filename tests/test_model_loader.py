from ml.inference.loader import load_model

def test_loader():
    model, tok, rec = load_model("kothagpt", device="cpu", quant="none")
    assert model is not None
    assert tok is not None
    assert rec.id == "kothagpt"
    assert rec.context_window == 8192

def test_loader_quant():
    model, _, _ = load_model("kothagpt-small", quant="int8")
    assert hasattr(model, "_quant_bits")

