"""跑法:~/voice-venv/bin/python test_engine_cache.py

測 CosyVoice 聲紋快取:同一支參考音只註冊一次 zero-shot 聲紋。
用 GlitchTTSEngine.__new__ 繞過 __init__,塞假的 cosyvoice,不載模型。
"""
import threading

import torch

import engine
from engine import COSY_INSTRUCT, GlitchTTSEngine


class FakeCosy:
    sample_rate = 24000

    def __init__(self):
        self.added = []      # (prompt_text, ref_wav, spk_id)
        self.infer = []      # (text, instruct, ref_wav, zero_shot_spk_id)

    def add_zero_shot_spk(self, prompt_text, prompt_wav, spk_id):
        self.added.append((prompt_text, prompt_wav, spk_id))
        return True

    def inference_instruct2(self, text, instruct_text, prompt_wav, zero_shot_spk_id="", **kw):
        self.infer.append((text, instruct_text, prompt_wav, zero_shot_spk_id))
        yield {"tts_speech": torch.full((1, 2400), 0.1)}


def make():
    e = GlitchTTSEngine.__new__(GlitchTTSEngine)
    e.cosyvoice = FakeCosy()
    e._lock = threading.Lock()
    e.active_engine = "cosyvoice_native"
    e.active_voice = "glitch"
    e.nfe_default = 12
    e._cosy_spk_ids = {}
    return e


def test_first_call_registers_speaker():
    e = make()
    e.synthesize_wav_bytes("你好", engine_name="cosyvoice", voice="輕柔女孩")
    assert len(e.cosyvoice.added) == 1, e.cosyvoice.added
    assert e.cosyvoice.added[0][0] == COSY_INSTRUCT


def test_second_call_reuses_speaker():
    e = make()
    e.synthesize_wav_bytes("你好", engine_name="cosyvoice", voice="輕柔女孩")
    e.synthesize_wav_bytes("再見", engine_name="cosyvoice", voice="輕柔女孩")
    assert len(e.cosyvoice.added) == 1, e.cosyvoice.added
    first, second = e.cosyvoice.infer
    assert first[3] == second[3] == e.cosyvoice.added[0][2] != ""
    assert second[1] == COSY_INSTRUCT


def test_different_voice_registers_again():
    e = make()
    e.synthesize_wav_bytes("你好", engine_name="cosyvoice", voice="輕柔女孩")
    e.synthesize_wav_bytes("你好", engine_name="cosyvoice", voice="glitch")
    assert len(e.cosyvoice.added) == 2, e.cosyvoice.added
    assert e.cosyvoice.added[0][2] != e.cosyvoice.added[1][2]


if __name__ == "__main__":
    bad = 0
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        try:
            t()
            print(f"ok   {t.__name__}")
        except Exception as ex:
            bad += 1
            print(f"FAIL {t.__name__}: {ex!r}")
    print(f"\n{len(tests)-bad}/{len(tests)} 通過")
    raise SystemExit(1 if bad else 0)
