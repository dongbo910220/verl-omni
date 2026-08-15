# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Dependency-light Qwen3-TTS actor and rollout contract tests."""

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

ROOT = Path(__file__).parents[2]


def _load(name: str, relative_path: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


forward = _load("qwen3_tts_forward_test", "verl_omni/pipelines/qwen3_tts/talker_forward.py")
rollout = _load("qwen3_tts_rollout_test", "verl_omni/pipelines/qwen3_tts/rollout_utils.py")

TOKENS = forward.TalkerTokens(900, 901, 902, 4196, 4197, 4198, 4203, 4204, 4205)


def test_talker_batch_matches_auto_language_teacher_forcing_layout():
    text_ids = torch.tensor([1, 2, 3, 4, 5, 6])
    codes = torch.arange(3 * 16, dtype=torch.long).reshape(3, 16) % 128
    codes[:, 0] = torch.tensor([10, 11, 12])

    batch = forward.build_talker_batch([text_ids], [codes], TOKENS, sub_codebook_vocab=2048)

    speaker_slot = 6
    codec_start = 8 + text_ids.numel() - 1
    assert batch.input_ids[0, 3:speaker_slot, 1].tolist() == [4203, 4204, 4205]
    assert not batch.codec_embedding_mask[0, speaker_slot]
    assert batch.input_ids[0, speaker_slot + 1, 1] == TOKENS.codec_pad
    torch.testing.assert_close(batch.codec_ids[0, codec_start : codec_start + 3], codes)
    assert batch.logit_start == [codec_start - 1]
    assert batch.codec_lens == [3]


def test_codec0_mask_matches_rollout_vocabulary():
    masked = forward.mask_codec0_logits(torch.zeros((1, 2, 3072)), 2048, 2150)

    assert (masked[..., 0] < -1e3).all()
    assert torch.isfinite(masked[..., 1:2048]).all()
    assert torch.isfinite(masked[..., 2150]).all()
    assert (masked[..., 2151:] < -1e3).all()


def test_only_validated_auto_language_layout_is_accepted():
    assert forward.require_auto_language("auto") == "Auto"
    with pytest.raises(ValueError, match="supports only tts_language=Auto"):
        forward.require_auto_language("Chinese")


def test_actor_logits_align_to_effective_codec0_response(monkeypatch):
    class CodePredictor:
        @staticmethod
        def get_input_embeddings():
            return [torch.nn.Embedding(2048, 4)]

    talker = SimpleNamespace(code_predictor=CodePredictor())
    model = SimpleNamespace(
        talker=talker,
        config=SimpleNamespace(
            tts_pad_token_id=TOKENS.tts_pad,
            tts_bos_token_id=TOKENS.tts_bos,
            tts_eos_token_id=TOKENS.tts_eos,
            talker_config=SimpleNamespace(
                codec_pad_id=TOKENS.codec_pad,
                codec_bos_id=TOKENS.codec_bos,
                codec_eos_token_id=TOKENS.codec_eos,
                codec_nothink_id=TOKENS.codec_nothink,
                codec_think_bos_id=TOKENS.codec_think_bos,
                codec_think_eos_id=TOKENS.codec_think_eos,
            ),
        ),
    )

    def fake_logits(_talker, batch, _speaker):
        return torch.ones((1, batch.input_ids.shape[1] - 1, 4300))

    monkeypatch.setattr(forward, "codec0_logits", fake_logits)
    input_ids = torch.zeros((1, 9), dtype=torch.long)
    input_ids[0, -3:] = torch.tensor([10, 11, 12])
    codes = torch.zeros((1, 6, 16), dtype=torch.long)
    codes[0, :3, 0] = torch.tensor([10, 11, 12])

    logits = forward.tts_actor_logits(
        model,
        input_ids,
        torch.ones_like(input_ids),
        torch.tensor([[1, 2, 3, 4, 5, 6]]),
        codes,
        torch.tensor([3]),
        torch.tensor([6]),
        torch.zeros((1, 4)),
    )

    assert torch.nonzero(logits.abs().sum(dim=-1)[0], as_tuple=False).reshape(-1).tolist() == [5, 6, 7]


@pytest.mark.parametrize("response_length", [2, 14, 15, 16, 17, 32])
def test_codec_alignment_recovers_exact_prefix_without_final_residual_row(response_length):
    token_ids = list(range(100, 100 + response_length - 1)) + [2150]
    generated = torch.arange((response_length - 1) * 16, dtype=torch.long).reshape(response_length - 1, 16) + 1
    generated[:, 0] = torch.tensor(token_ids[:-1])
    raw = torch.cat((torch.zeros(12, 16, dtype=torch.long), generated))

    aligned = rollout.align_audio_codes(raw, token_ids)

    assert aligned[:, 0].tolist() == token_ids
    torch.testing.assert_close(aligned[:-1, 1:], generated[:, 1:])
    assert not aligned[-1, 1:].any()


def test_codec_alignment_preserves_final_row_and_rejects_heuristic_match():
    token_ids = [101, 102, 103, 2150]
    generated = torch.arange(4 * 16, dtype=torch.long).reshape(4, 16) + 1
    generated[:, 0] = torch.tensor(token_ids)
    aligned = rollout.align_audio_codes(torch.cat((torch.zeros(12, 16, dtype=torch.long), generated)), token_ids)
    torch.testing.assert_close(aligned, generated)

    malformed = torch.zeros(15, 16, dtype=torch.long)
    malformed[12:, 0] = torch.tensor([101, 999, 103])
    with pytest.raises(RuntimeError, match="Could not exactly align"):
        rollout.align_audio_codes(malformed, token_ids)


def test_rollout_chunk_accumulator_handles_cumulative_and_delta_outputs():
    first = torch.zeros(12, 16, dtype=torch.long)
    cumulative = torch.cat((first, torch.ones(1, 16, dtype=torch.long)))
    assert torch.equal(rollout.append_tensor_chunk(first, cumulative), cumulative)

    delta = torch.full((1, 16), 2, dtype=torch.long)
    assert torch.equal(rollout.append_tensor_chunk(cumulative, delta), torch.cat((cumulative, delta)))
    with pytest.raises(RuntimeError, match="changed shape"):
        rollout.append_tensor_chunk(cumulative, torch.zeros(1, 15, dtype=torch.long))


def test_validation_seed_covers_both_codec_samplers_and_candidates_without_mutation():
    original = {"temperature": 0.8, "extra_args": {"existing": "kept"}}
    seeded = rollout.with_rollout_generation_seed(
        original,
        {"split": "validation", "generation_seed": np.int64(42017)},
        session_id=3,
        global_steps=100,
        require_session_id=True,
    )

    assert seeded == {
        "temperature": 0.8,
        "seed": 42020,
        "extra_args": {"existing": "kept", "tts_local_seed": 42020},
    }
    assert seeded == rollout.with_rollout_generation_seed(
        original,
        {"split": "validation", "generation_seed": np.int64(42017)},
        session_id=3,
        global_steps=0,
        require_session_id=True,
    )
    assert original == {"temperature": 0.8, "extra_args": {"existing": "kept"}}

    gate_first = rollout.with_rollout_generation_seed(
        {}, {"split": "gate", "id": "gate-7"}, session_id=0, global_steps=0, base_seed=42
    )
    gate_second = rollout.with_rollout_generation_seed(
        {}, {"split": "gate", "id": "gate-7"}, session_id=1, global_steps=100, base_seed=42
    )
    assert gate_first["seed"] != gate_second["seed"]
    assert gate_first == rollout.with_rollout_generation_seed(
        {}, {"split": "gate", "id": "gate-7"}, session_id=0, global_steps=100, base_seed=42
    )


def test_training_seeds_are_reproducible_and_group_diverse():
    kwargs = {
        "extra_info": {"split": "train", "id": "sample-7"},
        "global_steps": 12,
        "uid": "uid-7",
        "base_seed": 42,
        "require_session_id": True,
    }
    first = rollout.with_rollout_generation_seed({}, session_id=0, **kwargs)
    repeated = rollout.with_rollout_generation_seed({}, session_id=0, **kwargs)
    second = rollout.with_rollout_generation_seed({}, session_id=1, **kwargs)
    next_step = rollout.with_rollout_generation_seed({}, session_id=0, **{**kwargs, "global_steps": 13})

    assert first == repeated
    assert len({first["seed"], second["seed"], next_step["seed"]}) == 3
    assert first["seed"] == first["extra_args"]["tts_local_seed"]
    with pytest.raises(RuntimeError, match="session_id"):
        rollout.with_rollout_generation_seed({}, session_id=None, **kwargs)
