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
"""CPU contracts for Qwen3-TTS's multi-stage rollout integration."""

from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("verl")
pytest.importorskip("vllm_omni")

from vllm import SamplingParams

from verl_omni.pipelines.qwen3_tts import omni_rollout_adapter, vllm_plugin
from verl_omni.pipelines.qwen3_tts.omni_rollout_adapter import Qwen3TTSRolloutAdapter
from verl_omni.pipelines.qwen3_tts.rollout_model import _align_prompt_embedding_dtype
from verl_omni.pipelines.qwen3_tts.talker_training_adapter import Qwen3TTSTalkerAdapter
from verl_omni.workers.rollout.vllm_rollout.utils import _receive_model_weight_buckets
from verl_omni.workers.rollout.vllm_rollout.vllm_omni_async_server import vLLMOmniHttpServer


class _Tokenizer:
    def decode(self, token_ids, **kwargs):
        return "first text" if token_ids == [1] else "other text"

    def __call__(self, text, **kwargs):
        return {"input_ids": list(range(len(text)))}


def test_rollout_pipeline_registers_dtype_aligned_talker(monkeypatch):
    registered_models = []
    registered_pipelines = []
    monkeypatch.setattr(
        vllm_plugin.ModelRegistry,
        "register_model",
        lambda architecture, model_class: registered_models.append((architecture, model_class)),
    )
    monkeypatch.setattr(
        omni_rollout_adapter,
        "register_pipeline",
        lambda pipeline: registered_pipelines.append(pipeline),
    )

    Qwen3TTSRolloutAdapter.ensure_pipeline_registered()

    assert registered_models == [
        (
            "Qwen3TTSDtypeAlignedTalkerForConditionalGeneration",
            "verl_omni.pipelines.qwen3_tts.rollout_model:Qwen3TTSDtypeAlignedTalkerForConditionalGeneration",
        )
    ]
    assert registered_pipelines == [omni_rollout_adapter.QWEN3_TTS_RL_PIPELINE]
    assert registered_pipelines[0].model_arch == registered_models[0][0]


def test_rollout_model_aligns_talker_and_prompt_builder_embedding_dtype():
    model = SimpleNamespace(
        _embedding_dtype=torch.bfloat16,
        _prompt_builder=SimpleNamespace(_embedding_dtype=torch.bfloat16),
    )

    _align_prompt_embedding_dtype(model, torch.float32)

    assert model._embedding_dtype == torch.float32
    assert model._prompt_builder._embedding_dtype == torch.float32


def test_rollout_adapter_builds_unique_prompt_and_scopes_weight_sync(tmp_path):
    speaker = tmp_path / "speaker.json"
    speaker.write_text("[0.0, 1.0]")
    model_config = SimpleNamespace(
        tokenizer=_Tokenizer(),
        override_config={"tts_spk_embed_path": str(speaker), "tts_language": "Auto"},
        hf_config=SimpleNamespace(talker_config=SimpleNamespace(codec_eos_token_id=2150)),
    )

    first = Qwen3TTSRolloutAdapter.prepare_engine_prompt([1], model_config, {})
    second = Qwen3TTSRolloutAdapter.prepare_engine_prompt([2], model_config, {})

    assert first["additional_information"]["text"] == ["first text"]
    assert first["cache_salt"] != second["cache_salt"]
    assert Qwen3TTSRolloutAdapter.weight_sync_stage_ids("full") == [0]
    assert not Qwen3TTSRolloutAdapter.supports_cache_engine_sleep("full")
    assert Qwen3TTSRolloutAdapter.get_output_modalities("full") == ["latent", "audio"]


def test_rollout_adapter_requires_speaker_embedding():
    model_config = SimpleNamespace(
        tokenizer=_Tokenizer(),
        override_config={"tts_language": "Auto"},
        hf_config=SimpleNamespace(talker_config=SimpleNamespace(codec_eos_token_id=2150)),
    )

    with pytest.raises(ValueError, match="requires tts_spk_embed_path"):
        Qwen3TTSRolloutAdapter.prepare_engine_prompt([1], model_config, {})


def test_talker_adapter_pads_exact_rollout_fields_for_actor_forward():
    model_inputs = {"input_ids": torch.zeros(2, 6, dtype=torch.long)}
    micro_batch = {
        "extra_fields": [
            {"tts_text_ids": [1, 2], "tts_audio_codes": torch.ones(3, 16, dtype=torch.long)},
            {"tts_text_ids": [3, 4, 5], "tts_audio_codes": torch.full((2, 16), 2, dtype=torch.long)},
        ]
    }

    prepared = Qwen3TTSTalkerAdapter.prepare_model_inputs(model_inputs, micro_batch, None)

    assert prepared["tts_text_ids"].shape == (2, 3)
    assert prepared["tts_audio_codes"].shape == (2, 3, 16)
    assert prepared["text_len"].tolist() == [2, 3]
    assert prepared["response_len"].tolist() == [3, 2]
    assert not prepared["tts_audio_codes"][1, 2].any()


def test_talker_adapter_requires_v1_agent_loop_extra_fields():
    model_inputs = {"input_ids": torch.zeros(1, 4, dtype=torch.long)}

    with pytest.raises(RuntimeError, match="V1 agent-loop trainer"):
        Qwen3TTSTalkerAdapter.prepare_model_inputs(model_inputs, {}, None)


def test_rollout_adapter_combines_policy_codes_and_waveform():
    token_ids = [101, 102, 2150]
    generated = torch.arange(3 * 16, dtype=torch.long).reshape(3, 16) + 1
    generated[:, 0] = torch.tensor(token_ids)
    policy = SimpleNamespace(
        stage_id=0,
        request_output=SimpleNamespace(outputs=[SimpleNamespace(token_ids=token_ids)]),
        multimodal_output={"codes": {"audio": torch.cat((torch.zeros(12, 16), generated))}},
    )
    decoder = SimpleNamespace(
        stage_id=1,
        request_output=None,
        multimodal_output={"audio": torch.ones(2400), "sr": 24_000},
    )
    prompt = {"additional_information": {"text": ["first text"]}}

    selected, fields = Qwen3TTSRolloutAdapter.combine_engine_outputs([policy, decoder], prompt)

    assert selected is policy
    torch.testing.assert_close(fields["tts_audio_codes"], generated.long())
    torch.testing.assert_close(fields["audio"], torch.ones(2400))
    assert fields["audio_sample_rate"] == 24_000
    assert fields["tts_text"] == "first text"


def test_bucketed_weight_sync_rebuilds_derived_codec_table_once(monkeypatch):
    class Model:
        def __init__(self):
            self.rebuilds = 0
            self.loads = 0
            self._stacked_codec_embed = object()

        def _build_stacked_codec_embed(self):
            self.rebuilds += 1
            self._stacked_codec_embed = object()

        def load_weights(self, weights):
            self.loads += 1
            self._build_stacked_codec_embed()

    class Receiver:
        @staticmethod
        def receive_weights(on_bucket_received):
            on_bucket_received({"first": torch.tensor(1)})
            on_bucket_received({"second": torch.tensor(2)})

    model = Model()
    empty_cache_calls = []
    monkeypatch.setattr(
        "verl_omni.workers.rollout.vllm_rollout.utils.get_torch_device",
        lambda: SimpleNamespace(empty_cache=lambda: empty_cache_calls.append(True)),
    )

    _receive_model_weight_buckets(Receiver(), model)

    assert model.loads == 2
    assert model.rebuilds == 1
    assert empty_cache_calls == [True]
    model._build_stacked_codec_embed()
    assert model.rebuilds == 2


def test_server_prepares_stage_specific_sampling_params():
    class Adapter:
        @staticmethod
        def prepare_engine_prompt(**kwargs):
            return {
                "prompt_token_ids": [1, 1, 1, 1],
                "additional_information": {"text": ["hello"]},
            }

    server = object.__new__(vLLMOmniHttpServer)
    server._ar_mode = True
    server._omni_rollout_adapter = Adapter
    server._stage_sampling_constraints = {0: {}}
    server.model_config = SimpleNamespace()
    server.config = SimpleNamespace(
        max_model_len=64,
        prompt_length=16,
        response_length=8,
        repetition_penalty=1.0,
    )
    server.engine = SimpleNamespace(default_sampling_params_list=[SamplingParams(), SimpleNamespace(stage="decoder")])

    prompt, params = server._preprocess_input(
        [5, 6],
        {"temperature": 0.8, "logprobs": True},
        {},
        None,
        None,
    )

    assert prompt["additional_information"]["max_new_tokens"] == [8]
    assert len(params) == 2
    assert params[0].max_tokens == 8
    assert params[0].temperature == pytest.approx(0.8)
    assert params[0].logprobs == 0
    assert params[1].stage == "decoder"


@pytest.mark.asyncio
async def test_server_retains_requested_stage_outputs_and_targets_weight_sync():
    policy = SimpleNamespace(request_output=SimpleNamespace(outputs=[]))

    class Engine:
        def __init__(self):
            self.generate_kwargs = None
            self.rpc_kwargs = None

        async def generate(self, **kwargs):
            self.generate_kwargs = kwargs
            yield policy

        async def collective_rpc(self, **kwargs):
            self.rpc_kwargs = kwargs
            return "rpc-result"

    class Adapter:
        @staticmethod
        def combine_engine_outputs(outputs, prompt):
            assert outputs == [policy]
            return policy, {"audio_sample_rate": 24_000}

    server = object.__new__(vLLMOmniHttpServer)
    server._ar_mode = True
    server.engine = Engine()
    server._rollout_output_modalities = ["latent", "audio"]
    server._omni_rollout_adapter = Adapter
    server._weight_sync_stage_ids = [0]

    result = await server._run_generation({"prompt_token_ids": [1]}, SamplingParams(), "request-0", None, 0)
    rpc_result = await server.collective_rpc("update_weights_from_ipc", kwargs={"base_sync_done": True})

    assert result is policy
    assert result._verl_omni_rollout_fields == {"audio_sample_rate": 24_000}
    assert server.engine.generate_kwargs["output_modalities"] == ["latent", "audio"]
    assert server.engine.rpc_kwargs["stage_ids"] == [0]
    assert rpc_result == "rpc-result"
