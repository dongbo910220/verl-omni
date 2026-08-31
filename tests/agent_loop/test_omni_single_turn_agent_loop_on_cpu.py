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
"""CPU contracts for autoregressive omni rollout-to-replay policy output."""

from types import SimpleNamespace

import pytest
import torch
from verl.experimental.agent_loop.agent_loop import AgentLoopMetrics, AgentLoopOutput
from verl.experimental.agent_loop.single_turn_agent_loop import SingleTurnAgentLoop

from verl_omni.agent_loop.single_turn_agent_loop import OmniSingleTurnAgentLoop
from verl_omni.pipelines.model_base import OmniModelBase, OmniRolloutPipelineBase


def _output(extra_fields):
    return AgentLoopOutput(
        prompt_ids=[10, 11],
        response_ids=[1],
        response_mask=[1],
        response_logprobs=[-0.1],
        num_turns=2,
        metrics=AgentLoopMetrics(),
        extra_fields=extra_fields,
    )


def test_omni_single_turn_agent_resolves_registered_adapter(monkeypatch):
    class Adapter(OmniRolloutPipelineBase):
        @classmethod
        def build_stage_configs(cls, pipeline_mode="talker"):
            return []

    monkeypatch.setattr(OmniRolloutPipelineBase, "_registry", {"test_talker": Adapter})
    config = SimpleNamespace(engine_kwargs={"vllm_omni": {"pipeline_name": "test_talker"}})

    assert OmniSingleTurnAgentLoop._resolve_rollout_adapter(config) is Adapter

    with pytest.raises(ValueError, match="requires a registered"):
        OmniSingleTurnAgentLoop._resolve_rollout_adapter(
            SimpleNamespace(engine_kwargs={"vllm_omni": {"pipeline_name": "missing"}})
        )
    with pytest.raises(ValueError, match="requires engine_kwargs"):
        OmniSingleTurnAgentLoop._resolve_rollout_adapter(SimpleNamespace(engine_kwargs={}))


@pytest.mark.asyncio
async def test_talker_contract_supports_different_trajectory_and_conditioning_shapes(monkeypatch):
    codebooks = torch.arange(24, dtype=torch.long).reshape(3, 8)
    hidden_states = torch.ones(4, 6)
    upstream_outputs = iter(
        [
            _output({"rvq_codes": codebooks, "speaker_embedding": torch.ones(5)}),
            _output({"policy_tokens": [31, 32, 33, 34], "thinker_hidden_states": hidden_states}),
        ]
    )

    async def upstream_run(_self, sampling_params, **kwargs):
        assert sampling_params == {"temperature": 0.8}
        assert kwargs["raw_prompt"][0]["content"] == "hello"
        return next(upstream_outputs)

    monkeypatch.setattr(SingleTurnAgentLoop, "run", upstream_run)

    class RvqAdapter:
        @classmethod
        def postprocess_agent_loop_output(cls, output, *, tokenizer, response_length):
            del tokenizer
            output.response_ids = output.extra_fields["rvq_codes"][:response_length, 0].tolist()
            output.response_mask = [1] * len(output.response_ids)
            output.response_logprobs = [-0.2] * len(output.response_ids)
            return output

    class HiddenStateAdapter:
        @classmethod
        def postprocess_agent_loop_output(cls, output, *, tokenizer, response_length):
            del tokenizer
            output.response_ids = output.extra_fields["policy_tokens"][:response_length]
            output.response_mask = [1] * len(output.response_ids)
            output.response_logprobs = None
            return output

    loop = object.__new__(OmniSingleTurnAgentLoop)
    loop.response_length = 4
    loop.tokenizer = SimpleNamespace()

    loop.rollout_adapter = RvqAdapter
    rvq_output = await loop.run({"temperature": 0.8}, raw_prompt=[{"role": "user", "content": "hello"}])
    loop.rollout_adapter = HiddenStateAdapter
    hidden_output = await loop.run({"temperature": 0.8}, raw_prompt=[{"role": "user", "content": "hello"}])

    assert rvq_output.response_ids == codebooks[:, 0].tolist()
    assert rvq_output.extra_fields["rvq_codes"].shape == (3, 8)
    assert hidden_output.response_ids == [31, 32, 33, 34]
    assert hidden_output.extra_fields["thinker_hidden_states"].shape == (4, 6)

    class RvqTrainingAdapter(OmniModelBase):
        @classmethod
        def prepare_model_inputs(cls, model_inputs, micro_batch, model_config):
            del model_config
            return {**model_inputs, "rvq_codes": micro_batch["extra_fields"][0]["rvq_codes"]}

    class HiddenStateTrainingAdapter(OmniModelBase):
        @classmethod
        def prepare_model_inputs(cls, model_inputs, micro_batch, model_config):
            del model_config
            return {
                **model_inputs,
                "thinker_hidden_states": micro_batch["extra_fields"][0]["thinker_hidden_states"],
            }

    rvq_inputs = RvqTrainingAdapter.prepare_model_inputs(
        {"input_ids": torch.ones(1, 3, dtype=torch.long)},
        {"extra_fields": [rvq_output.extra_fields]},
        SimpleNamespace(),
    )
    hidden_inputs = HiddenStateTrainingAdapter.prepare_model_inputs(
        {"input_ids": torch.ones(1, 4, dtype=torch.long)},
        {"extra_fields": [hidden_output.extra_fields]},
        SimpleNamespace(),
    )
    assert rvq_inputs["rvq_codes"].shape == (3, 8)
    assert hidden_inputs["thinker_hidden_states"].shape == (4, 6)


@pytest.mark.asyncio
async def test_omni_single_turn_agent_rejects_misaligned_policy_output(monkeypatch):
    async def upstream_run(_self, sampling_params, **kwargs):
        return _output({"trajectory": torch.ones(2, 2)})

    monkeypatch.setattr(SingleTurnAgentLoop, "run", upstream_run)

    class MisalignedAdapter:
        @classmethod
        def postprocess_agent_loop_output(cls, output, *, tokenizer, response_length):
            output.response_ids = [1, 2]
            output.response_mask = [1]
            return output

    loop = object.__new__(OmniSingleTurnAgentLoop)
    loop.rollout_adapter = MisalignedAdapter
    loop.response_length = 4
    loop.tokenizer = SimpleNamespace()

    with pytest.raises(ValueError, match="response_mask must align"):
        await loop.run({"temperature": 0.8}, raw_prompt=[{"role": "user", "content": "hello"}])
