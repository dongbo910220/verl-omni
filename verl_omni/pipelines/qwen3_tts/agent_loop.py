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
"""Single-turn agent loop that makes codec-0 the policy sequence."""

import torch
from verl.experimental.agent_loop.agent_loop import register
from verl.experimental.agent_loop.single_turn_agent_loop import SingleTurnAgentLoop

from verl_omni.pipelines.qwen3_tts.rollout_utils import is_evaluation_split, with_rollout_generation_seed
from verl_omni.pipelines.qwen3_tts.talker_forward import TEXT_PROMPT_TRAILER_TOKENS, build_assistant_text


@register("qwen3_tts_single_turn")
class Qwen3TTSSingleTurnAgentLoop(SingleTurnAgentLoop):
    async def run(self, sampling_params, **kwargs):
        extra_info = kwargs.get("extra_info")
        evaluation = is_evaluation_split(extra_info)
        candidate_count = self.rollout_config.val_kwargs.n if evaluation else self.rollout_config.n
        sampling_params = with_rollout_generation_seed(
            sampling_params,
            extra_info,
            session_id=kwargs.get("session_id"),
            global_steps=kwargs.get("global_steps"),
            uid=kwargs.get("uid"),
            base_seed=int(self.config.data.get("seed", 0)),
            require_session_id=int(candidate_count) > 1,
        )
        output = await super().run(sampling_params, **kwargs)
        extra = output.extra_fields
        codes, text = extra.get("tts_audio_codes"), extra.get("tts_text")
        if codes is None or text is None:
            raise RuntimeError("Qwen3-TTS rollout did not return codec codes and text.")
        codes = torch.as_tensor(codes, dtype=torch.long)
        if codes.ndim != 2 or codes.shape[-1] != 16:
            raise ValueError("Qwen3-TTS codec codes must have shape (frames, 16).")
        codes = codes[: self.response_length]
        policy_ids = codes[:, 0].tolist()
        if not policy_ids:
            raise RuntimeError("Qwen3-TTS rollout returned an empty codec trajectory.")
        if output.response_logprobs is not None:
            if len(output.response_logprobs) < len(policy_ids):
                raise RuntimeError("Qwen3-TTS rollout logprobs are shorter than the policy trajectory.")
            output.response_logprobs = output.response_logprobs[: len(policy_ids)]
        text_ids = self.tokenizer(build_assistant_text(str(text)), return_tensors="pt", padding=False)["input_ids"]
        extra["tts_text_ids"] = text_ids[:, :-TEXT_PROMPT_TRAILER_TOKENS].reshape(-1).tolist()
        extra["tts_audio_codes"] = codes
        output.prompt_ids = [0]
        output.response_ids = policy_ids
        output.response_mask = [1] * len(policy_ids)
        return output
