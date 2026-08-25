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
"""Import-time compatibility for qwen-tts 0.1.1 on Transformers 5.x."""

from contextlib import contextmanager
from functools import wraps

import torch
from packaging.version import Version


def _default_rope_init(config, device=None, **kwargs):
    del kwargs
    base = getattr(config, "rope_theta", 10_000.0) or 10_000.0
    dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
    positions = torch.arange(0, dim, 2, dtype=torch.int64, device=device).float() / dim
    return 1.0 / (base**positions), 1.0


def _compatible_mask(original):
    @wraps(original)
    def wrapper(*args, input_embeds=None, inputs_embeds=None, cache_position=None, **kwargs):
        del cache_position
        embeddings = inputs_embeds if inputs_embeds is not None else input_embeds
        return original(*args, inputs_embeds=embeddings, **kwargs)

    return wrapper


def patch_qwen3_tts_config_defaults(config_cls) -> None:
    """Restore the config default that qwen-tts expects from Transformers 4.x."""
    talker_config_cls = getattr(config_cls, "sub_configs", {}).get("talker_config")
    if talker_config_cls is not None and not hasattr(talker_config_cls, "pad_token_id"):
        talker_config_cls.pad_token_id = None


@contextmanager
def qwen3_tts_import_context():
    """Expose the TF5 APIs expected while qwen-tts binds its imports.

    qwen-tts modules retain the mask wrappers they import. The global
    Transformers functions are restored immediately afterwards.
    """
    import transformers

    if Version(transformers.__version__).major < 5:
        yield
        return

    import transformers.masking_utils as masking_utils
    import transformers.utils.generic as generic_utils
    from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

    original_check = generic_utils.check_model_inputs
    original_causal_mask = masking_utils.create_causal_mask
    original_sliding_mask = masking_utils.create_sliding_window_causal_mask

    def compatible_check_model_inputs(func=None):
        return original_check if func is None else original_check(func)

    generic_utils.check_model_inputs = compatible_check_model_inputs
    masking_utils.create_causal_mask = _compatible_mask(original_causal_mask)
    masking_utils.create_sliding_window_causal_mask = _compatible_mask(original_sliding_mask)
    ROPE_INIT_FUNCTIONS.setdefault("default", _default_rope_init)
    try:
        yield
    finally:
        generic_utils.check_model_inputs = original_check
        masking_utils.create_causal_mask = original_causal_mask
        masking_utils.create_sliding_window_causal_mask = original_sliding_mask
