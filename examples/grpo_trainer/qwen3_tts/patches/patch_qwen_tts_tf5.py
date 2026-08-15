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
"""Patch qwen-tts 0.1.1 for the Transformers version used by verl-omni."""

import importlib.util
import pathlib
import re
import sys


def _package_dir() -> pathlib.Path:
    if len(sys.argv) > 1:
        return pathlib.Path(sys.argv[1])
    spec = importlib.util.find_spec("qwen_tts")
    if spec is None or not spec.origin:
        raise RuntimeError("qwen_tts is not installed")
    return pathlib.Path(spec.origin).parent


_ROPE_HELPER = (
    "def _qtts_default_rope_init(config, device=None, **kwargs):\n"
    "    import torch\n"
    "    base = getattr(config, 'rope_theta', 10000.0) or 10000.0\n"
    "    dim = getattr(config, 'head_dim', None) or (config.hidden_size // config.num_attention_heads)\n"
    "    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.int64)"
    ".to(device=device, dtype=torch.float) / dim))\n"
    "    return inv_freq, 1.0\n\n\n"
)
_CACHE_ANCHOR = (
    "                inputs_embeds = inputs_embeds + tts_pad_embed\n        if attention_mask is not None:\n"
)
_CACHE_FIX = (
    "                inputs_embeds = inputs_embeds + tts_pad_embed\n"
    "        if cache_position is None and past_key_values is not None:\n"
    "            past_seen_tokens = past_key_values.get_seq_length()\n"
    "            cache_position = torch.arange(past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], "
    "device=inputs_embeds.device)\n"
    "        if attention_mask is not None:\n"
)


def patch_file(path: pathlib.Path) -> None:
    if not path.exists():
        print(f"skip (missing) {path.name}")
        return
    source = original = path.read_text()
    source = source.replace("@check_model_inputs()", "@check_model_inputs")
    source = source.replace("config.pad_token_id", 'getattr(config, "pad_token_id", None)')
    if "ROPE_INIT_FUNCTIONS[self.rope_type]" in source:
        if "_qtts_default_rope_init" not in source:
            anchor = source.find("\nlogger = logging.get_logger(__name__)\n")
            insert_at = anchor + len("\nlogger = logging.get_logger(__name__)\n") + 1 if anchor != -1 else 0
            if not insert_at:
                match = re.search(r"^(?:@.*\n)*class ", source, re.MULTILINE)
                insert_at = match.start() if match else 0
            source = source[:insert_at] + "\n" + _ROPE_HELPER + source[insert_at:]
        source = source.replace(
            "ROPE_INIT_FUNCTIONS[self.rope_type]",
            "(ROPE_INIT_FUNCTIONS.get(self.rope_type) or _qtts_default_rope_init)",
        )
    source = source.replace('"input_embeds": inputs_embeds,', '"inputs_embeds": inputs_embeds,')
    source = source.replace(
        '"attention_mask": attention_mask,\n                "cache_position": cache_position,',
        '"attention_mask": attention_mask,',
    )
    source = source.replace(
        "            input_embeds=inputs_embeds,\n"
        "            attention_mask=attention_mask,\n"
        "            cache_position=cache_position,\n",
        "            inputs_embeds=inputs_embeds,\n            attention_mask=attention_mask,\n",
    )
    if "past_seen_tokens = past_key_values.get_seq_length()" not in source:
        source = source.replace(_CACHE_ANCHOR, _CACHE_FIX, 1)
    if source != original:
        path.write_text(source)
        print(f"patched {path.name}")
    else:
        print(f"nochange {path.name}")


def main() -> None:
    package_dir = _package_dir()
    for path in (
        package_dir / "core/models/modeling_qwen3_tts.py",
        package_dir / "core/tokenizer_12hz/modeling_qwen3_tts_tokenizer_v2.py",
    ):
        patch_file(path)
    print("PATCH_DONE")


if __name__ == "__main__":
    main()
