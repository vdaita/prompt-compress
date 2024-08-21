from importlib.metadata import version
import warnings
import transformers
from snapkv.monkeypatch.llama_hijack_4_37 import llama_flash_attn2_forward as llama_flash_attn2_forward_4_37, prepare_inputs_for_generation_llama as prepare_inputs_for_generation_llama_4_37

def check_version():
    try:
        transformers_version = version("transformers")
    except Exception as e:
        print(f"Transformers not installed: {e}")
    return transformers_version

def replace_llama():
    transformers_version = check_version()
    version_list = ['4.37']
    warning_flag = True
    for version in version_list:
        if version in transformers_version:
            warning_flag = False
            break
    if warning_flag:
        warnings.warn(f"Transformers version {transformers_version} might not be compatible with SnapKV. SnapKV is tested with Transformers version {version_list}.")
    transformers.models.llama.modeling_llama.LlamaForCausalLM.prepare_inputs_for_generation = prepare_inputs_for_generation_llama_4_37
    transformers.models.llama.modeling_llama.LlamaFlashAttention2.forward = llama_flash_attn2_forward_4_37