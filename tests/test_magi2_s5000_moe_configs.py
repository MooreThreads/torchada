import json
from pathlib import Path

CONFIG_DIR = (
    Path(__file__).parents[1] / "src/torchada/triton/autotune/fused_moe/configs/triton_3_2_0"
)
NAMES = (
    "E=768,N=1280,device_name=MTT_S5000.json",
    "E=768,N=1280,device_name=MTT_S5000_down.json",
)


def test_magi2_s5000_configs_have_tuned_runtime_shapes():
    for name in NAMES:
        data = json.loads((CONFIG_DIR / name).read_text())
        assert set(data) == {"21996", "24468"}
        for config in data.values():
            assert config == {
                "BLOCK_SIZE_M": 128,
                "BLOCK_SIZE_N": 128,
                "BLOCK_SIZE_K": 32,
                "GROUP_SIZE_M": 16,
                "num_warps": 16,
                "num_stages": 1,
            }
