from components.model import RainForCausalLM
from components.model_config import RainLLMConfig

model = RainForCausalLM(RainLLMConfig(
    dim=256,
    n_layers=1
))
model2 = RainForCausalLM(RainLLMConfig(
    dim=256,
    n_layers=1,
    ffn_type="conv"
))

print(f"1总参数量：{sum(p.numel() for p in model.parameters() if p.requires_grad)}")
print(f"2总参数量：{sum(p.numel() for p in model2.parameters() if p.requires_grad)}")