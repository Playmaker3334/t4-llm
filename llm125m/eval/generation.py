import torch


@torch.no_grad()
def generate_text(model, tokenizer, prompt: str, max_new_tokens: int = 128,
                  temperature: float = 0.8, top_k: int = 50, top_p: float = 0.9,
                  device: str = "cuda") -> str:
    model.eval()
    ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    out_ids = model.generate(
        ids,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        eos_token_id=tokenizer.eos_token_id,
    )
    return tokenizer.decode(out_ids[0].tolist(), skip_special_tokens=True)
