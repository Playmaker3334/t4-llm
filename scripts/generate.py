import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.model import ModelConfig
from configs.data import DataConfig
from llm125m.model import Transformer
from llm125m.data.tokenizer import load_tokenizer
from llm125m.utils.logging import setup_logger


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--prompt", type=str, default="The history of")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--n-samples", type=int, default=3)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    logger = setup_logger("generate")

    mcfg = ModelConfig()
    dcfg = DataConfig()

    logger.info(f"loading_checkpoint path={args.ckpt}")
    state = torch.load(args.ckpt, map_location=args.device)
    model = Transformer(mcfg).to(args.device)
    model.load_state_dict(state["model"])
    model.eval()

    n_params = model.num_parameters()
    logger.info(f"model_loaded params={n_params:,} step={state.get('global_step', 0)}")

    tokenizer = load_tokenizer(dcfg.tokenizer_path)

    for i in range(args.n_samples):
        ids = tokenizer.encode(args.prompt, return_tensors="pt").to(args.device)
        out = model.generate(
            ids,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            eos_token_id=tokenizer.eos_token_id,
        )
        text = tokenizer.decode(out[0].tolist(), skip_special_tokens=True)
        n_new = out.size(1) - ids.size(1)
        logger.info(f"sample={i+1} new_tokens={n_new}")
        print(f"=== sample {i+1}/{args.n_samples} ===")
        print(text)
        print()


if __name__ == "__main__":
    main()
