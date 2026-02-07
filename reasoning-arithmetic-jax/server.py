import asyncio
import json
from concurrent.futures import ThreadPoolExecutor

import jax
import jax.numpy as jnp
import yaml
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from train.grpo_train_config import GRPOTrainConfig
from utils.functions import logits_from_ids
from utils.loading import load_model_and_tokenizer_with_vocab_change, load_params

app = FastAPI(title="LLM Comparison")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# Load model
with open("configs/grpo_train.yaml", "r") as f:
    config_dict = yaml.safe_load(f)
cfg = GRPOTrainConfig(**config_dict)

model, tokenizer = load_model_and_tokenizer_with_vocab_change(cfg.model_id)
params_sft = load_params(cfg.save_dir + "/sft", model.params)
params_grpo = load_params(cfg.save_dir + "/grpo", model.params)
pad_id = tokenizer.pad_token_id

MODELS = {"sft": params_sft, "grpo": params_grpo}


class PromptRequest(BaseModel):
    prompt: str
    model_type: str


executor = ThreadPoolExecutor(max_workers=2)


async def token_generator(prompt: str, model_type: str):
    """streaming"""
    if model_type not in MODELS:
        yield "data: [ERROR]: Not found\n\n"

    # Sampling
    ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    prompt_len = len(ids)

    padded_input = jnp.full((1, cfg.max_len), pad_id, dtype=jnp.int32)
    n = cfg.max_len - len(ids)

    loop = asyncio.get_event_loop()

    for _ in range(n):
        cur_len = len(ids)
        padded_input = padded_input.at[0, :cur_len].set(jnp.asarray(ids, dtype=jnp.int32))

        def inference_step() -> jax.Array:
            logits = logits_from_ids(model, MODELS[model_type], padded_input, pad_id)[0, cur_len - 1]  # (B, V,)

            return logits

        logits = await loop.run_in_executor(executor, inference_step)

        next_id = int(jnp.argmax(logits))

        if next_id == int(tokenizer.eos_token_id):
            break

        ids.append(next_id)

        data = tokenizer.decode(ids[prompt_len:], skip_special_tokens=False)
        yield f"data: {json.dumps({'token': data})}\n\n"
        await asyncio.sleep(0.001)

    # Sampling end
    yield "\n"
    yield "data: [Done]\n\n"


@app.post("/run")
async def run_models(request: PromptRequest) -> StreamingResponse:
    return StreamingResponse(token_generator(request.prompt, request.model_type), media_type="text/event-stream")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
