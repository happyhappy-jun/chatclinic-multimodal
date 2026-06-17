# 08 · Serving Qwen3-8B with SGLang

SGLang is a drop-in alternative to vLLM for the model server. The backend only speaks the
**OpenAI-compatible `/v1` API**, so switching frameworks changes **nothing in the app** — only
`LOCAL_LLM_BASE_URL`.

## 1. Install SGLang (pip, in the conda env)

```bash
conda activate chatclinic
pip install "sglang[all]"
```

Verify it imports:
```bash
python -c "import sglang; print('sglang', sglang.__version__)"
```

> Note: `sglang[all]` may pull its own torch/flashinfer build via pip. If you want to keep the pinned
> `vllm==0.18.1` working in the same env, install SGLang in a separate env instead. Since the backend
> reaches the model only over HTTP, the serving env is independent of the backend.

## 2. Serve

```bash
bash scripts/serve_qwen3_sglang.sh
# defaults: Qwen/Qwen3-8B, port 30000, TP=1, max_len 12288, mem_frac 0.85
# multi-GPU:  TP=2 bash scripts/serve_qwen3_sglang.sh
# OOM on 24 GB:  MAX_LEN=8192 bash scripts/serve_qwen3_sglang.sh
```

The script sets `PYTHONNOUSERSITE=1` and redirects compile caches if `/tmp` is `noexec`. Wait for the
SGLang server to report it is ready on `http://0.0.0.0:30000`.

## 3. Point the backend at it

```bash
# .env  (only this line differs from the vLLM setup — note port 30000)
LOCAL_LLM_BASE_URL=http://localhost:30000/v1
LOCAL_LLM_MODEL=qwen3-8b
```

Then bring up the rest as usual:
```bash
bash scripts/run_all.sh
```

## 4. Verify it's live (not the extractive fallback)

```bash
curl -s localhost:30000/v1/models | grep -o qwen3-8b
curl -s -X POST localhost:8001/api/v1/guideline-rag/run \
  -H 'Content-Type: application/json' \
  -d '{"question":"vasopressor in septic shock?","top_k":3}' \
  | python -c "import sys,json;d=json.load(sys.stdin);print('fallback:',d['used_fallback'],'| model:',d['model'])"
```
`fallback: False | model: qwen3-8b` ⇒ SGLang is serving end-to-end.

## Notes
- Qwen3 runs **non-thinking** by default (the client sends `chat_template_kwargs={"enable_thinking":false}`,
  which SGLang honors). To serve thinking mode, add `--reasoning-parser qwen3` in the serve script.
- 3090 is Ampere (sm_86) — supported by SGLang. If a flashinfer kernel wheel is missing for your
  torch/CUDA, follow SGLang's install notes for the matching flashinfer wheel.
- Everything else (retrieval, MCP gateway, live PubMed, frontend) is identical to the vLLM path.
