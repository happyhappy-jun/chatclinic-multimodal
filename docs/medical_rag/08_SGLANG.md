# 08 · Serving Qwen3-8B with SGLang

SGLang is a drop-in alternative to vLLM for the model server. The backend only speaks the
**OpenAI-compatible `/v1` API**, so switching frameworks changes **nothing in the app** — only
`LOCAL_LLM_BASE_URL`.

## 1. Install SGLang (in a SEPARATE env — recommended)

> ⚠️ **Do not `pip install sglang[all]` into the `chatclinic` env.** SGLang pulls a CUDA-13 torch that
> overwrites the pinned `torch 2.5.1+cu121` and breaks the env (`OSError: libcudart.so.13`), which also
> breaks vLLM and the sentence-transformers embedder. The backend reaches the model only over HTTP, so
> the serving framework belongs in its own env.

```bash
conda create -y -n sglang python=3.10
conda activate sglang
pip install "sglang[all]"
python -c "import sglang; print('sglang', sglang.__version__)"
```

(If you already broke `chatclinic`, repair it: `pip uninstall -y sglang sgl-kernel flashinfer-python
torch torchvision torchaudio && pip install torch==2.5.1 torchvision==0.20.1 --index-url
https://download.pytorch.org/whl/cu121`.)

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
