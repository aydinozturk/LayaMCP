# Laya MCP

**English** | [Türkçe](README.tr.md)

An MCP server that exposes the [convaiinnovations/laya](https://huggingface.co/convaiinnovations/laya) decision model as tools for any MCP client (Claude Code, Claude Desktop, Cursor, VS Code, …).

Laya does not generate text. You give it a **state** (text or JSON) and **typed questions**, and it answers every question with a calibrated probability in a single forward pass. It supports 100+ languages, and non-English text is routed to the `multilingual` checkpoint automatically. Typical uses are classification, ticket and email triage, LLM guardrails (jailbreak and prompt injection detection), content moderation and request routing.

## Tools

| Tool | What it does |
|---|---|
| `laya_classify` | Assigns text to one of the given labels (`labels`: a list or `{label: description}`) |
| `laya_yes_no` | Asks a yes/no question and returns `p_yes` |
| `laya_score` | Rates text on an ordinal scale (e.g. `["calm", "annoyed", "furious"]`) |
| `laya_decide` | General purpose: answers several `choice` / `score` / `noul` questions in one call |
| `laya_preset` | Ready-made question sets: `triage`, `email`, `guard`, `moderation`, `router` |
| `laya_route` | Shows which checkpoint a text would go to and why (runs no model) |
| `laya_status` | Loaded checkpoints, the device each one runs on, and configuration |

Resource: `laya://presets/{name}` returns a preset's question definitions, which you can use as a template for `laya_decide`.

## Docker images

| Tag | Platform | For |
|---|---|---|
| `aydinozturk/laya-mcp:cuda` | `linux/amd64` | Servers with an NVIDIA GPU (CUDA 12.6, driver ≥ 525) |
| `aydinozturk/laya-mcp:latest` | `linux/amd64`, `linux/arm64` | CPU (laptops, GPU-less servers, stdio use) |

`0.1.1-cuda` and `0.1.1` tags are available for pinning. Both images bake in the weights of the `english` and `multilingual` checkpoints (~1.5 GB), so containers start without network access. Both images give the same answers; the GPU only makes them faster (about 35 ms per question on a T4, about 200–450 ms on CPU).

To build them yourself:

```bash
docker buildx build --platform linux/amd64 \
  --build-arg TORCH_INDEX=https://download.pytorch.org/whl/cu126 \
  -t aydinozturk/laya-mcp:cuda --push .
docker buildx build --platform linux/amd64,linux/arm64 -t aydinozturk/laya-mcp:latest --push .
```

| Build argument | Default | Description |
|---|---|---|
| `TORCH_INDEX` | CPU wheel index | For GPU: `https://download.pytorch.org/whl/cu126` |
| `LAYA_BAKE_MODELS` | `english,multilingual` | Checkpoints to bake into the image. If empty, they are downloaded on first use (together with `HF_HUB_OFFLINE=0`) |

## Deploying on a GPU server

The server needs the NVIDIA driver and the [nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html). To check that Docker can see the GPU:

```bash
docker run --rm --gpus all nvidia/cuda:12.6.3-base-ubuntu24.04 nvidia-smi
```

Then copy [docker-compose.yml](docker-compose.yml) to the server. It is the only file you need:

```bash
MCP_API_KEY=your-secret docker compose up -d
curl localhost:8000/health    # {"status":"ok","loaded":[...],"devices":{"english":"cuda:0","multilingual":"cuda:0"}}
docker compose pull && docker compose up -d   # update to the latest image
```

- `MCP_API_KEY` is required; compose refuses to start without it. You can also put the values in a `.env` file.
- `LAYA_REQUIRE_GPU=1` is on by default. laya silently falls back to CPU when it cannot use CUDA; with this setting the container exits with a clear error instead. Run `docker compose logs` to see why.
- Every checkpoint is warmed up once at startup, so the first request does not pay for CUDA initialisation.
- Optional variables: `LAYA_MCP_PORT` (8000), `LAYA_GPU` (GPU index, e.g. `0`; default `all`), `LAYA_GPU_COUNT` (1), `LAYA_MODELS`, `LAYA_DEFAULT`.
- VRAM: weights are kept in fp32 on the GPU and inference runs under fp16 autocast. Two checkpoints use about 3–4 GB of VRAM. To load all three, set `LAYA_MODELS=english,multilingual,typed-decisions`. `typed-decisions` is not baked into the image and is downloaded on first start, so also set `HF_HUB_OFFLINE=0`.

To run the same compose file with the CPU image on a machine without a GPU:

```bash
MCP_API_KEY=your-secret docker compose -f docker-compose.yml -f docker-compose.cpu.yml up -d
```

## Using it in your projects

### Claude Code (stdio, per project)

Add a `.mcp.json` file to the project root:

```json
{
  "mcpServers": {
    "laya": {
      "command": "docker",
      "args": ["run", "-i", "--rm", "-e", "LAYA_THREADS=4", "aydinozturk/laya-mcp:latest"]
    }
  }
}
```

Or add it from the command line (`-s user` makes it available in every project):

```bash
claude mcp add laya -s user -- docker run -i --rm -e LAYA_THREADS=4 aydinozturk/laya-mcp:latest
```

In stdio mode every session starts its own container and loads the model on the first call (about 2–10 s on CPU). If you have a GPU server, the HTTP connection below is faster and shared across all projects.

### Connecting to a shared server (HTTP)

Connect to a server running as described in [Deploying on a GPU server](#deploying-on-a-gpu-server):

```bash
claude mcp add --transport http laya -s user http://<server>:8000/mcp --header "Authorization: Bearer your-secret"
```

With `.mcp.json`:

```json
{
  "mcpServers": {
    "laya": {
      "type": "http",
      "url": "http://<server>:8000/mcp",
      "headers": { "Authorization": "Bearer ${LAYA_MCP_KEY}" }
    }
  }
}
```

### Claude Desktop / Cursor

Add the stdio block from the Claude Code section under `mcpServers` in `claude_desktop_config.json` or `~/.cursor/mcp.json`.

## Environment variables

| Variable | Default | Description |
|---|---|---|
| `MCP_TRANSPORT` | `stdio` | `stdio`, `http` (streamable HTTP, `/mcp`) or `sse` (`/sse`) |
| `MCP_HOST` / `MCP_PORT` | `0.0.0.0` / `8000` | HTTP bind address |
| `MCP_API_KEY` | – | If set, HTTP requests must send `Authorization: Bearer <key>` |
| `LAYA_PRELOAD` | `0` | `1` loads the checkpoints at startup (on in compose) |
| `LAYA_MODELS` | `english,multilingual` | Checkpoints to preload |
| `LAYA_MAX_LOADED` | `2` | How many checkpoints stay in memory at once (LRU) |
| `LAYA_DEVICE` | auto | `cpu` or `cuda` (`cuda` in the GPU compose file) |
| `LAYA_REQUIRE_GPU` | `0` | `1` makes the container exit if a checkpoint is not on CUDA (`1` in the GPU compose file) |
| `LAYA_THREADS` | torch default | CPU thread cap; keep it at or below the number of physical cores |
| `LAYA_DEFAULT` | `english` | Checkpoint for short Latin-script text whose language cannot be detected. Set it to `multilingual` if most of your traffic is not English |

Memory: with two checkpoints loaded on CPU, the container uses about 3–4 GB of RAM. The `laya_status` tool and `/health` show the device each checkpoint actually runs on (`devices`).

## Example calls

```jsonc
// laya_classify
{ "text": "I was billed twice for March, please refund the duplicate.",
  "labels": { "billing": "invoices, payments, refunds", "technical": "bugs, outages", "other": "everything else" } }
// -> { "label": "billing", "confidence": 0.95, "probabilities": {...}, "routing": {"model": "english", ...} }

// laya_decide
{ "state": { "subject": "Duplicate charge", "body": "Refund it today or we cancel our plan." },
  "questions": {
    "churn":   { "type": "noul",  "instructions": "Does the user threaten to cancel?" },
    "urgency": { "type": "score", "instructions": "How urgent is `body`?", "criteria": ["not urgent", "soon", "critical"] } } }
```

Question types:

- **choice**: `criteria` is a `{key: description}` dictionary. Works well with fewer than 20 options.
- **score**: `criteria` is a list of level descriptions ordered from low to high. Returns the expected level index as a float.
- **noul**: returns the probability of "yes" (0..1). The English checkpoint can be biased by the option labels, so prefer the `laya_yes_no` tool, which asks the same question as a neutral two-option `choice`.

## Limitations (from the upstream model card)

- The base checkpoints are **over-confident** zero-shot. Read the probabilities as relative rather than absolute, and base decisions on the `confidence` field. The upstream `action.act_probability` field carries no useful signal, so this server drops it from its output.
- Accuracy drops sharply for `choice` questions with more than 20 options. Split them into a two-step coarse-to-fine question.
- `score` is the weakest question type.
- The `typed-decisions` checkpoint is only used when selected explicitly with `model: "typed-decisions"`. It is not baked into the default images, and the images run in offline mode (`HF_HUB_OFFLINE=1`), so either add it with `LAYA_BAKE_MODELS` or start the container with `-e HF_HUB_OFFLINE=0`.

## Development

```bash
python3.12 -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"
pytest                          # tool tests without loading the model
python scripts/smoke.py         # end-to-end test against the real model in the Docker image
```

## License

Apache 2.0, the same as the Laya model.
