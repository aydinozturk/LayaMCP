# Laya MCP

[English](README.md) | **Türkçe**

[convaiinnovations/laya](https://huggingface.co/convaiinnovations/laya) karar modelini herhangi bir MCP istemcisine (Claude Code, Claude Desktop, Cursor, VS Code, …) araç olarak sunan MCP sunucusu.

Laya metin üretmez. Bir **state** (metin veya JSON) ve **tipli sorular** alır, tek forward pass'te her soruya cevap ve kalibre edilmiş olasılık döner. 100+ dil desteklenir; Türkçe gibi İngilizce olmayan metinler otomatik olarak `multilingual` checkpoint'ine yönlendirilir. Tipik kullanım alanları: sınıflandırma, ticket/e-posta triage, LLM guardrail (jailbreak / prompt injection), moderasyon, istek yönlendirme.

## Araçlar

| Araç | Ne yapar |
|---|---|
| `laya_classify` | Metni verilen etiketlerden birine atar (`labels`: liste veya `{etiket: açıklama}`) |
| `laya_yes_no` | Evet/hayır sorusu sorar, `p_yes` döner |
| `laya_score` | Sıralı ölçekte puanlar (ör. `["sakin", "sinirli", "öfkeli"]`) |
| `laya_decide` | Genel kullanım: birden çok `choice` / `score` / `noul` sorusunu tek çağrıda cevaplar |
| `laya_preset` | Hazır soru setleri: `triage`, `email`, `guard`, `moderation`, `router` |
| `laya_route` | Metnin hangi checkpoint'e gideceğini ve nedenini gösterir (model çalıştırmaz) |
| `laya_status` | Bellekteki checkpoint'ler ve yapılandırma |

Resource: `laya://presets/{name}`, bir preset'in soru tanımlarını döner. `laya_decide` için şablon olarak kullanılabilir.

## Docker imajları

| Etiket | Platform | Ne için |
|---|---|---|
| `aydinozturk/laya-mcp:cuda` | `linux/amd64` | NVIDIA GPU'lu sunucu (CUDA 12.6, sürücü ≥ 525) |
| `aydinozturk/laya-mcp:latest` | `linux/amd64`, `linux/arm64` | CPU (laptop, GPU'suz sunucu, stdio kullanımı) |

Sürüm sabitlemek için `0.1.1-cuda` ve `0.1.1` etiketleri de var. İki imajda da `english` ve `multilingual` checkpoint'lerinin ağırlıkları (~1.5 GB) gömülüdür, container internetsiz açılır. Cevaplar iki imajda da aynıdır; GPU sadece hız kazandırır (tek soru T4'te ~35 ms, CPU'da ~200–450 ms).

Kendin build etmek için:

```bash
docker buildx build --platform linux/amd64 \
  --build-arg TORCH_INDEX=https://download.pytorch.org/whl/cu126 \
  -t aydinozturk/laya-mcp:cuda --push .
docker buildx build --platform linux/amd64,linux/arm64 -t aydinozturk/laya-mcp:latest --push .
```

| Build argümanı | Varsayılan | Açıklama |
|---|---|---|
| `TORCH_INDEX` | CPU wheel index | GPU için `https://download.pytorch.org/whl/cu126` |
| `LAYA_BAKE_MODELS` | `english,multilingual` | İmaja gömülecek checkpoint'ler. Boşsa ilk kullanımda indirilir (`HF_HUB_OFFLINE=0` ile) |

## GPU sunucuda deployment

Sunucuda NVIDIA sürücüsü ve [nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) kurulu olmalı. Kontrol etmek için:

```bash
docker run --rm --gpus all nvidia/cuda:12.6.3-base-ubuntu24.04 nvidia-smi
```

Sonra sunucuya sadece [docker-compose.yml](docker-compose.yml) dosyasını kopyala:

```bash
MCP_API_KEY=gizli-anahtar docker compose up -d
curl localhost:8000/health    # {"status":"ok","loaded":[...],"devices":{"english":"cuda:0","multilingual":"cuda:0"}}
docker compose pull && docker compose up -d   # yeni sürüme güncelleme
```

- `MCP_API_KEY` zorunludur, tanımlı değilse compose başlamaz. Değerleri bir `.env` dosyasına da yazabilirsin.
- `LAYA_REQUIRE_GPU=1` varsayılan olarak açıktır. laya, CUDA'yı bulamazsa sessizce CPU'ya düşer; bu ayar sayesinde container açık bir hata mesajıyla kapanır. `docker compose logs` ile nedenini görebilirsin.
- Açılışta her checkpoint bir kez ısıtılır, böylece ilk istek CUDA başlatma maliyetini ödemez.
- İsteğe bağlı değişkenler: `LAYA_MCP_PORT` (8000), `LAYA_GPU` (GPU indeksi, ör. `0`; varsayılan `all`), `LAYA_GPU_COUNT` (1), `LAYA_MODELS`, `LAYA_DEFAULT`.
- VRAM: ağırlıklar GPU'da fp32 tutulur, hesap fp16 autocast ile yapılır. İki checkpoint yaklaşık 3–4 GB VRAM kullanır. Üçünü yüklemek için `LAYA_MODELS=english,multilingual,typed-decisions` ver; `typed-decisions` imajda gömülü olmadığından ilk açılışta indirilir, bunun için `HF_HUB_OFFLINE=0` da ekle.

GPU'suz bir makinede aynı compose'u CPU imajıyla çalıştırmak için:

```bash
MCP_API_KEY=gizli-anahtar docker compose -f docker-compose.yml -f docker-compose.cpu.yml up -d
```

## Projelerde kullanım

### Claude Code (stdio, proje bazında)

Projenin köküne `.mcp.json` ekleyin:

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

Ya da komut satırından ekleyin (`-s user` tüm projelerde geçerli olur):

```bash
claude mcp add laya -s user -- docker run -i --rm -e LAYA_THREADS=4 aydinozturk/laya-mcp:latest
```

stdio modunda her oturum kendi container'ını açar ve ilk çağrıda modeli belleğe yükler (CPU'da ~2–10 sn). GPU sunucun varsa aşağıdaki HTTP bağlantısı hem daha hızlı hem de tüm projeler için ortaktır.

### Paylaşımlı sunucuya bağlanma (HTTP)

[GPU sunucuda deployment](#gpu-sunucuda-deployment) bölümündeki gibi çalışan sunucuya bağlan:

```bash
claude mcp add --transport http laya -s user http://<sunucu>:8000/mcp --header "Authorization: Bearer gizli-anahtar"
```

`.mcp.json` ile:

```json
{
  "mcpServers": {
    "laya": {
      "type": "http",
      "url": "http://<sunucu>:8000/mcp",
      "headers": { "Authorization": "Bearer ${LAYA_MCP_KEY}" }
    }
  }
}
```

### Claude Desktop / Cursor

`claude_desktop_config.json` veya `~/.cursor/mcp.json` dosyasındaki `mcpServers` altına, Claude Code bölümündeki stdio bloğunu olduğu gibi ekleyin.

## Ortam değişkenleri

| Değişken | Varsayılan | Açıklama |
|---|---|---|
| `MCP_TRANSPORT` | `stdio` | `stdio`, `http` (streamable HTTP, `/mcp`) veya `sse` (`/sse`) |
| `MCP_HOST` / `MCP_PORT` | `0.0.0.0` / `8000` | HTTP bağlama adresi |
| `MCP_API_KEY` | – | Tanımlıysa HTTP istekleri `Authorization: Bearer <key>` ister |
| `LAYA_PRELOAD` | `0` | `1` ise checkpoint'ler açılışta yüklenir (compose'da açık) |
| `LAYA_MODELS` | `english,multilingual` | Preload edilecek checkpoint'ler |
| `LAYA_MAX_LOADED` | `2` | Bellekte aynı anda tutulacak checkpoint sayısı (LRU) |
| `LAYA_DEVICE` | otomatik | `cpu` veya `cuda` (GPU compose'da `cuda`) |
| `LAYA_REQUIRE_GPU` | `0` | `1` ise checkpoint CUDA'da değilse container kapanır (GPU compose'da `1`) |
| `LAYA_THREADS` | torch varsayılanı | CPU thread sınırı; fiziksel çekirdek sayısını aşmayın |
| `LAYA_DEFAULT` | `english` | Dili tespit edilemeyen kısa Latin metinler için checkpoint. Çoğunlukla Türkçe trafikte `multilingual` yapın |

Bellek: CPU'da iki checkpoint yüklüyken container yaklaşık 3–4 GB RAM kullanır. `laya_status` aracı ve `/health` her checkpoint'in gerçekte hangi cihazda çalıştığını (`devices`) gösterir.

## Örnek çağrılar

```jsonc
// laya_classify
{ "text": "Faturam iki kez kesildi, iade istiyorum.",
  "labels": { "billing": "fatura, ödeme, iade", "technical": "hata, çökme", "other": "diğer" } }
// -> { "label": "billing", "confidence": 0.9, "probabilities": {...}, "routing": {"model": "multilingual", ...} }

// laya_decide
{ "state": { "subject": "Çift ödeme", "body": "Bugün iade etmezseniz aboneliği iptal ediyoruz." },
  "questions": {
    "churn":   { "type": "noul",  "instructions": "Does the user threaten to cancel?" },
    "urgency": { "type": "score", "instructions": "How urgent is `body`?", "criteria": ["not urgent", "soon", "critical"] } } }
```

Soru tipleri:

- **choice**: `criteria` bir `{anahtar: açıklama}` sözlüğüdür. 20'den az seçenekte iyi çalışır.
- **score**: `criteria` düşükten yükseğe sıralı seviye açıklamalarından oluşan bir listedir. Beklenen seviye indeksini ondalık sayı olarak döner.
- **noul**: evet olasılığını (0..1) döner. İngilizce checkpoint'te etiket yanlılığı olabildiği için `laya_yes_no` aracı tercih edilmelidir. Bu araç aynı soruyu nötr iki seçenekli bir `choice` olarak sorar.

## Sınırlamalar (upstream model kartından)

- Base checkpoint'ler zero-shot'ta **aşırı özgüvenli** çalışır. Olasılıkları mutlak değer olarak değil göreli olarak yorumlayın ve kararlarınızı `confidence` alanına göre verin. Upstream'in `action.act_probability` alanı anlamlı sinyal taşımadığı için çıktıdan çıkarılmıştır.
- 20'den fazla seçenekli `choice` sorularında doğruluk belirgin biçimde düşer. Seçenekleri iki aşamalı (kaba → ince) sorulara bölün.
- En zayıf soru tipi `score`'dur.
- `typed-decisions` checkpoint'i yalnızca `model: "typed-decisions"` ile açıkça seçildiğinde kullanılır. Varsayılan imajda gömülü değildir. İmaj offline modda (`HF_HUB_OFFLINE=1`) çalıştığı için kullanmak istersen ya `LAYA_BAKE_MODELS` ile imaja ekle ya da container'ı `-e HF_HUB_OFFLINE=0` ile başlat.

## Geliştirme

```bash
python3.12 -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"
pytest                          # model yüklemeden araç testleri
python scripts/smoke.py         # Docker imajıyla gerçek model üzerinde uçtan uca test
```

## Lisans

Apache 2.0, Laya modeliyle aynı.
