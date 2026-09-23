# Laya MCP

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

## Docker imajı

Yayınlanan imaj: [`aydinozturk/laya-mcp`](https://hub.docker.com/r/aydinozturk/laya-mcp) (`linux/amd64` + `linux/arm64`).

```bash
docker pull aydinozturk/laya-mcp:latest
```

Kendin build edip yayınlamak için:

```bash
docker buildx build --platform linux/amd64,linux/arm64 \
  -t aydinozturk/laya-mcp:latest -t aydinozturk/laya-mcp:0.1.0 --push .
```

İmaj CPU sürümü torch ile gelir. `english` ve `multilingual` checkpoint'lerinin ağırlıkları (~1.5 GB) build sırasında indirilip imaja gömülür, yani container internetsiz de açılır. Build argümanları:

| Argüman | Varsayılan | Açıklama |
|---|---|---|
| `LAYA_BAKE_MODELS` | `english,multilingual` | İmaja gömülecek checkpoint'ler. Boş bırakılırsa ilk kullanımda indirilir |
| `TORCH_INDEX` | CPU wheel index | GPU için ör. `https://download.pytorch.org/whl/cu124` |

```bash
docker build --build-arg LAYA_BAKE_MODELS=english,multilingual,typed-decisions -t aydinozturk/laya-mcp:full .
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

stdio modunda her oturum kendi container'ını açar ve ilk çağrıda modeli belleğe yükler (CPU'da ~2–10 sn). Birden fazla proje veya istemci aynı anda kullanacaksa aşağıdaki HTTP modu daha uygundur.

### Paylaşımlı HTTP sunucusu (docker compose)

Sunucuya yalnızca [docker-compose.yml](docker-compose.yml) dosyasını kopyalamak yeterlidir, compose yayınlanan `aydinozturk/laya-mcp:latest` imajını çeker:

```bash
MCP_API_KEY=gizli-anahtar docker compose up -d
docker compose pull && docker compose up -d   # yeni sürüme güncelleme
```

`MCP_API_KEY` zorunludur, tanımlı değilse compose başlamaz. Model açılışta belleğe yüklenir ve endpoint `http://<sunucu>:8000/mcp` olur (streamable HTTP). Sağlık kontrolü `GET /health` adresindedir. İsteğe bağlı değişkenler: `LAYA_MCP_PORT` (varsayılan 8000), `LAYA_THREADS`, `LAYA_DEFAULT`. Anahtarları bir `.env` dosyasına da yazabilirsin.

```bash
claude mcp add --transport http laya http://localhost:8000/mcp --header "Authorization: Bearer gizli-anahtar"
```

`.mcp.json` ile:

```json
{
  "mcpServers": {
    "laya": {
      "type": "http",
      "url": "http://localhost:8000/mcp",
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
| `LAYA_DEVICE` | otomatik | `cpu`, `cuda`, `mps` |
| `LAYA_THREADS` | torch varsayılanı | CPU thread sınırı; fiziksel çekirdek sayısını aşmayın |
| `LAYA_DEFAULT` | `english` | Dili tespit edilemeyen kısa Latin metinler için checkpoint. Çoğunlukla Türkçe trafikte `multilingual` yapın |

Bellek: iki checkpoint yüklüyken container yaklaşık 3–4 GB RAM kullanır. Docker Desktop bellek limitinin buna yettiğinden emin olun.

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
