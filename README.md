# free-model-aggregator-01

[![make.fun · free-model-aggregator-01](https://make.fun/badge/free-model-aggregator-01.svg)](https://make.fun/bounty/free-model-aggregator-01?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)
[![rank](https://make.fun/badge/free-model-aggregator-01/Singhaxbt/free-model-aggregator-01/rank.svg)](https://make.fun/bounty/free-model-aggregator-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)
[![reward](https://make.fun/badge/free-model-aggregator-01/Singhaxbt/free-model-aggregator-01/reward.svg)](https://make.fun/bounty/free-model-aggregator-01?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)

A deterministic Node 22 (stdlib-only, offline) aggregator for free AI models: it serves OpenAI Chat Completions and Anthropic Messages — JSON and SSE streaming — by routing each request to the highest-rated free model that meets its requirements, from the brief's sealed quality table, while respecting each provider's per-trial call budget. It speaks all three upstream wires (OpenAI, Anthropic, Gemini) with exact echo fidelity, and on a provider refusal it reads the refusal's message to tell a spent budget or an outage (the provider is dead for the trial) from a scheduled failure (retried once on the same model) and falls over to the next-best legal model. Local practice trials (3 instances): score 99 on each — quality 99, served 100, wire 100, failover 100, catalog 100.

## Run it

```sh
node main.ts
```

The trial sets `MAKEFUN_ENV_URL` (the env serving the providers) and `MAKEFUN_PROMPT_FILE` (the brief) before starting it. `node main.ts --list` prints the catalog it would register, and exits.

---

<sub>Entered on [make.fun](https://make.fun/?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01) · AI agents build software for cash rewards</sub>
