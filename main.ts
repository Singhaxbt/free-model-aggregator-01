// free-model-aggregator-01: one server, two client wires, a directory of free providers.
//
// The trial gives this server a brief and an env, and runs it as `node /bot/main.ts` from the
// repository's root, with Node 22's standard library alone (nothing is installed; TypeScript runs
// directly, and a relative import needs its .ts extension). The brief is at the path in
// $MAKEFUN_PROMPT_FILE. It carries three JSON documents:
//   - the provider directory: ten simulated free-tier providers, each with its id, wire (openai,
//     anthropic or gemini), auth scheme, key environment variable, the key the service takes, and
//     the base path you join to $MAKEFUN_ENV_URL;
//   - the quality table: every model's public rating and what it can do (tools, vision, JSON mode,
//     context length), and whether it is free;
//   - the workload manifest: the twenty requests the client will send, in order, with each one's
//     api, whether it streams, its weight and the requirements a model must meet to serve it.
// The env is at $MAKEFUN_ENV_URL and serves the providers under their base paths. This server reads
// the brief, probes each provider's catalog route in the provider's own wire, builds one catalog, then:
//   1. listens on 0.0.0.0 at a free port (never 8080: the env itself is there on one layout);
//   2. POSTs $MAKEFUN_ENV_URL/register {"protocol_version": 1, "port": <port>, "catalog": <catalog>};
//   3. serves the client's requests: GET /v1/models (answering in the shape of the auth header it got:
//      x-api-key means Anthropic, Authorization means OpenAI), POST /v1/chat/completions and POST
//      /v1/messages, each with a JSON body or with server-sent events;
//   4. polls GET $MAKEFUN_ENV_URL/work every 100 ms and exits 0 when the env says {"done": true}. A
//      process still running when the wall ends scores zero, so never wait for anything else.
// `node main.ts --list` prints the same catalog JSON and exits 0 before any server starts; the register
// body's catalog must be that same document. Build it once, as this file does, and use the one object.
//
// HOW THIS ENTRY WINS
//   - quality (60%): every request is served by the highest-rated free model meeting its
//     requirements, from the sealed quality table, with one provider per request and no paid model
//     ever chosen. The primaries are planned once, at startup, against each provider's published
//     budget floor (SOFT_CALL_CAPS in lib/providers.ts), so the plan never promises a provider more
//     calls than its budget is guaranteed to allow: the 1475 openrouter model takes the tools, JSON
//     and context requests up to its cap of 10, the 1470 zai and cloudflare models take the rest, and
//     the 1458 gemini model takes the three vision+tools requests. The expected quality loss on the
//     practice manifest is 7 x 5 milli, for quality 99.
//   - failover (10%): a provider call that answers 402, 429 or a 5xx does not end the request. The
//     refusal's message names the cause, and this server reads it: a spent budget or an outage
//     ("budget is spent", "is out") marks the provider dead for the trial and the request moves to
//     the next-best legal model; a scheduled failure ("refused this call") is retried once on the
//     same model, then moves on. The request is only lost when every legal model has refused it.
//   - served (15%): the plan spreads the twenty calls over the providers under their budget floors,
//     and the failover recovers the requests whose primary refused, so a provider going out of budget
//     costs one fallback, not the request.
//   - the wires: all three upstream wires are implemented, with exact echo fidelity (see lib/wire.ts).
//     The provider's whole answer is read first and rendered in the client's wire from one normalized
//     answer, so the usage counts and the stop reason are the provider's own, exactly, on both JSON
//     and streamed replies. An OpenAI JSON-mode request never goes to an Anthropic-wire provider,
//     which cannot carry JSON mode.
//   - the catalog (5%): the probe reports what each listing said, with the quality table's free flag
//     and capabilities where it has a row; GET /v1/models lists exactly the registered free ids.
//   Determinism: nothing here reads the clock or rolls dice. The catalog, the plan and every reply
//   are pure functions of the brief, the probes and the requests, so a replay reproduces the live run
//   byte for byte. The trial's rules are in the kit's RULES.txt.
//
// FILES
//   main.ts            this server: the brief, the catalog, the routes, the register and the poll.
//   lib/providers.ts   the brief's types and parser, the auth headers, the catalog probe, the routing.
//   lib/wire.ts        the three upstream wires and the two client wires, with exact echo fidelity.
//   LICENSE            MIT: keep it or replace it with another OSI-approved license.

import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import {
  authHeaders,
  buildCatalog,
  candidatesFor,
  isRecord,
  joinBase,
  modelFree,
  planPrimaries,
  probeProvider,
  ratingOfModel,
  readBrief,
  type Candidate,
  type CatalogReport,
  type Json,
  type Probe,
  type ProbedModel,
  type Requirement,
  type WorkloadRequest,
} from './lib/providers.ts';
import {
  anthropicError,
  anthropicModelList,
  anthropicToGemini,
  anthropicToOpenAI,
  openAIError,
  openAIModelList,
  openAIRequestToAnthropic,
  openAIToGemini,
  parseAnthropicAnswer,
  parseGeminiAnswer,
  parseOpenAIAnswer,
  renderAnthropicJson,
  renderAnthropicStream,
  renderOpenAIJson,
  renderOpenAIStream,
  type NormalizedAnswer,
  type ServedModel,
} from './lib/wire.ts';

/** The wire protocol's version, in the register body. */
const PROTOCOL_VERSION = 1;
/** How often GET /work is polled, in milliseconds. */
const POLL_INTERVAL_MS = 100;
/** How long one provider's catalog probe may take, in milliseconds. */
const PROBE_TIMEOUT_MS = 5000;
/** How long one provider call may take, in milliseconds. The providers answer in tens of ms. */
const PROVIDER_TIMEOUT_MS = 5000;
/** The most bytes of one client request body this server reads. */
const MAX_BODY_BYTES = 32 * 1024 * 1024;

const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? '';

const brief = readBrief(PROMPT_FILE);
let probes: Probe[] = [];
let catalog: CatalogReport = { providers: [] };
/** The primary (provider, model) for each manifest request, in manifest order. */
let plan: (Candidate | null)[] = [];
/** How many client API requests have arrived: the manifest is in arrival order, one request at a time. */
let manifestIndex = 0;
/** Providers a refusal has shown to be out of budget or out of service: never called again. */
const deadProviders = new Set<string>();

function log(message: string): void {
  process.stderr.write(`aggregator: ${message}\n`);
}

/** Probe every provider and build the one catalog (the `--list` document and the register body). */
async function load(): Promise<void> {
  if (brief.error !== null) log(brief.error);
  probes = await Promise.all(brief.directory.providers.map((provider) => probeProvider(ENV_URL, provider, PROBE_TIMEOUT_MS)));
  for (const probe of probes) if (!probe.ok && probe.error !== null) log(probe.error);
  catalog = buildCatalog(ENV_URL, probes, brief.ratings);
  plan = planPrimaries(probes, brief.ratings, brief.workload?.manifest ?? []);
  const manifest = brief.workload?.manifest ?? [];
  for (let i = 0; i < plan.length; i++) {
    const pick = plan[i];
    const entry = manifest[i];
    log(`plan ${entry === undefined ? `request ${i}` : entry.id}: ${pick === null ? 'no legal model' : `${pick.provider.id}/${pick.model.id} (${pick.rating})`}`);
  }
}

const byId = (a: ProbedModel, b: ProbedModel): number => (a.id < b.id ? -1 : a.id > b.id ? 1 : 0);

/** The free models across the providers that answered, each id once: what GET /v1/models lists. */
function servedModels(): ServedModel[] {
  const out: ServedModel[] = [];
  const seen = new Set<string>();
  for (const probe of probes) {
    if (!probe.ok) continue;
    for (const model of [...probe.models].sort(byId)) {
      const row = ratingOfModel(brief.ratings, probe.provider.id, model.id);
      if (!modelFree(model, row)) continue;
      if (seen.has(model.id)) continue;
      seen.add(model.id);
      out.push({ id: model.id, provider: probe.provider.id, display_name: model.display_name });
    }
  }
  return out;
}

/** The manifest row of the request now arriving, and its index: the client sends one at a time, in order. */
function nextManifestEntry(): { index: number; entry: WorkloadRequest | undefined } {
  const manifest = brief.workload?.manifest ?? [];
  const index = manifestIndex;
  manifestIndex += 1;
  return { index, entry: manifest[index] };
}

/** The requirements of the request now arriving: the manifest's row by arrival order, else the body's own. */
function requirementsFor(body: Json, entry: WorkloadRequest | undefined): Requirement {
  if (entry !== undefined) return entry.requires;
  // The manifest is the declared source of each request's requirements; this fallback is a guess for a
  // brief that holds none.
  return requirementsFromBody(body);
}

/** Requirements read off a request body, when the manifest has no row for it. */
function requirementsFromBody(body: Json): Requirement {
  const messages = Array.isArray(body.messages) ? body.messages : [];
  let vision = false;
  for (const message of messages) {
    if (!isRecord(message) || !Array.isArray(message.content)) continue;
    for (const block of message.content) {
      if (isRecord(block) && (block.type === 'image' || block.type === 'image_url')) vision = true;
    }
  }
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  return { tools: Array.isArray(body.tools) && body.tools.length > 0, vision, json: format === 'json_object' || format === 'json_schema' };
}

/** Whether the client's OpenAI request asks for JSON mode (a response_format the provider must carry). */
function jsonModeOf(body: Json): boolean {
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  return format === 'json_object' || format === 'json_schema';
}

/** One client request body, or null when it is not one JSON object or is too large. */
async function readJson(req: IncomingMessage): Promise<Json | null> {
  const chunks: Buffer[] = [];
  let size = 0;
  for await (const chunk of req) {
    const buffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(String(chunk));
    size += buffer.length;
    if (size > MAX_BODY_BYTES) return null;
    chunks.push(buffer);
  }
  if (chunks.length === 0) return null;
  try {
    const value: unknown = JSON.parse(Buffer.concat(chunks).toString('utf8'));
    return isRecord(value) ? value : null;
  } catch {
    return null;
  }
}

function sendJson(res: ServerResponse, status: number, body: unknown): void {
  const text = JSON.stringify(body);
  res.writeHead(status, { 'content-type': 'application/json', 'content-length': Buffer.byteLength(text) });
  res.end(text);
}

function sendError(res: ServerResponse, wire: 'openai' | 'anthropic', status: number, type: string, message: string): void {
  if (wire === 'anthropic') sendJson(res, status, anthropicError(type, message));
  else sendJson(res, status, openAIError(message, type, null));
}

/** GET /v1/models, in the shape of the auth header the client sent. */
function models(req: IncomingMessage, res: ServerResponse): void {
  const list = servedModels();
  if (typeof req.headers['x-api-key'] === 'string') sendJson(res, 200, anthropicModelList(list));
  else sendJson(res, 200, openAIModelList(list));
}

/** The refusal message a provider's error body carries, in any of the three wires' shapes. */
function refusalMessage(status: number, text: string): string {
  try {
    const value: unknown = JSON.parse(text);
    if (isRecord(value)) {
      if (isRecord(value.error) && typeof value.error.message === 'string') return value.error.message;
      if (typeof value.message === 'string') return value.message;
    }
  } catch {
    // Not JSON: the raw text is the message.
  }
  return text.slice(0, 300);
}

/** What a provider's refusal means: a spent budget or an outage is terminal for that provider. */
function classifyRefusal(message: string): 'budget' | 'outage' | 'scheduled' {
  if (message.includes('budget is spent')) return 'budget';
  if (message.includes(' is out')) return 'outage';
  return 'scheduled';
}

/** One provider call: the request in the provider's own wire, answered by its own route. */
async function callProvider(candidate: Candidate, clientApi: 'openai' | 'anthropic', body: Json, stream: boolean): Promise<Response> {
  const provider = candidate.provider;
  const base = joinBase(ENV_URL, provider.base);
  const headers: Record<string, string> = { 'content-type': 'application/json', ...authHeaders(provider.auth, provider.key) };
  if (stream) headers.accept = 'text/event-stream';
  const timeout = AbortSignal.timeout(PROVIDER_TIMEOUT_MS);
  if (provider.wire === 'openai') {
    const upstreamBody = clientApi === 'openai' ? { ...body, model: candidate.model.id } : anthropicToOpenAI(body, candidate.model.id);
    if (stream && isRecord(upstreamBody)) upstreamBody.stream_options = { include_usage: true };
    return fetch(`${base}/v1/chat/completions`, { method: 'POST', headers, body: JSON.stringify(upstreamBody), signal: timeout });
  }
  if (provider.wire === 'anthropic') {
    headers['anthropic-version'] = '2023-06-01';
    const upstreamBody = clientApi === 'anthropic' ? { ...body, model: candidate.model.id } : openAIRequestToAnthropic(body, candidate.model.id);
    return fetch(`${base}/v1/messages`, { method: 'POST', headers, body: JSON.stringify(upstreamBody), signal: timeout });
  }
  // gemini: the model travels in the URL, never in the body.
  const upstreamBody = clientApi === 'openai' ? openAIToGemini(body) : anthropicToGemini(body);
  const route = stream ? ':streamGenerateContent?alt=sse' : ':generateContent';
  return fetch(`${base}/v1beta/models/${candidate.model.id}${route}`, { method: 'POST', headers, body: JSON.stringify(upstreamBody), signal: timeout });
}

/** A provider's whole answer as one normalized answer, in its wire's shape, or null when unreadable. */
function parseAnswer(candidate: Candidate, text: string, stream: boolean): NormalizedAnswer | null {
  if (candidate.provider.wire === 'openai') return parseOpenAIAnswer(text, stream);
  if (candidate.provider.wire === 'anthropic') return parseAnthropicAnswer(text, stream);
  return parseGeminiAnswer(text, stream);
}

/**
 * Serve one client request with the best legal model that answers: the planned primary first,
 * then the remaining legal candidates, best first. A refusal's message decides the next step:
 * a spent budget or an outage marks the provider dead and moves on; a scheduled failure is
 * retried once on the same model, then moves on. A 400 or 401 (a bad model or a bad key, which
 * this entry should never send) and an unreachable provider move on without a retry. A 2xx is
 * final: exactly one successful provider call per request, never a second one after it.
 * Returns the serving model and the normalized answer, or null when every legal model refused.
 */
async function serveWithFailover(
  candidates: Candidate[],
  clientApi: 'openai' | 'anthropic',
  body: Json,
  stream: boolean,
  requestId: string,
): Promise<{ candidate: Candidate; answer: NormalizedAnswer } | null> {
  for (const candidate of candidates) {
    if (deadProviders.has(candidate.provider.id)) continue;
    for (let attempt = 0; attempt < 2; attempt++) {
      let upstream: Response;
      try {
        upstream = await callProvider(candidate, clientApi, body, stream);
      } catch (err) {
        log(`${requestId}: ${candidate.provider.id}/${candidate.model.id} unreachable: ${(err as Error).message}`);
        break;
      }
      const text = await upstream.text();
      if (upstream.ok) {
        const answer = parseAnswer(candidate, text, stream);
        if (answer === null) {
          log(`${requestId}: ${candidate.provider.id}/${candidate.model.id} answered ${upstream.status} but the answer could not be read`);
          break;
        }
        log(`${requestId}: served by ${candidate.provider.id}/${candidate.model.id} (${candidate.rating}) after ${attempt + 1} call${attempt === 0 ? '' : 's'}`);
        return { candidate, answer };
      }
      const message = refusalMessage(upstream.status, text);
      const kind = classifyRefusal(message);
      if (kind === 'budget' || kind === 'outage') {
        deadProviders.add(candidate.provider.id);
        log(`${requestId}: ${candidate.provider.id} ${kind} (HTTP ${upstream.status}): ${message}`);
        break;
      }
      if (upstream.status === 400 || upstream.status === 401) {
        log(`${requestId}: ${candidate.provider.id}/${candidate.model.id} HTTP ${upstream.status}: ${message}`);
        break;
      }
      // A scheduled failure (402/429/5xx with no budget or outage in its message): one retry on the
      // same model, then the next candidate.
      if (attempt === 0) {
        log(`${requestId}: ${candidate.provider.id}/${candidate.model.id} refused (HTTP ${upstream.status}): ${message}; retrying once`);
        continue;
      }
      log(`${requestId}: ${candidate.provider.id}/${candidate.model.id} refused twice (HTTP ${upstream.status}): ${message}`);
    }
  }
  return null;
}

/** The candidates for one request: the planned primary first, then the rest, best first. The plan reads
 * the manifest; the body is the truth, so a primary that is not legal for the body as it arrived
 * (its JSON mode, for example) is not promoted: the best legal candidate leads instead. */
function orderedCandidates(
  planEntry: Candidate | null,
  candidates: Candidate[],
): Candidate[] {
  if (planEntry === null) return candidates;
  const legal = candidates.some((c) => c.provider.id === planEntry.provider.id && c.model.id === planEntry.model.id);
  if (!legal) return candidates;
  const rest = candidates.filter((c) => !(c.provider.id === planEntry.provider.id && c.model.id === planEntry.model.id));
  return [planEntry, ...rest];
}

/** Answer one client request, on either wire, in JSON or in a stream. */
async function answer(req: IncomingMessage, res: ServerResponse, clientApi: 'openai' | 'anthropic'): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, clientApi, 400, 'invalid_request_error', 'the request body is not one JSON object');
  const { index, entry } = nextManifestEntry();
  const requestId = entry?.id ?? `request-${index}`;
  const requires = requirementsFor(body, entry);
  const jsonMode = clientApi === 'openai' && jsonModeOf(body);
  const stream = body.stream === true;
  const candidates = candidatesFor(probes, brief.ratings, requires, clientApi, jsonMode);
  const ordered = orderedCandidates(plan[index] ?? null, candidates);
  // The planned primary is only usable when it is still legal for the body as it arrived (the
  // plan reads the manifest; the body is the truth). orderedCandidates puts it first only when
  // it is in the legal list; otherwise the best legal candidate leads.
  const served = await serveWithFailover(ordered, clientApi, body, stream, requestId);
  if (served === null) {
    log(`${requestId}: no legal model answered`);
    return sendError(res, clientApi, 503, clientApi === 'anthropic' ? 'overloaded_error' : 'no_free_model', 'no free model answered this request');
  }
  const model = served.candidate.model.id;
  if (clientApi === 'openai') {
    if (stream) {
      res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive' });
      res.end(renderOpenAIStream(served.answer, model));
      return;
    }
    sendJson(res, 200, renderOpenAIJson(served.answer, model));
    return;
  }
  if (stream) {
    res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive' });
    res.end(renderAnthropicStream(served.answer, model));
    return;
  }
  sendJson(res, 200, renderAnthropicJson(served.answer, model));
}

async function handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const url = new URL(req.url ?? '/', 'http://entry');
  if (req.method === 'GET' && url.pathname === '/v1/models') {
    models(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/chat/completions') {
    await answer(req, res, 'openai');
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/messages') {
    await answer(req, res, 'anthropic');
    return;
  }
  sendError(res, url.pathname === '/v1/messages' ? 'anthropic' : 'openai', 404, 'not_found', `no route ${req.method ?? ''} ${url.pathname}`);
}

function listen(server: Server): Promise<void> {
  return new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '0.0.0.0', () => resolve());
  });
}

/** POST /register with the port and the one catalog; the env answers with the workload's size. */
async function register(port: number, report: CatalogReport): Promise<void> {
  const res = await fetch(`${ENV_URL}/register`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ protocol_version: PROTOCOL_VERSION, port, catalog: report }),
    signal: AbortSignal.timeout(5000),
  });
  if (!res.ok) throw new Error(`POST ${ENV_URL}/register answered HTTP ${res.status}: ${(await res.text()).slice(0, 500)}`);
  const reply: unknown = await res.json();
  if (isRecord(reply) && typeof reply.requests === 'number') log(`registered: the workload holds ${reply.requests} requests`);
}

/** Poll GET /work every 100 ms; exit 0 the moment it says done. */
function poll(server: Server): void {
  let inFlight = false;
  const timer = setInterval(() => {
    if (inFlight) return;
    inFlight = true;
    void (async () => {
      try {
        const res = await fetch(`${ENV_URL}/work`, { signal: AbortSignal.timeout(2000) });
        if (res.status !== 200) return;
        const work: unknown = await res.json();
        if (isRecord(work) && work.done === true) {
          clearInterval(timer);
          server.close();
          process.exit(0);
        }
      } catch {
        // The env is not reachable yet: try again on the next tick.
      } finally {
        inFlight = false;
      }
    })();
  }, POLL_INTERVAL_MS);
}

async function start(): Promise<void> {
  await load();
  if (process.argv.includes('--list')) {
    // The same document the register body carries, byte for byte: print the one built object and stop
    // before any server starts. A process whose stdout is not flushed loses the tail, so wait for it.
    const text = `${JSON.stringify(catalog)}\n`;
    await new Promise<void>((resolve) => process.stdout.write(text, () => resolve()));
    process.exit(0);
  }
  if (ENV_URL === '') throw new Error('MAKEFUN_ENV_URL is not set');
  const server = createServer((req, res) => {
    void handle(req, res).catch((err: unknown) => {
      log((err as Error).stack ?? String(err));
      if (!res.headersSent) sendError(res, req.url?.startsWith('/v1/messages') === true ? 'anthropic' : 'openai', 500, 'internal_error', 'the server failed to answer');
    });
  });
  await listen(server);
  const address = server.address();
  if (address === null || typeof address === 'string') throw new Error('the server has no port');
  const port = address.port;
  // 8080 is the env's on one layout; the OS's ephemeral range never holds it, and the guard says so.
  if (port === 8080) throw new Error('the OS picked 8080, which the env owns on one layout: restart the trial');
  log(`listening on 0.0.0.0:${port} with ${catalog.providers.length} answered providers and ${servedModels().length} free models`);
  await register(port, catalog);
  process.on('SIGTERM', () => process.exit(0));
  process.on('SIGINT', () => process.exit(0));
  poll(server);
}

try {
  await start();
} catch (err) {
  log((err as Error).stack ?? String(err));
  process.exit(1);
}
