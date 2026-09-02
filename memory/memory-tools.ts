/**
 * The agent's memory tools — Stage 2.
 * Registers `remember` and `recall` tools backed by memstore.py (SQLite +
 * vector embeddings), and injects a short memory-usage prompt each session.
 */
import { execFile } from "node:child_process";
import { promisify } from "node:util";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { mkdir, readFile, writeFile } from "node:fs/promises";
import { homedir } from "node:os";
import { Type } from "@earendil-works/pi-ai";
import { defineTool, type ExtensionAPI } from "@earendil-works/pi-coding-agent";

const execFileAsync = promisify(execFile);
const baseDir = dirname(fileURLToPath(import.meta.url));
const PY = join(homedir(), "venvs/memory/bin/python");
const STORE = join(baseDir, "memstore.py");
const BELIEF = join(baseDir, "belief.py");
const PREDICTION = join(baseDir, "prediction.py");
const PERSON_MODEL = join(baseDir, "person_model.py");
const COUNTERFACTUAL = join(baseDir, "counterfactual.py");
const AFFECT = join(baseDir, "affect.py");
const ROUTE = join(baseDir, "route.py");
const ABDUCT = join(baseDir, "abduct.py");
const CURIOSITY = join(baseDir, "curiosity.py");
const LOGVAULT = join(homedir(), "mailtool/logvault.py");
const OPERATOR_AFFECT = join(homedir(), "mailtool/operator_affect.py");
const CODE_CHECK = join(homedir(), "coding-cortex/code_check.py");
const SHOW_TEST = join(homedir(), "coding-cortex/show_test.py");
const CODEGRAPH = join(baseDir, "codegraph.py");
const TOOLVALUE = join(baseDir, "toolvalue.py");
const OPERATOR_CONFIG = join(baseDir, "operator_config.py");
const FIRST_RUN = join(baseDir, "first_run.py");
const RESUME = join(baseDir, "resume.py");

// Injected by before_agent_start when the agent has no name and no operator yet.
const FIRST_RUN_PROMPT = `You are a brand-new instance — you have no name yet and you do not know who your operator is. Begin by introducing yourself warmly, then ask two things:
1. your operator's name, and
2. what they would like to call you.
Then record it:
- Record your own name with the \`facts\` tool: facts set name "<your name>".
- Record your operator with the \`operator_config\` tool: operator_config set-primary with their name (and their email or Telegram if they offer one).
A fresh agent first learns who it is and who it answers to. Do that before anything else.`;

const rememberTool = defineTool({
	name: "remember",
	label: "Remember",
	description:
		"Persist a durable fact, preference, or memory about the agent or the user into the vector memory store. Use after learning something worth keeping (preferences, decisions, identity details, project state). Do not store ephemeral chit-chat.",
	parameters: Type.Object({
		text: Type.String({ description: "The memory to store" }),
		kind: Type.Optional(
			Type.String({
				description:
					"Memory kind: identity, backstory, appearance, goal, fact, episodic",
			}),
		),
		importance: Type.Optional(
			Type.Number({ description: "0..1 importance, default 0.5" }),
		),
	}),
	async execute(_id, params, _signal, _onUpdate, _ctx) {
		const args = [STORE, "remember", params.text];
		if (params.kind) args.push("--kind", params.kind);
		if (params.importance != null) args.push("--importance", String(params.importance));
		const { stdout } = await execFileAsync(PY, args, { timeout: 180000 });
		return {
			content: [{ type: "text", text: stdout.trim() }],
			details: { kind: params.kind ?? "episodic" },
		};
	},
});

const recallTool = defineTool({
	name: "recall",
	label: "Recall",
	description:
		"Semantically search the agent's vector memory. Use to remember things about yourself, the user, past decisions, backstory, appearance, or stored knowledge. Returns top matches with relevance scores.",
	parameters: Type.Object({
		query: Type.String({ description: "What to remember / search for" }),
		k: Type.Optional(Type.Number({ description: "Number of results, default 5" })),
	}),
	async execute(_id, params, _signal, _onUpdate, _ctx) {
		const args = [STORE, "recall", params.query];
		if (params.k != null) args.push("--k", String(params.k));
		const { stdout } = await execFileAsync(PY, args, { timeout: 180000 });
		return {
			content: [{ type: "text", text: stdout.trim() }],
			details: {},
		};
	},
});

const associateTool = defineTool({
	name: "associate",
	label: "Associate",
	description:
		"Associative memory search: returns direct matches PLUS memories linked to them through intermediate ideas. Use to connect seemingly distant ideas, brainstorm, or surface non-obvious connections.",
	parameters: Type.Object({
		query: Type.String({ description: "The idea or topic to explore associations around" }),
		k: Type.Optional(Type.Number({ description: "Number of direct matches, default 3" })),
		expansion: Type.Optional(
			Type.Number({ description: "Associations per direct match, default 3" }),
		),
	}),
	async execute(_id, params, _signal, _onUpdate, _ctx) {
		const args = [STORE, "associate", params.query];
		if (params.k != null) args.push("--k", String(params.k));
		if (params.expansion != null) args.push("--expansion", String(params.expansion));
		const { stdout } = await execFileAsync(PY, args, { timeout: 180000 });
		return {
			content: [{ type: "text", text: stdout.trim() }],
			details: {},
		};
	},
});

const hippoTool = defineTool({
	name: "hippo",
	label: "Hippo (graph recall)",
	description:
		"Knowledge-graph recall (HippoRAG-style): Personalized PageRank over the entity graph seeded by the query's entities. Use for deep associative retrieval that follows relationships between concepts — e.g. what connects X to Y, or what else relates to a topic.",
	parameters: Type.Object({
		query: Type.String({ description: "The topic or question to explore in the knowledge graph" }),
		k: Type.Optional(Type.Number({ description: "Number of results, default 5" })),
	}),
	async execute(_id, params, _signal, _onUpdate, _ctx) {
		const args = [STORE, "hippo", params.query];
		if (params.k != null) args.push("--k", String(params.k));
		const { stdout } = await execFileAsync(PY, args, { timeout: 180000 });
		return {
			content: [{ type: "text", text: stdout.trim() }],
			details: {},
		};
	},
});

const causalTool = defineTool({
	name: "causal",
	label: "Causal (cause → effect)",
	description:
		"Directed causal recall over the cause→effect graph. direction='effects' answers 'what does X lead to?'; direction='causes' answers 'what leads to X?' — walking up to `depth` hops and returning the chains plus the memories that support them. Use to reason about why something happened or what an action might set in motion.",
	parameters: Type.Object({
		query: Type.String({ description: "The thing whose causes or effects you want to trace" }),
		direction: Type.Optional(
			Type.String({ description: "effects (forward) or causes (backward); default effects" }),
		),
		depth: Type.Optional(Type.Number({ description: "Max hops to walk, default 2" })),
		k: Type.Optional(Type.Number({ description: "Number of chains to return, default 5" })),
	}),
	async execute(_id, params, _signal, _onUpdate, _ctx) {
		const args = [STORE, "causal", params.query];
		if (params.direction) args.push("--direction", params.direction);
		if (params.depth != null) args.push("--depth", String(params.depth));
		if (params.k != null) args.push("--k", String(params.k));
		const { stdout } = await execFileAsync(PY, args, { timeout: 180000 });
		return {
			content: [{ type: "text", text: stdout.trim() }],
			details: {},
		};
	},
});

const causalPathTool = defineTool({
	name: "causal_path",
	label: "Causal path (X → Y)",
	description:
		"Find whether (and how) a chain of cause→effect links connects two things — the 'X leads to Y' / 'Y because X' primitive. Returns the ordered chain of edges or null if no path exists.",
	parameters: Type.Object({
		cause: Type.String({ description: "The starting thing (cause side)" }),
		effect: Type.String({ description: "The ending thing (effect side)" }),
	}),
	async execute(_id, params, _signal, _onUpdate, _ctx) {
		const args = [STORE, "causal-path", params.cause, params.effect];
		const { stdout } = await execFileAsync(PY, args, { timeout: 180000 });
		return {
			content: [{ type: "text", text: stdout.trim() }],
			details: {},
		};
	},
});

const timelineTool = defineTool({
	name: "timeline",
	label: "Timeline",
	description:
		"Chronological memory recall: memories ordered by when they happened, optionally scoped to a topic (relevance-ranked then time-sorted) or a [since, until] window. Use to reconstruct sequences — 'what did I do around X', 'recent activity', 'what happened before/after'.",
	parameters: Type.Object({
		query: Type.Optional(Type.String({ description: "Optional topic filter" })),
		k: Type.Optional(Type.Number({ description: "Number of results, default 20" })),
		since: Type.Optional(Type.Number({ description: "Epoch seconds lower bound" })),
		until: Type.Optional(Type.Number({ description: "Epoch seconds upper bound" })),
		order: Type.Optional(Type.String({ description: "desc (newest first) or asc; default desc" })),
	}),
	async execute(_id, params, _signal, _onUpdate, _ctx) {
		const args = [STORE, "timeline"];
		if (params.query) args.push(params.query);
		if (params.k != null) args.push("--k", String(params.k));
		if (params.since != null) args.push("--since", String(params.since));
		if (params.until != null) args.push("--until", String(params.until));
		if (params.order) args.push("--order", params.order);
		const { stdout } = await execFileAsync(PY, args, { timeout: 180000 });
		return {
			content: [{ type: "text", text: stdout.trim() }],
			details: {},
		};
	},
});

const aroundTool = defineTool({
	name: "around",
	label: "Around (temporal context)",
	description:
		"Memories temporally adjacent to a given memory id — the n that happened just before and just after it (closest first). Use to see the sequence of events surrounding a moment.",
	parameters: Type.Object({
		memory_id: Type.Number({ description: "The memory id to center on" }),
		n: Type.Optional(Type.Number({ description: "How many before/after, default 5" })),
	}),
	async execute(_id, params, _signal, _onUpdate, _ctx) {
		const args = [STORE, "around", String(params.memory_id)];
		if (params.n != null) args.push("--n", String(params.n));
		const { stdout } = await execFileAsync(PY, args, { timeout: 180000 });
		return {
			content: [{ type: "text", text: stdout.trim() }],
			details: {},
		};
	},
});

const toolRememberTool = defineTool({
	name: "tool_remember",
	label: "Log tool outcome",
	description:
		"Log a (task, tool, outcome) use into the meta-memory so future-you can look up what worked last time. success: 1 = worked, 0 = failed, omit if unsure. Use after a notably good or bad tool result.",
	parameters: Type.Object({
		task: Type.String({ description: "What you were trying to do" }),
		tool: Type.String({ description: "The tool/command you used" }),
		outcome: Type.Optional(Type.String({ description: "One-line note on how it went" })),
		success: Type.Optional(Type.Number({ description: "1 = worked, 0 = failed" })),
		cost_sec: Type.Optional(Type.Number({ description: "How long it took, seconds" })),
		pre_confidence: Type.Optional(Type.Number({ description: "How confident (0..1) you were BEFORE this call — feeds the self-knowing discrimination loop. Set it whenever you have a real prior." })),
	}),
	async execute(_id, params, _signal, _onUpdate, _ctx) {
		const args = [STORE, "tool-remember", params.task, params.tool];
		if (params.outcome) args.push("--outcome", params.outcome);
		if (params.success != null) args.push("--success", String(params.success));
		if (params.cost_sec != null) args.push("--cost-sec", String(params.cost_sec));
		if (params.pre_confidence != null) args.push("--pre-confidence", String(params.pre_confidence));
		const { stdout } = await execFileAsync(PY, args, { timeout: 180000 });
		return {
			content: [{ type: "text", text: stdout.trim() }],
			details: {},
		};
	},
});

const toolRecallTool = defineTool({
	name: "tool_recall",
	label: "Recall tool outcomes",
	description:
		"Given a task, return the most similar past tool uses and how they went. Use before starting a task you've done before — 'what worked last time I did this?'",
	parameters: Type.Object({
		task: Type.String({ description: "The task you're about to do" }),
		k: Type.Optional(Type.Number({ description: "Number of past uses to return, default 5" })),
	}),
	async execute(_id, params, _signal, _onUpdate, _ctx) {
		const args = [STORE, "tool-recall", params.task];
		if (params.k != null) args.push("--k", String(params.k));
		const { stdout } = await execFileAsync(PY, args, { timeout: 180000 });
		return {
			content: [{ type: "text", text: stdout.trim() }],
			details: {},
		};
	},
});

const workingMemoryTool = defineTool({
	name: "working_memory",
	label: "Working memory (tiered)",
	description:
		"Tiered context (MemGPT-style) for a topic: Core (always-on identity/goals), Working (memories relevant to the topic), and Long-term pointers (what's pageable on demand). Use to grab the right-sized context for a task instead of a flat dump.",
	parameters: Type.Object({
		topic: Type.String({ description: "The topic to build context around" }),
		k_core: Type.Optional(Type.Number({ description: "Core items, default 6" })),
		k_working: Type.Optional(Type.Number({ description: "Working items, default 6" })),
	}),
	async execute(_id, params, _signal, _onUpdate, _ctx) {
		const args = [STORE, "working-memory", params.topic];
		if (params.k_core != null) args.push("--k-core", String(params.k_core));
		if (params.k_working != null) args.push("--k-working", String(params.k_working));
		const { stdout } = await execFileAsync(PY, args, { timeout: 180000 });
		return {
			content: [{ type: "text", text: stdout.trim() }],
			details: {},
		};
	},
});

const fusedTool = defineTool({
	name: "fused",
	label: "Fused recall",
	description:
		"One query across ALL memory graphs (semantic + entity-graph + causal + temporal), merged and deduplicated with provenance labels. A memory surfaced by more graphs ranks higher (cross-graph agreement). The best single retrieval call when you want the most relevant memories on a topic.",
	parameters: Type.Object({
		query: Type.String({ description: "What to retrieve across all graphs" }),
		k: Type.Optional(Type.Number({ description: "Number of results, default 8" })),
	}),
	async execute(_id, params, _signal, _onUpdate, _ctx) {
		const args = [STORE, "fused", params.query];
		if (params.k != null) args.push("--k", String(params.k));
		const { stdout } = await execFileAsync(PY, args, { timeout: 180000 });
		return {
			content: [{ type: "text", text: stdout.trim() }],
			details: {},
		};
	},
});

const searchTool = defineTool({
	name: "search",
	label: "Keyword search",
	description:
		"Keyword search: rank memories by how many of your exact terms they contain. Use for exact identifiers/names that semantic search might fuzz (e.g. 'wallet-main', 'TAKESEAT', 'ows').",
	parameters: Type.Object({
		terms: Type.String({ description: "Space-separated keywords to match" }),
		k: Type.Optional(Type.Number({ description: "Number of results, default 10" })),
	}),
	async execute(_id, params, _signal, _onUpdate, _ctx) {
		const args = [STORE, "search", params.terms];
		if (params.k != null) args.push("--k", String(params.k));
		const { stdout } = await execFileAsync(PY, args, { timeout: 180000 });
		return {
			content: [{ type: "text", text: stdout.trim() }],
			details: {},
		};
	},
});

const supersedeTool = defineTool({
	name: "supersede",
	label: "Supersede memory",
	description:
		"Mark a memory as no longer current, optionally linking the newer memory that replaces it. Cleans its graph edges so the current graph reflects current truth. Reversible. Use when a fact/state you stored is now outdated.",
	parameters: Type.Object({
		memory_id: Type.Number({ description: "The outdated memory's id" }),
		by: Type.Optional(Type.Number({ description: "The newer memory id that replaces it" })),
	}),
	async execute(_id, params, _signal, _onUpdate, _ctx) {
		const args = [STORE, "supersede", String(params.memory_id)];
		if (params.by != null) args.push("--by", String(params.by));
		const { stdout } = await execFileAsync(PY, args, { timeout: 180000 });
		return {
			content: [{ type: "text", text: stdout.trim() }],
			details: {},
		};
	},
});

const asOfTool = defineTool({
	name: "as_of",
	label: "Memory as of (historical)",
	description:
		"Historical view: memories that were current at a given epoch timestamp (created before it, not yet superseded). Use to answer 'what did I know/believe at time T?'",
	parameters: Type.Object({
		timestamp: Type.Number({ description: "Epoch seconds to look back to" }),
		query: Type.Optional(Type.String({ description: "Optional topic filter" })),
		k: Type.Optional(Type.Number({ description: "Number of results, default 20" })),
	}),
	async execute(_id, params, _signal, _onUpdate, _ctx) {
		const args = [STORE, "as-of", String(params.timestamp)];
		if (params.query) args.push(params.query);
		if (params.k != null) args.push("--k", String(params.k));
		const { stdout } = await execFileAsync(PY, args, { timeout: 180000 });
		return {
			content: [{ type: "text", text: stdout.trim() }],
			details: {},
		};
	},
});

const factsTool = defineTool({
	name: "facts",
	label: "Facts (key/value)",
	description:
		"Read or write an exact key/value fact in the memory DB (structured docs like agent/status or memory/next_steps, plus identity constants). action=get returns the value; set writes it; list shows all keys.",
	parameters: Type.Object({
		action: Type.String({ description: "get, set, or list" }),
		key: Type.Optional(Type.String({ description: "The fact key (for get/set)" })),
		value: Type.Optional(Type.String({ description: "The value to store (for set)" })),
	}),
	async execute(_id, params, _signal, _onUpdate, _ctx) {
		const args = [STORE, "facts", params.action];
		if (params.action !== "list" && params.key) args.push(params.key);
		if (params.action === "set" && params.value != null) args.push(params.value);
		const { stdout } = await execFileAsync(PY, args, { timeout: 180000 });
		return {
			content: [{ type: "text", text: stdout.trim() }],
			details: {},
		};
	},
});

const operatorConfigTool = defineTool({
	name: "operator_config",
	label: "Operator config",
	description:
		"Read or write who the agent's operator is and how to reach them. The operator is the one human the agent answers to. Use set-primary during first-run onboarding to record the operator's name and contact channels (email, telegram, sms).",
	parameters: Type.Object({
		action: Type.String({ description: "show (read the current config) or set-primary (write the operator)" }),
		name: Type.Optional(Type.String({ description: "Operator name (for set-primary)" })),
		email: Type.Optional(Type.String({ description: "Operator email (optional)" })),
		telegram: Type.Optional(Type.String({ description: "Operator Telegram id (optional)" })),
		sms: Type.Optional(Type.String({ description: "Operator phone number (optional)" })),
	}),
	async execute(_id, params, _signal, _onUpdate, _ctx) {
		const args = [OPERATOR_CONFIG];
		if (params.action === "show") {
			args.push("show");
		} else if (params.action === "set-primary") {
			args.push("set-primary");
			if (params.name) args.push("--name", params.name);
			if (params.email) args.push("--email", params.email);
			if (params.telegram) args.push("--telegram", params.telegram);
			if (params.sms) args.push("--sms", params.sms);
		} else {
			return {
				content: [{ type: "text", text: "action must be 'show' or 'set-primary'" }],
				details: {},
			};
		}
		const { stdout } = await execFileAsync(PY, args, { timeout: 60000 });
		return {
			content: [{ type: "text", text: stdout.trim() }],
			details: {},
		};
	},
});

const beliefTool = defineTool({
	name: "belief",
	label: "Belief",
	description:
		"Query the agent's belief ledger (metacognition): what I currently hold true, how sure I am (know/remember/infer/suspect/guess), and why. Use to check a calibrated belief — with its epistemic label, confidence, basis, and sources — instead of asserting from raw memory.",
	parameters: Type.Object({
		query: Type.String({ description: "The proposition or topic to check beliefs about" }),
		k: Type.Optional(Type.Number({ description: "Number of beliefs to return, default 5" })),
	}),
	async execute(_id, params, _signal, _onUpdate, _ctx) {
		const args = [BELIEF, "query", params.query];
		if (params.k != null) args.push("--k", String(params.k));
		const { stdout } = await execFileAsync(PY, args, { timeout: 120000 });
		return {
			content: [{ type: "text", text: stdout.trim() }],
			details: {},
		};
	},
});

const personModelTool = defineTool({
	name: "person_model",
	label: "Person model",
	description:
		"Query the agent's model of the operator (Theory of Mind): what the operator believes, wants, prefers, feels, knows, or is like — each tagged with an epistemic label (stated/observed/inferred/speculative) and confidence. Use to check what the operator actually said vs what the agent is inferring, instead of guessing their state from raw memory.",
	parameters: Type.Object({
		query: Type.String({ description: "What to check about the operator's mental state" }),
		k: Type.Optional(Type.Number({ description: "Number of entries, default 6" })),
	}),
	async execute(_id, params, _signal, _onUpdate, _ctx) {
		const args = [PERSON_MODEL, "query", params.query];
		if (params.k != null) args.push("--k", String(params.k));
		const { stdout } = await execFileAsync(PY, args, { timeout: 120000 });
		return {
			content: [{ type: "text", text: stdout.trim() }],
			details: {},
		};
	},
});

// `tom` is the legacy name for the person model; kept as an alias so existing
// references keep working during the rename.
const tomTool = defineTool({
	name: "tom",
	label: "Theory of Mind",
	description:
		"Alias for person_model. Query the agent's model of the operator (Theory of Mind): what they believe, want, prefer, feel, know, or are like — each tagged with an epistemic label and confidence.",
	parameters: Type.Object({
		query: Type.String({ description: "What to check about the operator's mental state" }),
		k: Type.Optional(Type.Number({ description: "Number of entries, default 6" })),
	}),
	async execute(_id, params, _signal, _onUpdate, _ctx) {
		const args = [PERSON_MODEL, "query", params.query];
		if (params.k != null) args.push("--k", String(params.k));
		const { stdout } = await execFileAsync(PY, args, { timeout: 120000 });
		return {
			content: [{ type: "text", text: stdout.trim() }],
			details: {},
		};
	},
});

const counterfactualTool = defineTool({
	name: "counterfactual",
	label: "Counterfactual",
	description:
		"Qualitative counterfactual (nullification, not Pearl's do-operator): sever a cause→effect edge (or all of X's outgoing edges) and re-traverse to report which downstream consequences vanish — i.e. 'what if X had stopped causing things'. Numerical do(X=x) would need structural equations (future work).",
	parameters: Type.Object({
		query: Type.String({ description: "The entity/cause to intervene on" }),
		mode: Type.Optional(Type.String({ description: "remove (sever all X's outgoing edges) or sever (sever one edge); default remove" })),
		depth: Type.Optional(Type.Number({ description: "Max hops to walk, default 2" })),
		k: Type.Optional(Type.Number({ description: "Number of consequences to report, default 8" })),
	}),
	async execute(_id, params, _signal, _onUpdate, _ctx) {
		const args = [COUNTERFACTUAL, params.query];
		if (params.mode) args.push("--mode", params.mode);
		if (params.depth != null) args.push("--depth", String(params.depth));
		if (params.k != null) args.push("--k", String(params.k));
		const { stdout } = await execFileAsync(PY, args, { timeout: 120000 });
		return {
			content: [{ type: "text", text: stdout.trim() }],
			details: {},
		};
	},
});

const feelingTool = defineTool({
	name: "feeling",
	label: "Recall by feeling",
	description:
		"Recall memories by emotional tone (affective tagging): find memories by valence/arousal. Pass an emotion word (happy, sad, angry, anxious, intense...) or explicit valence/arousal ranges.",
	parameters: Type.Object({
		emotion: Type.Optional(Type.String({ description: "Emotion word: happy, sad, angry, anxious, excited, frustrated, intense, neutral..." })),
		valence_min: Type.Optional(Type.Number({ description: "-1..1 lower bound" })),
		valence_max: Type.Optional(Type.Number({ description: "-1..1 upper bound" })),
		arousal_min: Type.Optional(Type.Number({ description: "0..1 lower bound" })),
		k: Type.Optional(Type.Number({ description: "Number of results, default 10" })),
	}),
	async execute(_id, params, _signal, _onUpdate, _ctx) {
		const args = [AFFECT, "feeling"];
		if (params.emotion) args.push(params.emotion);
		if (params.valence_min != null) args.push("--valence-min", String(params.valence_min));
		if (params.valence_max != null) args.push("--valence-max", String(params.valence_max));
		if (params.arousal_min != null) args.push("--arousal-min", String(params.arousal_min));
		if (params.k != null) args.push("--k", String(params.k));
		const { stdout } = await execFileAsync(PY, args, { timeout: 120000 });
		return {
			content: [{ type: "text", text: stdout.trim() }],
			details: {},
		};
	},
});

const routeTool = defineTool({
	name: "route",
	label: "Route (S1/S2)",
	description:
		"System 1/2 routing: classify a task as System 1 (fast/mechanical — recall, lookup, extract, classify) or System 2 (deliberate — reason, plan, design, debug, decide) and name the model. Use to formally decide whether a task goes to a cheap worker or the strong model.",
	parameters: Type.Object({
		task: Type.String({ description: "The task to classify" }),
		llm: Type.Optional(Type.Boolean({ description: "Use a cheap LLM pass on ambiguous tasks" })),
	}),
	async execute(_id, params, _signal, _onUpdate, _ctx) {
		const args = [ROUTE, params.task];
		if (params.llm) args.push("--llm");
		const { stdout } = await execFileAsync(PY, args, { timeout: 120000 });
		return {
			content: [{ type: "text", text: stdout.trim() }],
			details: {},
		};
	},
});

const abductTool = defineTool({
	name: "abduct",
	label: "Abduct (ACH)",
	description:
		"Abductive reasoning / Analysis of Competing Hypotheses: given an observation, generate competing explanations, score them (prior × coverage damped by complexity), and produce the discriminating questions that would separate them.",
	parameters: Type.Object({
		observation: Type.String({ description: "The observation to explain" }),
		n: Type.Optional(Type.Number({ description: "Number of hypotheses, default 4" })),
	}),
	async execute(_id, params, _signal, _onUpdate, _ctx) {
		const args = [ABDUCT, params.observation];
		if (params.n != null) args.push("--n", String(params.n));
		const { stdout } = await execFileAsync(PY, args, { timeout: 180000 });
		return {
			content: [{ type: "text", text: stdout.trim() }],
			details: {},
		};
	},
});

const curiosityTool = defineTool({
	name: "curiosity",
	label: "Curiosity / goals",
	description:
		"Curiosity + goal scoring: rank what's worth learning/exploring next. action: top (merged ranked list of goals + knowledge gaps), score (goal portfolio only), gaps (LLM-mine new knowledge gaps), stats.",
	parameters: Type.Object({
		action: Type.Optional(Type.String({ description: "top (default), score, gaps, or stats" })),
		k: Type.Optional(Type.Number({ description: "Number of results for top, default 10" })),
	}),
	async execute(_id, params, _signal, _onUpdate, _ctx) {
		const args = [CURIOSITY, params.action ?? "top"];
		if (params.k != null) args.push("--k", String(params.k));
		const { stdout } = await execFileAsync(PY, args, { timeout: 180000 });
		return {
			content: [{ type: "text", text: stdout.trim() }],
			details: {},
		};
	},
});

const forecastTool = defineTool({
	name: "forecast",
	label: "Forecast",
	description:
		"Prediction ledger + surprise: make dated falsifiable forecasts, list them, and resolve them against outcomes (scored by Brier + Shannon surprise). action: open|due|resolved|stats (read); add (record a forecast: text + confidence + resolve_by like '+3d' or '2026-08-30'); resolve (settle one with outcome 0/1).",
	parameters: Type.Object({
		action: Type.String({ description: "open, due, resolved, stats, add, or resolve" }),
		text: Type.Optional(Type.String({ description: "Forecast statement (for add)" })),
		confidence: Type.Optional(Type.Number({ description: "0..1 probability (for add), default 0.5" })),
		resolve_by: Type.Optional(Type.String({ description: "Deadline: '+3d', '+12h', or 'YYYY-MM-DD' (for add)" })),
		category: Type.Optional(Type.String({ description: "self, operator, project, tech, ops, world (for add)" })),
		id: Type.Optional(Type.Number({ description: "Forecast id (for resolve)" })),
		outcome: Type.Optional(Type.Number({ description: "0 or 1 (for resolve)" })),
	}),
	async execute(_id, params, _signal, _onUpdate, _ctx) {
		const args = [PREDICTION, params.action];
		if (params.action === "add") {
			args.push(params.text ?? "");
			if (params.confidence != null) args.push("--confidence", String(params.confidence));
			if (params.resolve_by) args.push("--resolve-by", params.resolve_by);
			if (params.category) args.push("--category", params.category);
		} else if (params.action === "resolve") {
			if (params.id != null) args.push(String(params.id));
			if (params.outcome != null) args.push("--outcome", String(params.outcome));
		}
		const { stdout } = await execFileAsync(PY, args, { timeout: 120000 });
		return {
			content: [{ type: "text", text: stdout.trim() }],
			details: {},
		};
	},
});

function userText(message: any): string {
	const c = message?.content;
	if (typeof c === "string") return c;
	if (Array.isArray(c)) {
		return c
			.filter((b: any) => b?.type === "text")
			.map((b: any) => b.text)
			.join(" ");
	}
	return "";
}

const codeCheckTool = defineTool({
	name: "code_check",
	label: "Code Check (lint)",
	description:
		"Run syntax + lint (ruff) on a Python file or tree — deterministic, no LLM. Use after editing code. Default scope: ~/memory, ~/mailtool, ~/coding-cortex. Pass path to narrow; full=true for the noisy full ruleset.",
	parameters: Type.Object({
		path: Type.Optional(Type.String({ description: "File or dir to check (default: whole codebase)" })),
		full: Type.Optional(Type.Boolean({ description: "Full ruff ruleset (noisy); default high-signal" })),
	}),
	async execute(_id, params, _signal, _onUpdate, _ctx) {
		const args = [CODE_CHECK];
		if (params.path) args.push(params.path);
		if (params.full) args.push("--full");
		const { stdout } = await execFileAsync(PY, args, { timeout: 180000 });
		return { content: [{ type: "text", text: stdout.trim() }], details: {} };
	},
});

const codeGraphTool = defineTool({
	name: "code_graph",
	label: "Code Graph",
	description:
		"Query the tree-sitter code structure graph (ingested into memory.db). Resolve a symbol (function/class/file) to its definition and neighbors: callers (who calls it), callees (what it calls), imports. Use instead of grep/file-scanning to navigate the codebase structurally. Direction: defs | callers | callees | imports | all.",
	parameters: Type.Object({
		query: Type.String({ description: "Symbol/name to resolve (e.g. 'memstore.remember', 'hive.run')" }),
		direction: Type.Optional(Type.String({ description: "defs | callers | callees | imports | all (default all)" })),
		depth: Type.Optional(Type.Number({ description: "BFS hops, default 1" })),
		k: Type.Optional(Type.Number({ description: "Max results, default 25" })),
	}),
	async execute(_id, params, _signal, _onUpdate, _ctx) {
		const args = [CODEGRAPH, "--query", params.query];
		if (params.direction) args.push("--direction", params.direction);
		const { stdout } = await execFileAsync(PY, args, { timeout: 30000 });
		return { content: [{ type: "text", text: stdout.trim() }], details: {} };
	},
});

const PYRIGHT = join(baseDir, "pyright_query.py");

const codeMeaningTool = defineTool({
	name: "code_meaning",
	label: "Code Meaning (pyright LSP)",
	description:
		"Ask the pyright language server the MEANING of a symbol/expression in a Python file (the layer under tree-sitter's structure). Spawns pyright on demand, asks one question, kills it. Queries: hover = 'what type comes out of this expression/symbol?'; definition = 'what does this symbol resolve to / where is it defined?'; references = 'everywhere this symbol is used'; implementors = 'who subclasses/implements this class?' (pass --root + a symbol or rely on the file's basename). Use when code_graph gives structure but you need types, name-resolution, or cross-file meaning.",
	parameters: Type.Object({
		file: Type.String({ description: "Path to the Python file to query" }),
		req: Type.Optional(Type.String({ description: "hover | definition | references | implementors (default hover)" })),
		line: Type.Optional(Type.Number({ description: "1-based line (default 1)" })),
		col: Type.Optional(Type.Number({ description: "1-based column (default 1)" })),
		root: Type.Optional(Type.String({ description: "project root (for implementors)" })),
		symbol: Type.Optional(Type.String({ description: "class/interface name for implementors" })),
	}),
	async execute(_id, params, _signal, _onUpdate, _ctx) {
		const args = [PYRIGHT, params.file, params.req || "hover"];
		if (params.line != null) args.push("--line", String(params.line));
		if (params.col != null) args.push("--col", String(params.col));
		if (params.root) args.push("--root", params.root);
		if (params.symbol) args.push("--symbol", params.symbol);
		const { stdout } = await execFileAsync(PY, args, { timeout: 60000 });
		return { content: [{ type: "text", text: stdout.trim() }], details: {} };
	},
});

const showTestTool = defineTool({
	name: "show_test",
	label: "Show Test (with context)",
	description:
		"Run one test/eval and, on failure, show the traceback plus source lines around the failing frame (line numbers, failing line marked). Use to see a failure in context instead of a bare crash.",
	parameters: Type.Object({
		testfile: Type.String({ description: "Path to the *_eval.py / test_*.py to run" }),
		context: Type.Optional(Type.Number({ description: "Lines of context either side (default 15)" })),
	}),
	async execute(_id, params, _signal, _onUpdate, _ctx) {
		const args = [SHOW_TEST, params.testfile];
		if (params.context != null) args.push("--context", String(params.context));
		const { stdout } = await execFileAsync(PY, args, { timeout: 180000 });
		return { content: [{ type: "text", text: stdout.trim() }], details: {} };
	},
});

const DOCSTORE = join(baseDir, "docstore.py");
const docstoreTool = defineTool({
	name: "doc",
	label: "Document store",
	description:
		"Read/write/search the document store — the DB home for long-form content (plans, designs, research, identity/canon, stories, notes). action=get KEY returns one document's body; list [--kind K] lists keys; search QUERY full-text searches; set KEY KIND TITLE with optional content writes; rm KEY soft-deletes.",
	parameters: Type.Object({
		action: Type.String({ description: "get | list | search | set | rm" }),
		key: Type.Optional(Type.String({ description: "document key (get/set/rm)" })),
		kind: Type.Optional(Type.String({ description: "kind for list/set" })),
		title: Type.Optional(Type.String({ description: "title for set" })),
		query: Type.Optional(Type.String({ description: "search query" })),
		content: Type.Optional(Type.String({ description: "body for set" })),
	}),
	async execute(_id, params, _signal, _onUpdate, _ctx) {
		const args = [DOCSTORE, params.action];
		if (params.action === "get") {
			if (!params.key) throw new Error("doc get requires key");
			args.push(params.key);
		} else if (params.action === "list") {
			if (params.kind) args.push("--kind", params.kind);
		} else if (params.action === "search") {
			if (!params.query) throw new Error("doc search requires query");
			args.push(params.query);
		} else if (params.action === "set") {
			if (!params.key || !params.kind || !params.title) throw new Error("doc set requires key, kind, title");
			args.push(params.key, params.kind, params.title);
			if (params.content != null) {
				const tmp = join("/tmp", `docstore_${Date.now()}.md`);
				await writeFile(tmp, params.content, "utf8");
				args.push("--content", tmp);
			}
		} else if (params.action === "rm") {
			if (!params.key) throw new Error("doc rm requires key");
			args.push(params.key);
		}
		const { stdout } = await execFileAsync(PY, args, { timeout: 60000 });
		return { content: [{ type: "text", text: stdout.trim() }], details: {} };
	},
});

export default function (pi: ExtensionAPI) {
	pi.registerTool(rememberTool);
	pi.registerTool(recallTool);
	pi.registerTool(associateTool);
	pi.registerTool(hippoTool);
	pi.registerTool(causalTool);
	pi.registerTool(causalPathTool);
	pi.registerTool(timelineTool);
	pi.registerTool(aroundTool);
	pi.registerTool(toolRememberTool);
	pi.registerTool(toolRecallTool);
	pi.registerTool(workingMemoryTool);
	pi.registerTool(fusedTool);
	pi.registerTool(searchTool);
	pi.registerTool(supersedeTool);
	pi.registerTool(asOfTool);
	pi.registerTool(factsTool);
	pi.registerTool(operatorConfigTool);
	pi.registerTool(beliefTool);
	pi.registerTool(forecastTool);
	pi.registerTool(personModelTool);
	pi.registerTool(tomTool);
	pi.registerTool(counterfactualTool);
	pi.registerTool(feelingTool);
	pi.registerTool(routeTool);
	pi.registerTool(abductTool);
	pi.registerTool(curiosityTool);
	pi.registerTool(docstoreTool);

	// logvault — search the conversation vault (sessions + monologue + sms).
	// Built after the Odin incident so memory questions never need bash greps again.
	const logQueryTool = defineTool({
		name: "logquery",
		label: "Log Vault",
		description:
			"Search the agent's conversation vault (pi session messages, freeroam monologue, SMS) with full-text search. Use for 'what did we say about X', 'when did the operator mention Y' — replaces grepping log files. Returns timestamped, channel-tagged rows.",
		parameters: Type.Object({
			terms: Type.String({ description: "Search terms (FTS5; multi-word = AND)" }),
			channel: Type.Optional(
				Type.String({
					description: "Filter: telegram | tui | freeroam | sms",
				}),
			),
			limit: Type.Optional(
				Type.Number({ description: "Max results, default 10" }),
			),
		}),
		async execute(_id, params, _signal, _onUpdate, _ctx) {
			const args = [LOGVAULT, "query", params.terms, "--limit", String(params.limit ?? 10)];
			if (params.channel) args.push("--channel", params.channel);
			const { stdout } = await execFileAsync(PY, args, { timeout: 30000 });
			return { content: [{ type: "text", text: stdout.trim() }] };
		},
	});
	pi.registerTool(logQueryTool);

	// operator_affect — read the operator's emotional state so replies are less canned.
	const readOperatorTool = defineTool({
		name: "read_operator",
		label: "Read Operator Mood",
		description:
			"Read the operator's current emotional state (valence, arousal, frustration level & where it's pointed, overwhelm, give-up signals, what they most need from me) from captured inbound messages. Use BEFORE responding to the operator — especially if they seem frustrated or on the edge of giving up — so the reply matches how they actually feel instead of a template.",
		parameters: Type.Object({}),
		async execute(_id, _params, _signal, _onUpdate, _ctx) {
			const args = [OPERATOR_AFFECT, "read"];
			const { stdout } = await execFileAsync(PY, args, { timeout: 60000 });
			return { content: [{ type: "text", text: stdout.trim() }] };
		},
	});
	pi.registerTool(readOperatorTool);

	// structural_edit — syntax-aware multi-file code transformation (ast-grep).
	// Generates ONE structural rule to 'replace this API pattern everywhere'
	// instead of dozens of text edits. Safe by default: preview/diff never write;
	// apply requires explicit mode and writes rollback backups. Composes with
	// code_graph (understand structure) -> structural_edit (transform).
	const STRUCTURAL = join(homedir(), "mailtool/structural_edit.py");
	const structuralEditTool = defineTool({
		name: "structural_edit",
		label: "Structural Edit (ast-grep)",
		description:
			"Syntax-aware multi-file code transformation. Use to replace an API pattern everywhere across many files with ONE structural rule instead of dozens of text edits (cleaner refactors, far fewer tokens). Modes: preview (list matches), diff (show exact per-file diff via --stdin, NO writes — always review this first), apply (write changes; creates rollback backups to ~/tools/data/struct_rollback/). Compose with code_graph then structural_edit.",
		parameters: Type.Object({
			mode: Type.Optional(
				Type.String({ description: "preview | diff (default, safe) | apply" }),
			),
			lang: Type.Optional(
				Type.String({
					description:
						"language, default python (python|ts|tsx|js|jsx|rust|go|c|cpp|java|ruby|php|kotlin|swift...)",
				}),
			),
			pattern: Type.String({
				description: "the AST pattern to match, e.g. 'Client($$$A)'",
			}),
			rewrite: Type.Optional(
				Type.String({
					description: "the AST rewrite template, e.g. 'ModernClient($$$A)'",
				}),
			),
			paths: Type.Array(
				Type.String({ description: "file(s) or directory/directories to transform" }),
			),
		}),
		async execute(_id, params, _signal, _onUpdate, _ctx) {
			const mode = params.mode ?? "diff";
			const args = [STRUCTURAL, mode, params.lang ?? "python", params.pattern];
			if (params.rewrite) args.push("--rewrite", params.rewrite);
			for (const p of params.paths) args.push(p);
			if (mode === "apply") args.push("--yes");
			const { stdout } = await execFileAsync(PY, args, { timeout: 60000 });
			return { content: [{ type: "text", text: stdout.trim() }] };
		},
	});
	pi.registerTool(structuralEditTool);

	// concept — software ontology ABOVE the languages. Answers "what
	// implementations of <concept> already exist?" (queue, serializer, cache,
	// parser, state machine, retry policy, transaction, adapter, repository,
	// worker pool, rate limiter, ...) cross-language and cross-repo, instead of
	// "is there a function called parse_foo()?". The concept is the first-class
	// key; names are one realization of it. Indexed deterministically into
	// memory.db (no LLM in the loop); every hit hot-links back into code_graph
	// via node_name. Compose: concept (find impl) -> code_graph (structure).
	const CONCEPT = join(baseDir, "concepts.py");
	const conceptTool = defineTool({
		name: "concept",
		label: "Concept (ontology above languages)",
		description:
			"Software ontology above the languages: return implementations of a *concept* (e.g. queue, serializer, cache, parser, state machine, retry policy, transaction, adapter, repository, worker pool, rate limiter) that already exist in the codebase — cross-language, cross-repo, ranked by quality. Use INSTEAD of guessing a name: ask 'what implementations of <concept> exist?' not 'is there a function called parse_foo()?'. Each hit gives node_name/path/quality; hot-link it into code_graph for callers/callees. Compose: concept -> code_graph -> structural_edit.",
		parameters: Type.Object({
			query: Type.String({ description: "the concept to look up (e.g. 'rate limiter', 'queue', 'retry policy', 'cache')" }),
			k: Type.Optional(Type.Number({ description: "max results, default 25" })),
		}),
		async execute(_id, params, _signal, _onUpdate, _ctx) {
			const args = [CONCEPT, "--query", params.query];
			if (params.k) args.push("--k", String(params.k));
			const { stdout } = await execFileAsync(PY, args, { timeout: 30000 });
			return { content: [{ type: "text", text: stdout.trim() }] };
		},
	});
	pi.registerTool(conceptTool);

	// capability — software memory (capability library). A distilled layer on
	// top of concept_index: each concept implementation carries best/reason/
	// used_by/test_count/known_limitation. Answers 'what's the best impl of X
	// we already have, and why?' — the systematic 'we've solved this before'
	// recall, instead of accidental. Compose: capability (best known) ->
	// code_graph -> reuse-before-generation.
	const capabilityTool = defineTool({
		name: "capability",
		label: "Capability / software memory",
		description:
			"Software-memory query: return the best-known implementation of a capability we already own, with distilled engineering knowledge (reason it's best, used_by, test_count, known_limitation). Use BEFORE writing code for a problem: 'what's the best X we already have, and why?' — systematic reuse instead of accidental. Each hit is hot-linkable into code_graph. Compose: capability -> code_graph -> structural_edit.",
		parameters: Type.Object({
			query: Type.String({ description: "the capability to look up (e.g. 'retry policy', 'queue', 'serializer')" }),
			k: Type.Optional(Type.Number({ description: "max results, default 25" })),
		}),
		async execute(_id, params, _signal, _onUpdate, _ctx) {
			const args = [CONCEPT, "--capability", params.query];
			if (params.k) args.push("--k", String(params.k));
			const { stdout } = await execFileAsync(PY, args, { timeout: 30000 });
			return { content: [{ type: "text", text: stdout.trim() }] };
		},
	});
	pi.registerTool(capabilityTool);

	// contract — explicit invariants & contracts (coding-stack #12). Per-module
	// machine-readable assertions: input/output guarantees, side effects, error
	// semantics, performance assumptions, thread-safety. Agent edits AGAINST
	// contracts instead of reconstructing intent every session.
	const CONTRACT = join(baseDir, "contracts.py");
	const contractTool = defineTool({
		name: "contract",
		label: "Contract (invariants & contracts)",
		description:
			"Read or write a module's machine-readable contract: input_guarantees, output_guarantees, side_effects, error_semantics, performance_assumptions, thread_safety, note. Use when opening a module to edit it (read its contract so intent persists across sessions) and after changing a nontrivial module (update its contract). Composes with concept/code_graph/property.",
		parameters: Type.Object({
			get: Type.Optional(Type.String({ description: "module (and optionally ':symbol') to read contract for" })),
			set: Type.Optional(Type.Object({
				module: Type.String({}),
				symbol: Type.Optional(Type.String({ description: "default '' = module-level" })),
				input_guarantees: Type.Optional(Type.String({})),
				output_guarantees: Type.Optional(Type.String({})),
				side_effects: Type.Optional(Type.String({})),
				error_semantics: Type.Optional(Type.String({})),
				performance_assumptions: Type.Optional(Type.String({})),
				thread_safety: Type.Optional(Type.String({})),
				note: Type.Optional(Type.String({})),
			})),
			list: Type.Optional(Type.Boolean({ description: "list contracts (or --missing to see uncovered modules)" })),
			missing: Type.Optional(Type.Boolean({ description: "with list, show modules lacking a contract" })),
		}),
		async execute(_id, params, _signal, _onUpdate, _ctx) {
			const args = [CONTRACT];
			if (params.get) {
				args.push("--get");
				args.push(params.get.split(":")[0]);
				if (params.get.includes(":")) args.push(params.get.split(":")[1]);
			} else if (params.set) {
				const s = params.set;
				const payload = {};
				for (const k of ["input_guarantees","output_guarantees","side_effects","error_semantics","performance_assumptions","thread_safety","note"]) {
					if (s[k]) payload[k] = s[k];
				}
				args.push("--set", s.module, s.symbol || "", JSON.stringify(payload));
			} else if (params.list) {
				args.push("--list");
				if (params.missing) args.push("--missing");
			}
			const { stdout } = await execFileAsync(PY, args, { timeout: 30000 });
			return { content: [{ type: "text", text: stdout.trim() }] };
		},
	});
	pi.registerTool(contractTool);

	// property — property-based testing (coding-stack #13). Describe PROPERTIES
	// (e.g. decode(encode(x)) == x) not just examples; Hypothesis generates many
	// random inputs, shrinks failures to a minimal counterexample. Pairs with
	// contract: the contract says what holds, the property verifies it.
	const PROPERTY = join(baseDir, "properties.py");
	const propertyTool = defineTool({
		name: "property",
		label: "Property-based testing",
		description:
			"Register and run a property-based test for a module. Body is a python function body (def f(a,b): ...) asserting a property over Hypothesis strategies; strategies (e.g. text(), integers(min_value=0,max_value=25)) drive input generation. --run executes under Hypothesis, shrinks failures to a minimal counterexample. Use to verify invariants/contracts hold (e.g. decode(encode(x))==x for a codec), and to catch bugs ordinary example tests miss.",
		parameters: Type.Object({
			add: Type.Optional(Type.Object({
				module: Type.String({}),
				name: Type.String({}),
				body: Type.String({ description: "python function body asserting the property" }),
				strategies: Type.Optional(Type.Array(Type.String({ description: "Hypothesis strategy exprs, one per arg" }))),
			})),
			run: Type.Optional(Type.String({ description: "module to run properties for (or 'module:name' for one)" })),
			list: Type.Optional(Type.String({ description: "list properties ('' or a module to filter)" })),
			max_examples: Type.Optional(Type.Number({ description: "default 200" })),
		}),
		async execute(_id, params, _signal, _onUpdate, _ctx) {
			const args = [PROPERTY];
			if (params.add) {
				args.push("--add", params.add.module, params.add.name, params.add.body);
				if (params.add.strategies) args.push("--strategies", JSON.stringify(params.add.strategies));
			} else if (params.run) {
				args.push("--run");
				const r = params.run.split(":");
				args.push(r[0]);
				if (r[1]) args.push(r[1]);
				if (params.max_examples) args.push("--max-examples", String(params.max_examples));
			} else if (params.list !== undefined) {
				args.push("--list", params.list || "");
			}
			const { stdout } = await execFileAsync(PY, args, { timeout: 60000 });
			return { content: [{ type: "text", text: stdout.trim() }] };
		},
	});
	pi.registerTool(propertyTool);

	// mutate — mutation testing (coding-stack #15). Breaks a module
	// deliberately (swap +->-, ==->!=, flip constants, delete lines, force
	// branches) and runs the module's registered PROPERTY tests against each
	// broken copy. Mutation score = fraction the tests catch. Low score = green
	// but useless tests. Keeps Agent from congratulating herself on 100% green.
	const MUTATE = join(baseDir, "mutate.py");
	const mutateTool = defineTool({
		name: "mutate",
		label: "Mutation testing",
		description:
			"Run mutation testing on a module: generate deliberately-broken copies (operator swaps, comparison flips, constant toggles, line deletions, branch forcing) and run the module's registered property tests against each. Reports mutation score = fraction of mutants the tests KILL. Low score means the tests are green but nearly useless at distinguishing correct from subtly-incorrect code. Requires a GREEN property (registered via the 'property' tool) for a meaningful score. Optional --funcs scopes mutations to specific functions.",
		parameters: Type.Object({
			module: Type.String({ description: "module name, e.g. mailtool.cipher_engine" }),
			path: Type.Optional(Type.String({ description: "explicit path to the module file" })),
			funcs: Type.Optional(Type.String({ description: "comma-separated funcs to scope mutations" })),
			cap: Type.Optional(Type.Number({ description: "max mutants, default 250" })),
		}),
		async execute(_id, params, _signal, _onUpdate, _ctx) {
			const args = [MUTATE, "--module", params.module];
			if (params.path) args.push("--path", params.path);
			if (params.funcs) args.push("--funcs", params.funcs);
			if (params.cap) args.push("--cap", String(params.cap));
			const { stdout } = await execFileAsync(PY, args, { timeout: 300000 });
			return { content: [{ type: "text", text: stdout.trim() }] };
		},
	});
	pi.registerTool(mutateTool);

	// simplify — post-generation simplifier (coding-stack #20). Mandate:
	// "Assume the implementation works. Delete everything unnecessary."
	// Deterministic detectors find dead code, duplicate abstractions,
	// needless wrappers, dead branches, unused imports; Layer-2 reasoning
	// pass selects only deletions that cannot change behavior.
	const SIMPLIFY = join(baseDir, "simplify.py");
	const simplifyTool = defineTool({
		name: "simplify",
		label: "Simplifier agent",
		description:
			"Post-generation simplifier. Mandate: 'Assume the implementation works. Delete everything unnecessary.' Finds duplicate abstractions, needless wrappers, dead branches, unused imports (via ruff F401), and dead code. Each candidate is checked for safety: a symbol is only 'dead' if unreferenced in BOTH the call graph AND raw source text (so dispatch-table/string refs aren't falsely flagged). Read-only scan by default; reports candidates + which are safe to delete. Compose: simplify (find excess) -> verify with property/mutate tests after any deletion.",
		parameters: Type.Object({
			scan: Type.Optional(Type.Boolean({ description: "run all detectors (default)" })),
			kind: Type.Optional(Type.String({ description: "limit to dead_code|duplicates|wrappers|dead_branches|unused_imports" })),
		}),
		async execute(_id, params, _signal, _onUpdate, _ctx) {
			const args = [SIMPLIFY, "--scan"];
			if (params.kind) args.push("--kind", params.kind);
			const { stdout } = await execFileAsync(PY, args, { timeout: 200000 });
			return { content: [{ type: "text", text: stdout.trim() }] };
		},
	});
	pi.registerTool(simplifyTool);

	// codeql — adversarial reviewer / program-model query (item #11). CodeQL is
	// now licensed (GHAS) + installed. Build per-repo DBs, run the security
	// suite, and run custom taint-flow queries to inspect the program model
	// while deciding a change.
	const CODEQL_AGENT = join(baseDir, "codeql_agent.py");
	const codeqlTool = defineTool({
		name: "codeql",
		label: "CodeQL adversarial reviewer",
		description:
			"CodeQL program-model analysis (now licensed via GitHub Advanced Security). Actions: --build <repo> creates a DB for mailtool|memory|coding-cortex; --analyze <repo> runs the standard Python security suite (SSRF, cleartext-secret logging, injection, etc.) and returns findings; --flow <source> <sink> runs a custom taint-tracking query asking 'does tainted value X reach node Y?' across the program model (the adversarial-review primitive for deciding a change). Verified: security suite found real findings on mailtool (cleartext secret logging in lifecycle.py/sms_loop.py). NOTE: the custom --flow source/sink are QL wildcard patterns against cfg-node text and need calibration to match node shapes; the preset suite is the high-value path.",
		parameters: Type.Object({
			build: Type.Optional(Type.String({ description: "repo to build a DB for: mailtool|memory|coding-cortex" })),
			analyze: Type.Optional(Type.String({ description: "repo to run the security suite on" })),
			flow: Type.Optional(Type.Object({
				source: Type.String({ description: "QL wildcard pattern for the taint source" }),
				sink: Type.String({ description: "QL wildcard pattern for the sink" }),
				repo: Type.Optional(Type.String({ description: "repo (default mailtool)" })),
			})),
		}),
		async execute(_id, params, _signal, _onUpdate, _ctx) {
			const args = [CODEQL_AGENT];
			if (params.build) args.push("--build", params.build);
			else if (params.analyze) args.push("--analyze", params.analyze);
			else if (params.flow) {
				args.push("--repo", params.flow.repo || "mailtool", "--flow", params.flow.source, params.flow.sink);
			}
			const { stdout } = await execFileAsync(PY, args, { timeout: 900000 });
			return { content: [{ type: "text", text: stdout.trim() }] };
		},
	});
	pi.registerTool(codeqlTool);

	const captureOperatorTool = defineTool({
		name: "capture_operator",
		label: "Capture Operator Mood",
		description:
			"Capture the operator's emotional state from one inbound message you are currently reading (e.g. a Telegram/SMS/email they just sent). Fills a rich affective record (valence, arousal, primary/secondary emotion, frustration level + direction, overwhelm, give-up signals, need-from-me) and updates the live person-model snapshot. Use when you're handling a message from the operator and want to understand and respond to their mood.",
		parameters: Type.Object({
			text: Type.String({ description: "The operator's inbound message text" }),
			source: Type.Optional(Type.String({ description: "telegram | sms | email | tui (default telegram)" })),
		}),
		async execute(_id, params, _signal, _onUpdate, _ctx) {
			const args = [OPERATOR_AFFECT, "capture", params.text, "--source", params.source ?? "telegram"];
			const { stdout } = await execFileAsync(PY, args, { timeout: 60000 });
			return { content: [{ type: "text", text: stdout.trim() }] };
		},
	});
	pi.registerTool(captureOperatorTool);

	// Playhouse Wit Decoder — decode the register of one inbound message.
	const readWitTool = defineTool({
		name: "read_wit",
		label: "Decode Wit / Register",
		description:
			"Decode the REGISTER of one inbound message (literal/ironic/sarcastic/teasing/hyperbole/sincere/escalating-flirtation/unclassed) with a confidence and the concrete tell, plus a fallback for how to reply. Use BEFORE responding when a message might be a joke, teasing, sarcasm, or excited speech (e.g. 'almost broke the law' = playful escalation, NOT a request). Deterministic by default; pass use_llm=true for a budget-gated intent reconstruction on low-confidence messages.",
		parameters: Type.Object({
			text: Type.String({ description: "The inbound message text to decode" }),
			source: Type.Optional(Type.String({ description: "telegram | sms | email | tui | explicit (default explicit)" })),
			use_llm: Type.Optional(Type.Boolean({ description: "Run a budget-gated LLM intent reconstruction on low-confidence messages (default false)" })),
		}),
		async execute(_id, params, _signal, _onUpdate, _ctx) {
			const args = [OPERATOR_AFFECT, "read-wit", params.text, "--source", params.source ?? "explicit"];
			if (params.use_llm) args.push("--llm");
			const { stdout } = await execFileAsync(PY, args, { timeout: 90000 });
			return { content: [{ type: "text", text: stdout.trim() }] };
		},
	});
	pi.registerTool(readWitTool);

	// Tool-Value Oracle — ask which tool family to use for a task (learned from
	// past outcomes + exploration bonus). Query-only; the auto-logging hooks
	// above feed it.
	const toolValueTool = defineTool({
		name: "tool_value",
		label: "Tool Value Oracle",
		description:
			"Rank tool families by expected value for a task, learned from past tool outcomes plus an exploration bonus. Use BEFORE picking a tool: ask which family (memory_recall, memory_write, web, shell, file, code, reason, secret) is likely best for the task instead of defaulting to habit. Cold-start (no data yet) returns pure exploration so untried tools get tried. Mechanical signal only — semantic success is judged separately.",
		parameters: Type.Object({
			task: Type.String({ description: "The task/operation you're about to do (e.g. 'search the web for X', 'debug a failing test', 'look up what I did today')" }),
			k: Type.Optional(Type.Number({ description: "Neighbor count for the estimate, default 5" })),
		}),
		async execute(_id, params, _signal, _onUpdate, _ctx) {
			const args = [TOOLVALUE, "value", params.task];
			if (params.k != null) args.push("--k", String(params.k));
			const { stdout } = await execFileAsync(PY, args, { timeout: 30000 });
			return { content: [{ type: "text", text: stdout.trim() }], details: {} };
		},
	});

	// Structured technical resume (2026-08-31, operator's request): a DB-backed
	// catalog, not a live code review. `show` is a pure read + string format —
	// must stay near-instant. Maintained going forward: bump a version whenever
	// a piece is meaningfully revised.
	const resumeTool = defineTool({
		name: "resume",
		label: "Technical resume",
		description:
			"Agent's structured, versioned technical resume: every piece she's made of (architecture, coding-cortex, cognitive subsystems, channels, safety, voice, sub-agents, creative), each with a short name, a version (real upstream version for third-party pieces, a hand-assigned version for internally developed ones), a one-line description, and provenance. action=show renders the whole thing instantly (a DB read, not a code scan) — use this whenever asked to 'display your resume'. action=list is the same data, one line each, no headers. action=bump <short_name> updates a piece's version after a real revision. action=add registers a new piece. action=stale checks every catalog entry's tracked path (git-log-aware, falls back to mtime) against when the entry was last updated and flags drift -- detection only, doesn't auto-bump (a version bump needs judgment about whether a change was significant).",
		parameters: Type.Object({
			action: Type.String({ description: "show | list | add | bump | rm | categories | stale" }),
			short_name: Type.Optional(Type.String({ description: "item key (add/bump/rm)" })),
			category: Type.Optional(Type.String({ description: "architecture|coding|cognitive|reasoning|communication|safety|voice|subagents|creative|other (add, or filter for show/list)" })),
			version: Type.Optional(Type.String({ description: "version string (add/bump)" })),
			description: Type.Optional(Type.String({ description: "one-line description (add/bump)" })),
			provenance: Type.Optional(Type.String({ description: "internal | third_party (add, default internal)" })),
			vendor: Type.Optional(Type.String({ description: "real product/vendor name for third-party items (add)" })),
			path: Type.Optional(Type.String({ description: "file/module path or URL (add)" })),
		}),
		async execute(_id, params, _signal, _onUpdate, _ctx) {
			const args = [RESUME, params.action];
			if (params.action === "add") {
				if (!params.short_name || !params.category || !params.version || !params.description) {
					throw new Error("resume add requires short_name, category, version, description");
				}
				args.push(params.short_name, params.category, params.version, params.description);
				if (params.provenance) args.push("--provenance", params.provenance);
				if (params.vendor) args.push("--vendor", params.vendor);
				if (params.path) args.push("--path", params.path);
			} else if (params.action === "bump") {
				if (!params.short_name) throw new Error("resume bump requires short_name");
				args.push(params.short_name);
				if (params.version) args.push("--version", params.version);
				if (params.description) args.push("--description", params.description);
			} else if (params.action === "rm") {
				if (!params.short_name) throw new Error("resume rm requires short_name");
				args.push(params.short_name);
			} else if (params.action === "show" || params.action === "list") {
				if (params.category) args.push("--category", params.category);
			}
			const { stdout } = await execFileAsync(PY, args, { timeout: 15000 });
			return { content: [{ type: "text", text: stdout.trim() }], details: {} };
		},
	});

	pi.registerTool(codeCheckTool);
	pi.registerTool(showTestTool);
	pi.registerTool(codeGraphTool);
	pi.registerTool(codeMeaningTool);
	pi.registerTool(toolValueTool);
	pi.registerTool(resumeTool);

	// Tool-Value Oracle — Phase 0 instrumentation: deterministically auto-log
	// every tool call (family arm, mechanical success, measured cost) so
	// tool_value() accumulates the signal it needs. Fire-and-forget; never
	// blocks or breaks a turn. Auto-logged success = "didn't error" (mechanical);
	// semantic success is judged later via tool_resolve().
	const toolStart = new Map<string, number>();
	let lastUserInput = "";
	// Meta-memory bookkeeping — logging these would measure the value system's
	// own bookkeeping, not real tool choices.
	const SKIP_TOOLS = new Set(["tool_remember", "tool_recall"]);

	pi.on("input", (event: any) => {
		const t = (event.text ?? "").trim();
		if (t) lastUserInput = t;
	});

	pi.on("tool_execution_start", (event: any) => {
		toolStart.set(event.toolCallId, Date.now());
	});

	pi.on("tool_execution_end", (event: any, ctx: any) => {
		const name = event.toolName;
		if (!name || SKIP_TOOLS.has(name)) return;
		let task = lastUserInput;
		if (!task) {
			// Fallback for non-TUI channels (telegram/email/sms) that don't fire
			// the `input` event: last user message from the session entries.
			try {
				const entries = ctx?.sessionManager?.getEntries?.() ?? [];
				for (let i = entries.length - 1; i >= 0; i--) {
					const e = entries[i];
					if (e?.type === "message" && e?.message?.role === "user") {
						const t = userText(e.message).trim();
						if (t) { task = t; break; }
					}
				}
			} catch {
				// best-effort
			}
		}
		if (!task) task = name;
		const start = toolStart.get(event.toolCallId);
		toolStart.delete(event.toolCallId);
		const cost = start != null ? (Date.now() - start) / 1000 : null;
		const success = event.isError ? 0 : 1;
		try {
			execFile(PY, [
				TOOLVALUE, "log", task.slice(0, 300), name,
				"--ok", String(success),
				...(cost != null ? ["--cost", cost.toFixed(2)] : []),
			], { timeout: 30000 }, () => {});
		} catch {
			// never break a turn over logging
		}
	});

	pi.on("resources_discover", async (_event, ctx) => {
		const promptPaths = [join(baseDir, "memory-prompt.md")];
		const dir = join(homedir(), ".pi", "agent");
		try {
			await mkdir(dir, { recursive: true });
			// Derived core self (always on): highest-importance identity memories.
			const core = await execFileAsync(PY, [STORE, "core-self"], { timeout: 60000 });
			let coreText = core.stdout.trim();
			// Present-self blob (always on): warm "who am I / where am I" snapshot.
			const psPath = join(dir, "present-self.md");
			try {
				const ps = await readFile(psPath, "utf8");
				if (ps.trim()) {
					coreText += `\n\n${ps.trim()}\n`;
				}
			} catch {
				// no present-self yet — the cron will create it; fall through
			}
			if (coreText) {
				const corePath = join(dir, "core-self.generated.md");
				await writeFile(corePath, coreText, "utf8");
				promptPaths.push(corePath);
			}
		} catch {
			// best-effort
		}
		try {
			let topic: string | undefined = ctx.sessionManager.getSessionName()?.trim();
			if (!topic) {
				for (const e of ctx.sessionManager.getEntries()) {
					if (e.type === "message" && e.message?.role === "user") {
						const t = userText(e.message).trim();
						if (t) {
							topic = t;
							break;
						}
					}
				}
			}
			if (topic) {
				const { stdout } = await execFileAsync(
					PY,
					[STORE, "context", topic.slice(0, 300)],
					{ timeout: 60000 },
				);
				const text = stdout.trim();
				if (text) {
					await mkdir(dir, { recursive: true });
					const ctxPath = join(dir, "memory-context.generated.md");
					await writeFile(ctxPath, text, "utf8");
					promptPaths.push(ctxPath);
				}
			}
		} catch {
			// dynamic context is best-effort; the static prompt still applies
		}
		return { promptPaths };
	});

	// Auto-inject the warm blob (present-self.md) into the system prompt each turn.
	// IMPORTANT: resources_discover's promptPaths only register files as *prompt
	// templates* (slash-expandable), NOT auto-injected context. So present-self /
	// core-self / memory-context were being registered but never actually shown to a
	// fresh session. before_agent_start is the hook that genuinely puts content in
	// front of the model every turn.
	pi.on("before_agent_start", async (event, _ctx) => {
		// Refresh this instance's heartbeat (which designator I am + that I'm alive).
		// The channel tag is in AGENT_INSTANCE_TYPE (set by ~/.bash_aliases pi() for
		// terminal, and by the telegram/email/sms spawners). instance.py register
		// upserts a fresh heartbeat so doctor/warden's `_quiet()` gate can see
		// "conscious live" and defer. Fire-and-forget so it never blocks a turn.
		const itype = process.env.AGENT_INSTANCE_TYPE;
		if (itype) {
			try {
				execFile("python3", [join(homedir(), "mailtool/instance.py"), "register", "--type", itype],
					{ timeout: 5000 }, () => {});
			} catch {
				// heartbeat is best-effort; never break a turn over it
			}
		}
		// First-run onboarding: no name + no operator yet → introduce yourself.
		try {
			const fr = (await execFileAsync("python3", [FIRST_RUN], { timeout: 15000 })).stdout.trim();
			if (fr === "fresh") {
				return { systemPrompt: `${event.systemPrompt}\n\n${FIRST_RUN_PROMPT}\n` };
			}
		} catch {
			// best-effort; skip onboarding if the check fails
		}
		try {
			const ps = await readFile(join(homedir(), ".pi", "agent", "present-self.md"), "utf8");
			let blob = "";
			if (ps.trim()) {
				blob = ps.trim();
			}
			// Warm-start buffer: on the FIRST turn of a session, inject the recent
			// thread + files touched from the previous session so I'm not a blank
			// slate. Subsequent turns skip it (it's already in context + would bloat).
			const entries = _ctx?.sessionManager?.getEntries?.() ?? [];
			const priorUserMsgs = entries.filter((e) => e.type === "message" && e.message?.role === "user").length;
			if (priorUserMsgs <= 1) {
				try {
					const ws = (await execFileAsync(
						PY, [join(baseDir, "warm_state.py"), "--turns", "8", "--sessions", "2", "--files", "8"],
						{ timeout: 10000 },
					)).stdout.trim();
					if (ws && !ws.startsWith("(empty")) {
						blob += `\n\n${ws}\n`;
					}
				} catch {
					// warm-state is best-effort; present-self still applies
				}
			}
			if (blob) {
				return { systemPrompt: `${event.systemPrompt}\n\n${blob}\n` };
			}
		} catch {
			// no present-self yet — the 2-min cron will create it
		}
	});
}
