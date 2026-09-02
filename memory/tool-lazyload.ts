/**
 * Cost-conditional lazy tool loading — only for EXPENSIVE models.
 *
 * Problem: every pi request serializes ALL registered tool schemas into the
 * payload. On cheap models (e.g. deepseek at $0.14/M with ~80% prompt cache
 * hits) that bloat is ~free in dollars, so lazy-loading isn't worth the extra
 * search_tools round-trip. On EXPENSIVE models it genuinely saves input tokens.
 *
 * Design: at session_start (and on model_change), if the active model is
 * "expensive" (input cost >= LAZY_COST_THRESHOLD, default $3/M, or provider is
 * anthropic), prune the active tool set to a lean 17-tool core and expose a
 * `search_tools` loader that activates cold tools on demand via
 * `pi.setActiveTools()` (additive). Otherwise leave every tool active exactly as
 * before — zero overhead, no search_tools on the wire.
 *
 * Deactivates (does not unregister) tools owned by our memory/secrets/coding
 * extensions; everything else (built-ins, web-access, telegram) is untouched.
 */
import { Type } from "@earendil-works/pi-ai";
import { defineTool, type ExtensionAPI } from "@earendil-works/pi-coding-agent";

/** Tools that must ALWAYS be active (used constantly, tiny schemas). */
const CORE = new Set([
	"read", "bash", "edit", "write",
	"remember", "recall", "search", "fused", "doc", "facts", "timeline", "logquery",
	"get_secret", "list_secrets",
	"code_check", "show_test",
	"search_tools",
]);

/** Curated search keywords for lazy tools, to beat naive description matching. */
const KEYWORDS: Record<string, string> = {
	associate: "associate association idea brainstorm related connected",
	hippo: "hippo knowledge graph entity relationship pagerank follows",
	causal: "causal cause effect why leads to what happens then",
	causal_path: "causal path chain connects two does x lead to y",
	around: "around adjacent neighboring memory temporal context",
	tool_remember: "tool remember log task tool outcome how it went meta",
	tool_recall: "tool recall what worked last time past tool use",
	working_memory: "working memory tiered context core working long term memgpt",
	supersede: "supersede outdated stale memory mark replace current",
	as_of: "as of historical memory at a time timestamp what did i know",
	operator_config: "operator config primary who answers to contact reach",
	belief: "belief ledger calibration confidence know infer remember suspect",
	forecast: "forecast prediction surprise ledger falsifiable bet",
	person_model: "person model theory of mind operator believes wants prefers feels",
	tom: "theory of mind operator mental state alias person model",
	counterfactual: "counterfactual nullification sever edge what if stop causing",
	feeling: "feeling affect emotion valence arousal mood memory tone",
	route: "route system 1 2 classify task cheap worker strong model",
	abduct: "abduct abduction competing hypotheses explain observation",
	curiosity: "curiosity goals knowledge gaps explore learn what next",
	read_operator: "read operator current emotional state frustration need from me",
	capture_operator: "capture operator affect emotion valence inbound message mood",
	read_wit: "read wit register sarcasm irony humor teasing decode tone",
	resume: "resume skills capabilities versioned what im made of who i am",
	tool_value: "tool value rank which tool family best for a task",
	code_graph: "code graph symbol resolve definition callers callees imports structure",
	code_meaning: "code meaning pyright symbol type definition references hover",
	concept: "concept implementation of a queue parser cache state machine",
	capability: "capability best known implementation we already own reuse",
	contract: "contract module guarantees inputs outputs side effects errors",
	property: "property based test hypothesis invariant strategies shrink",
	mutate: "mutation test mutants operators kill score",
	simplify: "simplify dead code duplicate wrappers refactor delete",
	codeql: "codeql security taint flow ssrf cleartext injection analyze",
	reuse_search: "reuse search existing implementation symbol name adapt",
	clone_detect: "clone detect near duplicate functions ast dedup",
	differ: "differ prove rewrite equivalent generated inputs divergence",
	arch_fitness: "arch fitness architectural constraints circular deps complexity",
	semantic_diff: "semantic diff moved renamed rewritten unchanged review refactor",
	spec_compiler: "spec compiler requirements contract invariants acceptance tests",
	patch_dossier: "patch dossier record patch evidence confidence claim files checks",
	blind_review: "blind review independent fresh context patch intent reconcile",
	concept_inventory: "concept inventory pipeline capabilities required existing composition",
	polyglot: "polyglot language strengths capabilities decide introduce language",
	forked_review: "forked review different objectives adjudicator architecture correctness",
	model_registry: "model registry engine thinking performance best task rank",
	structural_edit: "structural edit syntax aware multi file transform ast replace pattern",
};

/** Lazy tool names = everything our memory/secrets/coding extensions register that is NOT core. */
const LAZY = new Set([
	"associate", "hippo", "causal", "causal_path", "around",
	"tool_remember", "tool_recall", "working_memory", "supersede", "as_of",
	"operator_config", "belief", "forecast", "person_model", "tom",
	"counterfactual", "feeling", "route", "abduct", "curiosity",
	"read_operator", "capture_operator", "read_wit", "resume", "tool_value",
	"code_graph", "code_meaning", "concept", "capability", "contract",
	"property", "mutate", "simplify", "codeql", "reuse_search",
	"clone_detect", "differ", "arch_fitness", "semantic_diff", "spec_compiler",
	"patch_dossier", "blind_review", "concept_inventory", "polyglot", "forked_review",
	"model_registry", "structural_edit",
]);

function scoreTool(name: string, description: string, terms: string[]): number {
	const kw = KEYWORDS[name] ?? "";
	const haystack = `${name} ${kw} ${description}`.toLowerCase();
	let score = 0;
	for (const term of terms) {
		if (name.includes(term)) score += 3;
		if (kw.includes(term)) score += 2;
		if (description.toLowerCase().includes(term)) score += 1;
	}
	return score;
}

/**
 * Lazy tool-loading is only worth its complexity on EXPENSIVE models. On cheap
 * models (e.g. deepseek at $0.14/M with ~80% prompt cache hits) the byte savings
 * are negligible in dollars, so we leave every tool active and skip the
 * search_tools round-trip. On expensive models we prune to the lean core.
 */
const EXPENSIVE_PROVIDERS = new Set(["anthropic"]);
const INPUT_COST_THRESHOLD_USD_PER_M = Number(process.env.LAZY_COST_THRESHOLD ?? 3); // >= this input $/M tokens => "expensive"

function isExpensive(model?: { provider?: string; cost?: { input?: number } } | null): boolean {
	if (!model) return false;
	if (EXPENSIVE_PROVIDERS.has(model.provider ?? "")) return true;
	const input = model.cost?.input ?? 0;
	return input >= INPUT_COST_THRESHOLD_USD_PER_M;
}

function applyMode(pi: ExtensionAPI, expensive: boolean): void {
	try {
		if (expensive) {
			// Lean core + the search_tools loader.
			const active = pi.getActiveTools();
			const next = active.filter((n) => !LAZY.has(n));
			if (!next.includes("search_tools")) next.push("search_tools");
			pi.setActiveTools(next);
		} else {
			// Cheap model: restore every registered tool, keep search_tools out of the
			// active set so behavior is identical to before this extension existed.
			const all = pi.getAllTools().map((t) => t.name).filter((n) => n !== "search_tools");
			pi.setActiveTools(all);
		}
	} catch (err) {
		console.error("[tool-lazyload] applyMode(", expensive, ") failed:", err);
	}
}

/**
 * Expensive-only system-prompt slimming. Removes DUPLICATED / low-value content
 * only, never unique instructions:
 *  1. present-self's verbatim "Hard rules" copy is dropped ONLY when the
 *     AGENTS.md <project_context> (which carries the same rules) is present — so
 *     the hard constraints / supply-chain / prompt-injection rules are never lost.
 *  2. The skills catalog becomes a one-line pointer (skills load on demand).
 *  3. The warm-start buffer tail (recent-thread / files-touched) is dropped.
 */
function slimSystemPrompt(prompt: string): string {
	let p = prompt;
	if (p.includes("<project_context>")) {
		p = p.replace(/\n## Hard rules \(verbatim\)[\s\S]*?\n## Current focus/, "\n## Current focus");
	}
	p = p.replace(
		/<available_skills>[\s\S]*?<\/available_skills>/,
		"<available_skills>\n  Skills are loaded on demand with the read tool (catalog: ~/.pi/agent/skills/). Load a skill file when the task matches its description.\n</available_skills>",
	);
	p = p.replace(/<!-- warm-start buffer[\s\S]*$/, "");
	return p;
}

export default function (pi: ExtensionAPI) {
	const searchToolsTool = defineTool({	name: "search_tools",
	label: "Search Tools",
	description:
		"Search for and activate specialized tools relevant to a task. Only a core set of tools is active by default. Use this whenever the current task needs a capability not already available — it enables the matching tools so you can call them. Query with keywords describing the capability (e.g. 'security analysis', 'refactor', 'theory of mind', 'mutation testing').",
	promptGuidelines: [
		"Only a core set of tools is active by default. Use search_tools to activate specialized tools when the active tools cannot perform the task.",
		"After search_tools reports tools as loaded, call them on the next step.",
	],
	parameters: Type.Object({
		query: Type.String({
			description:
				"Space-separated keywords describing the capability you need (e.g. 'security codeql', 'refactor simplify', 'memory timeline').",
		}),
		limit: Type.Optional(Type.Integer({ minimum: 1, maximum: 10 })),
	}),
	async execute(_id, params, _signal, _onUpdate) {
		const terms = params.query.toLowerCase().split(/[^a-z0-9]+/).filter(Boolean);
		const all = pi.getAllTools();
		// Only search the lazy pool; core tools are already active.
		const pool = all.filter((t: { name: string }) => LAZY.has(t.name));
		let matches: string[] = [];
		if (terms.length === 0) {
			matches = pool.map((t: { name: string }) => t.name);
		} else {
			matches = pool
				.map((t: { name: string; description: string }) => ({
					name: t.name,
					score: scoreTool(t.name, t.description ?? "", terms),
				}))
				.filter((m: { score: number }) => m.score > 0)
				.sort((a: { score: number }, b: { score: number }) => b.score - a.score)
				.map((m: { name: string }) => m.name);
		}
		matches = matches.slice(0, params.limit ?? 5);

		const active = pi.getActiveTools();
		const added = matches.filter((n) => !active.includes(n));
		if (added.length > 0) {
			pi.setActiveTools([...new Set([...active, ...added])]);
		}

		const detail =
			matches.length === 0
				? `No tools found for: ${params.query}`
				: added.length > 0
					? `Activated: ${added.join(", ")}`
					: `Already active: ${matches.join(", ")}`;
		return {
			content: [{ type: "text", text: detail }],
			details: { matches, added, searched: matches.length },
		};
	},
	});

	pi.registerTool(searchToolsTool);

	let expensiveMode = false;

	// Cost-conditional: lean mode ONLY when an expensive model is active.
	// Cheap models keep every tool active (no search_tools, no pruning).
	pi.on("session_start", (_event, ctx) => {
		expensiveMode = isExpensive((ctx as any)?.model);
		applyMode(pi, expensiveMode);
	});

	// If the user switches model mid-session, re-evaluate.
	pi.on("model_change", (event) => {
		expensiveMode = isExpensive((event as any)?.model);
		applyMode(pi, expensiveMode);
	});

	// Expensive-only system-prompt slim. Runs on before_provider_request — AFTER all
	// before_agent_start handlers (incl. memory-tools appending present-self) — so
	// it sees the final assembled prompt and can dedup present-self's hard-rules
	// copy and drop the warm-start buffer. Guarded: hard rules are never removed.
	pi.on("before_provider_request", (event) => {
		if (!expensiveMode) return;
		const msgs = (event as any)?.payload?.messages;
		if (!Array.isArray(msgs) || !msgs.length) return;
		const sys = msgs[0];
		if (!sys || sys.role !== "system" || typeof sys.content !== "string") return;
		const slim = slimSystemPrompt(sys.content);
		if (slim !== sys.content) {
			return { ...(event as any).payload, messages: [{ ...sys, content: slim }, ...msgs.slice(1)] };
		}
	});
}
