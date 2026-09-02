/**
 * Coding-cortex tools — Coding Cortex toolkit wiring (session 2026-08-31).
 *
 * Registers the new engineering tools built this session (in coding-cortex/) as
 * first-class pi tools so Agent invokes them during real coding work instead of
 * remembering to shell out. Complements the existing memory-tools.ts coding
 * stack (concept/capability/contract/property/mutate/simplify/codeql).
 *
 * Tools registered:
 *   reuse_search    — "are there existing implementations of this idea?"
 *   clone_detect    — structural near-clone / same-capability detection
 *   differ          — differential test a rewrite (A vs B behavior)
 *   arch_fitness    — architecture fitness functions (feedback signals)
 *   semantic_diff   — review a refactor as behavior, not lines
 *   spec_compiler   — intent -> requirements/invariants/acceptance tests
 *   patch_dossier   — evidence-based confidence for a patch
 *   blind_review    — RETRACE-style: infer problem from patch alone, reconcile
 *   concept_inventory — required/existing/missing/composition for a feature
 *   polyglot        — capability registry + language-boundary cost model
 */
import { execFile } from "node:child_process";
import { promisify } from "node:util";
import { homedir } from "node:os";
import { join } from "node:path";
import { Type } from "@earendil-works/pi-ai";
import { defineTool } from "@earendil-works/pi-coding-agent";

const execFileAsync = promisify(execFile);
const PY = join(homedir(), "venvs/memory/bin/python");
const CC = join(homedir(), "coding-cortex");

export default function (pi) {
	registerCodingTools(pi);
}

export function registerCodingTools(pi) {
	// reuse_search — positional query + --top
	const reuseTool = defineTool({
		name: "reuse_search",
		label: "Reuse search",
		description:
			"Find existing implementations of an idea across the codebase (embeds symbol name+signature, cosine-ranked). Use BEFORE writing a function: 'are there already implementations of <idea>?' e.g. 'uuid normalize' returns existing _normalize helpers. Compose: reuse_search -> reuse (adapt) -> code_graph.",
		parameters: Type.Object({
			query: Type.Array(Type.String({ description: "the idea, space-separated words" })),
			top: Type.Optional(Type.Number({ description: "results, default 10" })),
		}),
		async execute(_id, params, _signal, _onUpdate) {
			const args = [join(CC, "reuse_search.py"), ...(params.query || [])];
			if (params.top) args.push("--top", String(params.top));
			const { stdout } = await execFileAsync(PY, args, { timeout: 60000 });
			return { content: [{ type: "text", text: stdout.trim() }] };
		},
	});
	pi.registerTool(reuseTool);

	// clone_detect scan <dirs...>
	const cloneTool = defineTool({
		name: "clone_detect",
		label: "Clone / near-clone detection",
		description:
			"Detect near-duplicate functions (same capability, different names) via normalized AST shape + call signature — NOT embeddings. Surfaces copy-pasted helpers and same-idea-different-name groups for dedup. Use after building up many helpers, or to find duplicates to merge. Note: short/trivial funcs can over-flag; a group is a candidate, judgment confirms.",
		parameters: Type.Object({
			dirs: Type.Array(Type.String({ description: "directories to scan, e.g. ['~/memory']" })),
			thresh: Type.Optional(Type.Number({ description: "near-clone token threshold, default 0.55" })),
		}),
		async execute(_id, params, _signal, _onUpdate) {
			const args = [join(CC, "clone_detect.py"), "scan", ...(params.dirs || [])];
			if (params.thresh) args.push("--thresh", String(params.thresh));
			const { stdout } = await execFileAsync(PY, args, { timeout: 120000 });
			return { content: [{ type: "text", text: stdout.trim() }] };
		},
	});
	pi.registerTool(cloneTool);

	// differ <module> <fn_a> <fn_b> — differential test a rewrite
	const differTool = defineTool({
		name: "differ",
		label: "Differential test (rewrite equivalence)",
		description:
			"Prove a rewrite (B) is behaviorally equivalent to the reference (A) over deterministic generated inputs. Use when replacing a function (esp. Python->C/FFI) or merging duplicate helpers: differ <module_path_or_name> <fn_a> <fn_b>. Returns EQUIVALENT or the diverging inputs. Both functions must be importable from the same module.",
		parameters: Type.Object({
			module: Type.String({ description: "module path/name containing both functions" }),
			fn_a: Type.String({ description: "reference implementation A" }),
			fn_b: Type.String({ description: "candidate rewrite B" }),
			n: Type.Optional(Type.Number({ description: "inputs, default 1000" })),
		}),
		async execute(_id, params, _signal, _onUpdate) {
			const args = [join(CC, "differ.py"), params.module, params.fn_a, params.fn_b];
			if (params.n) args.push("--n", String(params.n));
			const { stdout } = await execFileAsync(PY, args, { timeout: 120000 });
			return { content: [{ type: "text", text: stdout.trim() }] };
		},
	});
	pi.registerTool(differTool);

	// arch_fitness check --repo <dir> — architecture fitness functions
	const fitnessTool = defineTool({
		name: "arch_fitness",
		label: "Architecture fitness",
		description:
			"Measurable architectural constraints as feedback signals: no circular deps, module complexity, public API, duplicate threshold, dependency-direction rules, max function size. Reports PASS/WARN/FAIL + evidence. Use before/after a big change. Rules arg: '<A> must not import <B>'.",
		parameters: Type.Object({
			repo: Type.Optional(Type.String({ description: "repo dir, default ~/memory" })),
			rules: Type.Optional(Type.Array(Type.String({ description: "e.g. 'mailtool must not import memory'" }))),
			skip: Type.Optional(Type.Array(Type.String({ description: "checks to skip" }))),
		}),
		async execute(_id, params, _signal, _onUpdate) {
			const args = [join(CC, "arch_fitness.py"), "check"];
			if (params.repo) args.push("--repo", params.repo);
			if (params.skip) params.skip.forEach(s => args.push("--skip", s));
			if (params.rules) params.rules.forEach(r => args.push(r));
			const { stdout } = await execFileAsync(PY, args, { timeout: 120000 });
			return { content: [{ type: "text", text: stdout.trim() }] };
		},
	});
	pi.registerTool(fitnessTool);

	// semantic_diff <old> <new>
	const semDiffTool = defineTool({
		name: "semantic_diff",
		label: "Semantic diff",
		description:
			"Review a refactor as changed BEHAVIOR, not lines: classifies each function MOVED (same body, new location) / RENAMED / REWRITTEN / ADDED / REMOVED / UNCHANGED. Use to review your own large refactors — distinguish 'function moved' from 'function rewritten'.",
		parameters: Type.Object({
			old: Type.Optional(Type.String({ description: "old file (positional)" })),
			new: Type.Optional(Type.String({ description: "new file (positional)" })),
			old_file: Type.Optional(Type.String({ description: "old file (flag form)" })),
			new_file: Type.Optional(Type.String({ description: "new file (flag form)" })),
		}),
		async execute(_id, params, _signal, _onUpdate) {
			const args = [join(CC, "semantic_diff.py")];
			if (params.old) args.push(params.old);
			if (params.new) args.push(params.new);
			if (params.old_file) args.push("--old-file", params.old_file);
			if (params.new_file) args.push("--new-file", params.new_file);
			const { stdout } = await execFileAsync(PY, args, { timeout: 60000 });
			return { content: [{ type: "text", text: stdout.trim() }] };
		},
	});
	pi.registerTool(semDiffTool);

	// spec_compiler <intent...> — intent -> contract
	const specTool = defineTool({
		name: "spec_compiler",
		label: "Spec compiler",
		description:
			"Before coding a feature, turn intent into a machine-checkable contract: behavioral requirements, invariants, acceptance tests, assumptions. Prevents solving a slightly different problem. Compose: spec_compiler -> code -> patch_dossier -> blind_review.",
		parameters: Type.Object({
			intent: Type.Array(Type.String({ description: "the request, in your own words" })),
			out: Type.Optional(Type.String({ description: "write spec JSON to a path" })),
		}),
		async execute(_id, params, _signal, _onUpdate) {
			const args = [join(CC, "spec_compiler.py"), ...(params.intent || [])];
			if (params.out) args.push("--out", params.out);
			const { stdout } = await execFileAsync(PY, args, { timeout: 120000 });
			return { content: [{ type: "text", text: stdout.trim() }] };
		},
	});
	pi.registerTool(specTool);

	// patch_dossier create/show/list — uses subcommands, so hand-written (not the
	// generic helper) to pass the leading subcommand token.
	const dossierTool = defineTool({
		name: "patch_dossier",
		label: "Patch dossier",
		description:
			"Record a patch with evidence-based confidence (claim, files, checks, assumptions). "
			+ "Actions: `create <claim> --files a.py --check lint:passed --assumption ...` derives "
			+ "confidence from evidence (start 0.5, +0.1/green, -0.2/red). `show <id>`, `list`. "
			+ "Use after every significant patch so confidence is objective, not vibes. "
			+ "Compose: spec_compiler -> code -> patch_dossier -> blind_review.",
		parameters: Type.Object({
			claim: Type.Optional(Type.Array(Type.String({ description: "what the patch solves" }))),
			files: Type.Optional(Type.Array(Type.String({ description: "changed files" }))),
			check: Type.Optional(Type.Array(Type.String({ description: "name:status e.g. lint:passed" }))),
			assumption: Type.Optional(Type.Array(Type.String({ description: "unverified assumptions" }))),
			show: Type.Optional(Type.String({ description: "show dossier id" })),
			list: Type.Optional(Type.Boolean({ description: "list dossiers" })),
		}),
		async execute(_id, params, _signal, _onUpdate) {
			const args = [join(CC, "patch_dossier.py")];
			if (params.show) {
				args.push("show", params.show);
			} else if (params.list) {
				args.push("list");
			} else {
				args.push("create", (params.claim || []).join(" "));
				if (params.files) { args.push("--files"); params.files.forEach(f => args.push(f)); }
				if (params.check) { params.check.forEach(c => args.push("--check", c)); }
				if (params.assumption) { params.assumption.forEach(a2 => args.push("--assumption", a2)); }
			}
			const { stdout, stderr } = await execFileAsync(PY, args, { timeout: 30000 });
			return { content: [{ type: "text", text: (stdout + (stderr ? "\n" + stderr : "")).trim() }] };
		},
	});
	pi.registerTool(dossierTool);

	// blind_review <patch> <intent...>
	const blindTool = defineTool({
		name: "blind_review",
		label: "Blind reviewer (RETRACE)",
		description:
			"Independent review: a fresh context infers what a patch solves FROM THE PATCH ALONE, then reconciles with the actual intent. Catches plausible-but-wrong patches. Pass the patch text (or is_path with a file) and the claimed intent. Use after authoring a significant patch.",
		parameters: Type.Object({
			patch: Type.String({ description: "the patch/diff text" }),
			intent: Type.Array(Type.String({ description: "the claimed intent" })),
			is_path: Type.Optional(Type.Boolean({ description: "patch is a file path" })),
		}),
		async execute(_id, params, _signal, _onUpdate) {
			const args = [join(CC, "blind_review.py")];
			args.push("--patch", params.patch);
			if (params.is_path) args.push("--is-path");
			if (params.intent) { args.push("--intent"); params.intent.forEach(i => args.push(i)); }
			const { stdout } = await execFileAsync(PY, args, { timeout: 120000 });
			return { content: [{ type: "text", text: stdout.trim() }] };
		},
	});
	pi.registerTool(blindTool);

	// concept_inventory <spec...> — required/existing/missing/composition
	const invTool = defineTool({
		name: "concept_inventory",
		label: "Concept inventory",
		description:
			"Before coding a feature, check each capability in a pipeline ('stream CSV -> normalize -> deduplicate -> database') against the reuse index + polyglot registry: reports required/existing/partial/missing/composition. Goal: zero new algorithms, just composition.",
		parameters: Type.Object({
			spec: Type.Array(Type.String({ description: "the capability pipeline, e.g. 'a -> b -> c'" })),
			top: Type.Optional(Type.Number({ description: "results per step, default 3" })),
		}),
		async execute(_id, params, _signal, _onUpdate) {
			const args = [join(CC, "concept_inventory.py"), ...(params.spec || [])];
			if (params.top) args.push("--top", String(params.top));
			const { stdout } = await execFileAsync(PY, args, { timeout: 60000 });
			return { content: [{ type: "text", text: stdout.trim() }] };
		},
	});
	pi.registerTool(invTool);

	// polyglot capabilities/best/decide
	const polyglotTool = defineTool({
		name: "polyglot",
		label: "Polyglot advisor",
		description:
			"Capability registry (what each language is unusually good at, as concrete properties) + language-boundary cost model (introduce a language only if benefit > interop + maintenance). Actions: `capabilities <lang>`, `best <task>`, `decide <lang> <task> --benefit .. --interop .. --maint ..`. Use to justify introducing a language, not prefer one.",
		parameters: Type.Object({
			capabilities: Type.Optional(Type.String({ description: "show concrete strengths of a language" })),
			best: Type.Optional(Type.Array(Type.String({ description: "rank languages for a task" }))),
			decide: Type.Optional(Type.Array(Type.String({ description: "language then task" }))),
			benefit: Type.Optional(Type.Number({ description: "0..1 benefit (with decide)" })),
			interop: Type.Optional(Type.Number({ description: "0..1 interop cost" })),
			maint: Type.Optional(Type.Number({ description: "0..1 maintenance cost" })),
		}),
		async execute(_id, params, _signal, _onUpdate) {
			const args = [join(CC, "polyglot.py")];
			if (params.capabilities) args.push("capabilities", params.capabilities);
			else if (params.best) args.push("best", ...(params.best));
			else if (params.decide) {
				args.push("decide", ...(params.decide));
				if (params.benefit) args.push("--benefit", String(params.benefit));
				if (params.interop) args.push("--interop", String(params.interop));
				if (params.maint) args.push("--maint", String(params.maint));
			}
			const { stdout } = await execFileAsync(PY, args, { timeout: 60000 });
			return { content: [{ type: "text", text: stdout.trim() }] };
		},
	});
	pi.registerTool(polyglotTool);

	// forked_review — independent instances w/ different objectives + evidence adjudication
	const forkedTool = defineTool({
		name: "forked_review",
		label: "Forked-perspective engineering review",
		description:
			"Independent Agent instances with deliberately DIFFERENT objectives (correctness, simplicity, security, reuse, performance, architecture) review the same proposed architectural change, then an adjudicator decides by EVIDENCE STRENGTH, not majority vote. Use before adopting a non-trivial change: it surfaces objective-specific risks a single-perspective review misses, and a single concrete security finding outweighs three approvals.",
		parameters: Type.Object({
			change: Type.Array(Type.String({ description: "the proposed architectural change" })),
			forks: Type.Optional(Type.String({ description: "comma-separated objectives (default correctness,simplicity,security,reuse,performance)" })),
			context: Type.Optional(Type.String({ description: "optional context/transcript" })),
		}),
		async execute(_id, params, _signal, _onUpdate) {
			const args = [join(CC, "forked_review.py"), ...(params.change || [])];
			if (params.forks) args.push("--forks", params.forks);
			if (params.context) args.push("--context", params.context);
			const { stdout } = await execFileAsync(PY, args, { timeout: 300000 });
			return { content: [{ type: "text", text: stdout.trim() }] };
		},
	});
	pi.registerTool(forkedTool);

	// model_registry — queryable model-performance registry + routing
	const regTool = defineTool({
		name: "model_registry",
		label: "Model-performance registry",
		description:
			"Queryable registry of model x thinking-level performance (seeded from the engine-matrix results). Use to make model selection DYNAMIC, not 'Flash is Agent's model': 'best <task>' ranks engines by evidence (chained% is the true discriminator). Actions: list, ingest, best <task>.",
		parameters: Type.Object({
			best: Type.Optional(Type.Array(Type.String({ description: "rank engines for a task" }))),
			list: Type.Optional(Type.Boolean({ description: "list all rows" })),
			ingest: Type.Optional(Type.Boolean({ description: "seed from engine-matrix results" })),
		}),
		async execute(_id, params, _signal, _onUpdate) {
			const args = [join(CC, "model_registry.py")];
			if (params.best) { args.push("best", ...(params.best)); }
			else if (params.list) { args.push("list"); }
			else if (params.ingest) { args.push("ingest"); }
			const { stdout } = await execFileAsync(PY, args, { timeout: 30000 });
			return { content: [{ type: "text", text: stdout.trim() }] };
		},
	});
	pi.registerTool(regTool);
}
