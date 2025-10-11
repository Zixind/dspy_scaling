import logging
from collections import defaultdict
import copy
import dspy
from dspy.evaluate.evaluate import Evaluate
from dspy.signatures import Signature
from dspy.teleprompt.teleprompt import Teleprompter
import random
import numpy as np
logger = logging.getLogger(__name__)
import sys, logging
try:
    import wandb
except Exception:
    wandb = None
import textwrap, wandb

def _short(s: str, n=160):
    s = " ".join(s.split())  # collapse whitespace
    return (s[:n] + " …") if len(s) > n else s
def setup_wandb_logging(level=logging.INFO):
    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    root = logging.getLogger()
    root.setLevel(level)

    # avoid duplicate handlers on re-import
    for h in list(root.handlers):
        root.removeHandler(h)

    # 1) stdout (W&B agent captures stdout)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    root.addHandler(sh)

    # 2) W&B termlog handler (shows up in Logs tab immediately)
    if wandb is not None:
        class WandbHandler(logging.Handler):
            def emit(self, record):
                try:
                    wandb.termlog(self.format(record))
                except Exception:
                    pass
        wh = WandbHandler()
        wh.setLevel(level)
        wh.setFormatter(fmt)
        root.addHandler(wh)

"""
USAGE SUGGESTIONS:

The following code can be used to compile a optimized signature teleprompter, and evaluate it on an end task:

teleprompter = COPRO(prompt_model=prompt_model, metric=metric, breadth=BREADTH, depth=DEPTH, init_temperature=INIT_TEMPERATURE)
kwargs = dict(num_threads=NUM_THREADS, display_progress=True, display_table=0)
compiled_prompt_opt = teleprompter.compile(program.deepcopy(), trainset=trainset[:DEV_NUM], eval_kwargs=kwargs)
eval_score = evaluate(compiled_prompt_opt, devset=evalset[:EVAL_NUM], **kwargs)

Note that this teleprompter takes in the following parameters:

* prompt_model: The model used for prompt generation. When unspecified, defaults to the model set in settings (ie. dspy.settings.configure(lm=task_model)).
* metric: The task metric used for optimization.
* breadth: The number of new prompts to generate at each iteration. Default=10.
* depth: The number of times we should ask our prompt model to generate new prompts, with the history of the past prompts as input. Default=3.
* init_temperature: The temperature used to generate new prompts. Higher roughly equals more creative. Default=1.4.
* track_stats: Tells the method whether or not to track statistics about the optimization process.
                If True, the method will track the following statistics:
                    * results_best: The min,max,avg,stddev of top 10 scores for each predictor at each depth.
                    * results_latest: The min,max,avg,stddev of newest prompt scores for each predictor at each depth.
                    * total_calls: The total number of calls to the task metric.
                These statistics will be returned as attributes of the best program.
"""


class BasicGenerateInstruction(Signature):
    """You are an instruction optimizer for large language models. I will give you a ``signature`` of fields (inputs and outputs) in English. Your task is to propose an instruction that will lead a good language model to perform the task well. Don't be afraid to be creative."""

    basic_instruction = dspy.InputField(desc="The initial instructions before optimization")
    proposed_instruction = dspy.OutputField(desc="The improved instructions for the language model")
    proposed_prefix_for_output_field = dspy.OutputField(
        desc="The string at the end of the prompt, which will help the model start solving the task",
    )


class GenerateInstructionGivenAttempts(dspy.Signature):
    """You are an instruction optimizer for large language models. I will give some task instructions I've tried, along with their corresponding validation scores. The instructions are arranged in increasing order based on their scores, where higher scores indicate better quality.

    Your task is to propose a new instruction that will lead a good language model to perform the task even better. Don't be afraid to be creative."""

    attempted_instructions = dspy.InputField()
    proposed_instruction = dspy.OutputField(desc="The improved instructions for the language model")
    proposed_prefix_for_output_field = dspy.OutputField(
        desc="The string at the end of the prompt, which will help the model start solving the task",
    )

class GenerateInstructionGivenAttempts_Single_Generation(dspy.Signature):
    """You are an instruction optimizer for large language models. I will give you a task instruction I've tried, along with its validation scores. The instruction is the BEST instruction I have obtained so far.

    Your task is to propose a new instruction that will lead a good language model to perform the task even better. Don't be afraid to be creative."""

    attempted_instructions = dspy.InputField()
    proposed_instruction = dspy.OutputField(desc="The improved instructions for the language model")
    proposed_prefix_for_output_field = dspy.OutputField(
        desc="The string at the end of the prompt, which will help the model start solving the task",
    )


class COPRO(Teleprompter):
    def __init__(
        self,
        prompt_model=None, #produce the prompt
        metric=None,
        breadth=10,
        depth=3,
        init_temperature=1.4,
        track_stats=False,
        **_kwargs,
    ):
        if breadth <= 1:
            raise ValueError("Breadth must be greater than 1")   #Zixin modifies
        self.metric = metric
        self.breadth = breadth
        self.depth = depth
        self.init_temperature = init_temperature
        self.prompt_model = prompt_model
        self.track_stats = track_stats

    def _check_candidates_equal(self, candidate1, candidate2):
        for p1, p2 in zip(candidate1["program"].predictors(), candidate2["program"].predictors(), strict=False):
            if self._get_signature(p1).instructions != self._get_signature(p2).instructions:
                return False
            *_, p1_last_field = self._get_signature(p1).fields.values()
            *_, p2_last_field = self._get_signature(p2).fields.values()
            if p1_last_field != p2_last_field:
                return False
        return True

    def _drop_duplicates(self, candidates):
        final_candidates = []
        last_batch = []
        last_batch_score = -1
        for c in candidates:
            repeat = False
            if c["score"] == last_batch_score:
                for c2 in last_batch:
                    if self._check_candidates_equal(c, c2):
                        repeat = True
                        break
                if not repeat:
                    last_batch.append(c)
            else:
                last_batch = [c]
                last_batch_score = c["score"]
            if not repeat:
                final_candidates.append(c)
        return final_candidates

    def _print_signature(self, predictor):
        signature = self._get_signature(predictor)

        logger.debug(f"i: {signature.instructions}")
        logger.debug(f"p: {list(signature.fields.values())[-1].json_schema_extra['prefix']}")

    def _get_signature(self, predictor):
        assert hasattr(predictor, "signature")
        return predictor.signature

    def _set_signature(self, predictor, updated_signature):
        assert hasattr(predictor, "signature")
        predictor.signature = updated_signature

    def compile(self, student, *, trainset, eval_kwargs):
        """
        Optimizes the signatures of `student` by searching over instructions/prefixes.

        Args:
            student: DSPy Program to optimize (mutated during search).
            trainset: iterable of Examples used for scoring.
            eval_kwargs: dict of kwargs forwarded to Evaluate(...).

        Returns:
            best_program: a fresh Program of the same class with each predictor set
                      to its best (instruction, prefix), plus some stats attached.
        """
        setup_wandb_logging()
        module = student  # work in-place; avoid deepcopy
        evaluate = Evaluate(devset=trainset, metric=self.metric, **eval_kwargs)
        total_calls = 0

        if self.track_stats:
            import numpy as np
            results_best = {id(p): {"depth": [], "max": [], "average": [], "min": [], "std": []}
                            for p in module.predictors()}
            results_latest = {id(p): {"depth": [], "max": [], "average": [], "min": [], "std": []}
                            for p in module.predictors()}

        # Helper: set a predictor's signature (instruction, prefix)
        def _apply_sig(pred, instruction, prefix):
            *_, last_key = self._get_signature(pred).fields.keys()
            updated = (
                self._get_signature(pred)
                .with_instructions(instruction)
                .with_updated_fields(last_key, prefix=prefix)
            )
            self._set_signature(pred, updated)

        # Seed candidates and remember each predictor's initial "best" config
        candidates = {}
        evaluated_candidates = defaultdict(dict)  # per-predictor: {(instr, prefix): {...}}
        best_config = {}  # per-predictor id -> (instr, prefix)

        for predictor in module.predictors():
            # read baseline
            *_, last_key = self._get_signature(predictor).fields.keys()
            basic_instruction = self._get_signature(predictor).instructions
            basic_prefix = self._get_signature(predictor).fields[last_key].json_schema_extra["prefix"]
            best_config[id(predictor)] = (basic_instruction, basic_prefix)

            # generate breadth-1 new prompts from prompt_model (plus the baseline)
            if self.prompt_model:
                with dspy.settings.context(lm=self.prompt_model):
                    instruct = dspy.Predict(
                        BasicGenerateInstruction,
                        n=self.breadth - 1,
                        temperature=self.init_temperature,
                    )(basic_instruction=basic_instruction)
            else:
                instruct = dspy.Predict(
                    BasicGenerateInstruction,
                    n=self.breadth - 1,
                    temperature=self.init_temperature,
                )(basic_instruction=basic_instruction)

            # include baseline as a candidate
            instruct.completions.proposed_instruction.append(basic_instruction)
            instruct.completions.proposed_prefix_for_output_field.append(basic_prefix)

            candidates[id(predictor)] = instruct.completions
            evaluated_candidates[id(predictor)] = {}

        if self.prompt_model:
            logger.debug(f"{self.prompt_model.inspect_history(n=1)}")

        latest_candidates = candidates
        all_candidates = candidates  # grows over depth for multi-predictor programs

        # ==== Search over depth ====
        for d in range(self.depth):
            logger.info(f"Iteration Depth: {d+1}/{self.depth}.")

            # Iterate predictors; keep others fixed at their current best while testing this one's candidates
            for p_i, p in enumerate(module.predictors()):
                # Choose the candidate pool to test for this predictor
                cand_pool = latest_candidates[id(p)]
                if len(module.predictors()) > 1:
                    cand_pool = all_candidates[id(p)]

                latest_scores = []

                for c_i, c in enumerate(cand_pool):
                    instruction = c.proposed_instruction.strip('"').strip()
                    prefix = c.proposed_prefix_for_output_field.strip('"').strip()

                    # Apply candidate for current predictor; others use their best_config
                    for q in module.predictors():
                        instr_q, pref_q = best_config[id(q)]
                        if q is p:
                            _apply_sig(q, instruction, prefix)
                        else:
                            _apply_sig(q, instr_q, pref_q)

                    # Optional debug print
                    for idx, pred_dbg in enumerate(module.predictors()):
                        logger.debug(f"Predictor {idx+1}")
                        self._print_signature(pred_dbg)

                    logger.info(
                        f"At Depth {d+1}/{self.depth}, Evaluating Prompt Candidate "
                        f"#{c_i+1}/{len(cand_pool)} for Predictor {p_i+1} of {len(module.predictors())}."
                    )
                    result = evaluate(module, devset=trainset, **eval_kwargs)
                    score = result.score
                    if self.prompt_model:
                        logger.debug(f"prompt_model.inspect_history(n=1) {self.prompt_model.inspect_history(n=1)}")
                    total_calls += 1

                    # Keep the best score per (instruction, prefix)
                    key = (instruction, prefix)
                    prev = evaluated_candidates[id(p)].get(key)
                    if prev is None or prev["score"] < score:
                        evaluated_candidates[id(p)][key] = {
                            "score": score,
                            "instruction": instruction,
                            "prefix": prefix,
                            "depth": d,
                        }

                    # track only the newest breadth scores (for stats)
                    if len(cand_pool) - self.breadth <= c_i:
                        latest_scores.append(score)

                # Track "latest" stats
                if self.track_stats and latest_scores:
                    results_latest[id(p)]["depth"].append(d)
                    results_latest[id(p)]["max"].append(max(latest_scores))
                    results_latest[id(p)]["average"].append(sum(latest_scores) / len(latest_scores))
                    results_latest[id(p)]["min"].append(min(latest_scores))
                    results_latest[id(p)]["std"].append(np.std(latest_scores))

                # Update greedy best config for this predictor
                best_for_p = max(evaluated_candidates[id(p)].values(), key=lambda cc: cc["score"])
                best_config[id(p)] = (best_for_p["instruction"], best_for_p["prefix"])

                logger.debug(
                    f"Updating Predictor {id(p)} to:\n"
                    f"i: {best_for_p['instruction']}\n"
                    f"p: {best_for_p['prefix']}"
                )

            # Stop before generating next wave
            if d == self.depth - 1:
                break

            # === Generate next batch from attempts (top-K history) ===
            new_candidates = {}
            for p_base in module.predictors():
                # Build attempts list from best-so-far evaluated candidates
                best_list = sorted(
                    evaluated_candidates[id(p_base)].values(),
                    key=lambda x: x["score"],
                    reverse=True,
                )
                k = min(len(best_list), self.breadth) #breadth
                attempts = []
                # Put k items, lower-scored to higher-scored (as your original did)
                for i in range(k - 1, -1, -1):
                    attempts.append(f'Instruction #{k-i}: {best_list[i]["instruction"]}')
                    attempts.append(f'Prefix #{k-i}: {best_list[i]["prefix"]}')
                    attempts.append(f'Resulting Score #{k-i}: {best_list[i]["score"]}')

                if self.track_stats and best_list:
                    top10 = [x["score"] for x in best_list[:10]]
                    results_best[id(p_base)]["depth"].append(d)
                    results_best[id(p_base)]["max"].append(max(top10))
                    results_best[id(p_base)]["average"].append(sum(top10) / len(top10))
                    results_best[id(p_base)]["min"].append(min(top10))
                    results_best[id(p_base)]["std"].append(np.std(top10))

                # Propose next prompts conditioned on attempts
                if self.prompt_model:
                    with dspy.settings.context(lm=self.prompt_model):
                        instr = dspy.Predict(
                            GenerateInstructionGivenAttempts,
                            n=self.breadth,
                            temperature=self.init_temperature,
                        )(attempted_instructions=attempts)
                else:
                    instr = dspy.Predict(
                        GenerateInstructionGivenAttempts,
                        n=self.breadth,
                        temperature=self.init_temperature,
                    )(attempted_instructions=attempts)

                # Save/accumulate
                new_candidates[id(p_base)] = instr.completions
                all_candidates[id(p_base)].proposed_instruction.extend(instr.completions.proposed_instruction)
                all_candidates[id(p_base)].proposed_prefix_for_output_field.extend(
                    instr.completions.proposed_prefix_for_output_field
                )

            latest_candidates = new_candidates

    # ==== Final selection & return a fresh program ====
    # Flatten all evaluated candidates just for reporting/debug
        flat = []
        for predictor in module.predictors():
            flat.extend(list(evaluated_candidates[id(predictor)].values()))

        # Deduplicate by (instruction, prefix) while keeping best score
        seen = {}
        for c in flat:
            k = (c["instruction"], c["prefix"])
            if k not in seen or seen[k]["score"] < c["score"]:
                seen[k] = c
        candidates_flat = sorted(seen.values(), key=lambda x: x["score"], reverse=True)

        # Apply each predictor's own best config to the module and return it
        for p in module.predictors():
            instr, pref = best_config[id(p)]
            _apply_sig(p, instr, pref)

        # Attach diagnostics
        module.candidate_programs = candidates_flat
        module.total_calls = total_calls
        if self.track_stats:
            module.results_best = results_best
            module.results_latest = results_latest

        return module
    
    def compile_gumbel_top_k(self, student, *, trainset, minibatch_val_size=None, rng_seed = 1, k_top = None, beta = 1.0, Promptwise_Generation = True, eval_kwargs):
        """
        Optimizes the signatures of `student` by searching over instructions/prefixes.

        Args:
            student: DSPy Program to optimize (mutated during search).
            trainset: iterable of Examples used for scoring.
            eval_kwargs: dict of kwargs forwarded to Evaluate(...).
            minibatch_val_size: if set (int), evaluate on a random mini-batch of this size per depth.
                        A new batch is drawn at every depth; all candidates in that depth share it.
            rng_seed: base seed for reproducible per-depth mini-batch sampling.


        Returns:
            best_program: a fresh Program of the same class with each predictor set
                      to its best (instruction, prefix), plus some stats attached.
        """
        setup_wandb_logging()
        import numpy as np
        module = student  # work in-place; avoid deepcopy
        
        total_calls = 0
        import numpy as np
        rng_draw = np.random.default_rng(rng_seed)
        # Materialize once so we can sample indices reproducibly
        if k_top is None:
            k_top = self.breadth
        else:
            k_top = int(k_top)
        print('We are using Gumbel Top K with K = {}'.format(k_top))
        reevaluate_size = minibatch_val_size


        trainset_list = list(trainset)
        evaluate = Evaluate(devset=trainset, metric=self.metric, **eval_kwargs)

        if self.track_stats:
            import numpy as np
            results_best = {id(p): {"depth": [], "max": [], "average": [], "min": [], "std": []}
                            for p in module.predictors()}
            results_latest = {id(p): {"depth": [], "max": [], "average": [], "min": [], "std": []}
                            for p in module.predictors()}

        # Helper: set a predictor's signature (instruction, prefix)
        def _apply_sig(pred, instruction, prefix):
            *_, last_key = self._get_signature(pred).fields.keys()
            updated = (
                self._get_signature(pred)
                .with_instructions(instruction)
                .with_updated_fields(last_key, prefix=prefix)
            )
            self._set_signature(pred, updated)
        
        def _pairs_from_completions(comps):
            # Convert DSPy completions to a list[(instr, prefix)] to prevent cycles.
            return [
                (ins.strip('"').strip(), pref.strip('"').strip())
                    for ins, pref in zip(
                    comps.proposed_instruction,
                    comps.proposed_prefix_for_output_field,
                )
            ]

        

        # Seed candidates and remember each predictor's initial "best" config
        candidates = {}
        latest_candidates = {}
        all_candidates = {}
        evaluated_candidates = defaultdict(dict)  # per-predictor: {(instr, prefix): {...}}
        best_config = {}  # per-predictor id -> (instr, prefix)
        best_records = {}         # NEW: per-predictor id -> dict with text+score

        # helpful: stable names for predictors
        predictor_names = {id(p): (getattr(p, "name", None) or f"predictor_{i+1}")
                       for i, p in enumerate(module.predictors())}


        for predictor in module.predictors():
            # read baseline
            *_, last_key = self._get_signature(predictor).fields.keys()
            basic_instruction = self._get_signature(predictor).instructions
            basic_prefix = self._get_signature(predictor).fields[last_key].json_schema_extra["prefix"]
            best_config[id(predictor)] = (basic_instruction, basic_prefix)
            best_records[id(predictor)] = {           # NEW: seed the record
                "instruction": basic_instruction,
                "prefix": basic_prefix,
                "mean": None,                         # filled later
                "n": 0,
            }


            # generate breadth-1 new prompts from prompt_model (plus the baseline)
            if self.prompt_model:
                with dspy.settings.context(lm=self.prompt_model):
                    instruct = dspy.Predict(
                        BasicGenerateInstruction,
                        n=self.breadth - 1,
                        temperature=self.init_temperature,
                    )(basic_instruction=basic_instruction)
            else:
                instruct = dspy.Predict(
                    BasicGenerateInstruction,
                    n=self.breadth - 1,
                    temperature=self.init_temperature,
                )(basic_instruction=basic_instruction)

            pool = _pairs_from_completions(instruct.completions)
            pool.append((basic_instruction, basic_prefix))  # include baseline

            candidates[id(predictor)] = list(pool)
            latest_candidates[id(predictor)] = list(pool)
            all_candidates[id(predictor)] = list(pool)
            evaluated_candidates[id(predictor)] = {}

            # ==== Search over depth ====
        for d in range(self.depth):
            if minibatch_val_size is None or minibatch_val_size >= len(trainset_list):
                dev_batch = trainset_list
            else:
                import numpy as np
                dev_batch = self.subsample(trainset_list,k = minibatch_val_size, rng = rng_draw)
            logger.info(f"Iteration Depth: {d+1}/{self.depth}.")

            # Iterate predictors; keep others fixed at their current best while testing this one's candidates
            for p_i, p in enumerate(module.predictors()):
                cand_pool = latest_candidates.get(id(p), [])
                latest_scores = []

                for (instruction, prefix) in cand_pool:
                    # apply this candidate for current predictor, others fixed
                    for q in module.predictors():
                        instr_q, pref_q = best_config[id(q)]
                        if q is p:
                            _apply_sig(q, instruction, prefix)
                        else:
                            _apply_sig(q, instr_q, pref_q)
                    print(
                        'Instruction {}'.format(instruction),
                        'Prefix {}'.format(prefix)
                    )
                    wandb.termlog(
                        f"[COPRO Gumbel Top K][depth {d+1}][pred {p_i+1}/{len(module.predictors())}] "
                        f"Eval cand #{len(latest_scores)+1}/{len(cand_pool)} | "
                        f"instruction: {_short(instruction)} | prefix: {_short(prefix)}"
                    )
                    result = evaluate(module, devset=dev_batch, **eval_kwargs)
                    score = result.score
                    wandb.termlog(
                        f"[COPRO Gumbel Top K][depth {d+1}][pred {p_i+1}] score={score:.4f}"
                    )
                    total_calls += 1
                    key = (instruction, prefix)
                    prev = evaluated_candidates[id(p)].get(key)
                    if prev is None:
                        evaluated_candidates[id(p)][key] = {
                            "instruction": instruction,
                            "prefix": prefix,
                            "depth_first_seen": d,
                            "score": score,                 # latest/best scalar seen on any batch
                            "rs": self._rs_init(score),     # running stats (n, mean, sumsq)
                        }
                    else:
                        # keep a latest/best scalar
                        prev["score"] = max(prev.get("score", float("-inf")), score)
                    latest_scores.append(score)

                # Track "latest" stats
                if self.track_stats and latest_scores:
                    import numpy as _np
                    rb = results_latest[id(p)]
                    rb["depth"].append(d)
                    rb["max"].append(max(latest_scores))
                    rb["average"].append(sum(latest_scores) / len(latest_scores))
                    rb["min"].append(min(latest_scores))
                    rb["std"].append(_np.std(latest_scores))


                # ---------- Gumbel-Top-K selection on current cand_pool ----------
                # Gather the UNIQUE candidates tested for this predictor up to this depth
                cand_items = list(evaluated_candidates[id(p)].values())  # all depths so far
                if not cand_items:
                    # fallback: greedy by mean/score
                    best_for_p = max(
                        evaluated_candidates[id(p)].values(),
                        key=lambda cc: cc["rs"]["mean"] if "rs" in cc else cc["score"],
                    )
                    best_config[id(p)] = (best_for_p["instruction"], best_for_p["prefix"])
                    # NEW: also store a clean record for this predictor
                    best_records[id(p)]["instruction"] = best_for_p["instruction"]
                    best_records[id(p)]["prefix"] = best_for_p["prefix"]
                    best_records[id(p)]["mean"] = float(best_for_p.get("rs", {}).get("mean",
                                                      best_for_p.get("score", None)))
                    best_records[id(p)]["n"] = int(best_for_p.get("rs", {}).get("n", 1))
                else:
                    scores_now = np.array(
                        [ci["rs"]["mean"] if "rs" in ci else ci["score"] for ci in cand_items],
                        dtype=float,
                    )
                    k = min(k_top, len(cand_items))
                    topk_idx = self._gumbel_top_k(scores_now, k=k, beta=beta, rng=rng_draw)
                    wandb.termlog(f"[COPRO] Top-K indices: {topk_idx}")
                    topk_items = [cand_items[i] for i in topk_idx]

                    # freeze membership for next gen
                    for ci in topk_items:
                        ci["selected_for_next_gen"] = True

                    # fresh minibatch for re-evaluation of Top-K
                    dev_batch_fresh = (
                        trainset_list
                        if reevaluate_size is None or reevaluate_size >= len(trainset_list)
                        else self.subsample(trainset_list, k=reevaluate_size, rng=rng_draw)
                    )

                    # re-evaluate only Top-K and update running stats
                    for ci in topk_items:
                        for q in module.predictors():
                            instr_q, pref_q = best_config[id(q)]
                            if q is p:
                                _apply_sig(q, ci["instruction"], ci["prefix"])
                            else:
                                _apply_sig(q, instr_q, pref_q)
                        res_fresh = evaluate(module, devset=dev_batch_fresh, **eval_kwargs)
                        self.update_stats(ci["rs"], res_fresh.score)
                        ci["score"] = max(ci.get("score", float("-inf")), res_fresh.score)

                    # update best_config using UPDATED means
                    best_for_p = max(
                        evaluated_candidates[id(p)].values(),
                        key=lambda cc: cc["rs"]["mean"] if "rs" in cc else cc["score"],
                    )
                    best_config[id(p)] = (best_for_p["instruction"], best_for_p["prefix"])

            # Stop before generating next wave
            if d == self.depth - 1:
                break

            # === Generate next batch from attempts (top-K history) ===
            new_candidates = {}
            for p_base in module.predictors():
                pool = list(evaluated_candidates[id(p_base)].values())
                pool_sorted = sorted(
                    pool,
                    key=lambda v: (v["rs"]["mean"] if "rs" in v else v["score"]),
                    reverse=True,
                )
                pool_sorted = pool_sorted[:k_top]  # parents = global Top-K

                attempts = []
                # Put k items, lower-scored to higher-scored (as your original did)
                k_use = len(pool_sorted)
                for i in range(k_use - 1, -1, -1):
                    v = pool_sorted[i]
                    mean_val = v["rs"]["mean"] if "rs" in v else v["score"]
                    attempts.append(f'Instruction #{k_use-i}: {v["instruction"]}')
                    attempts.append(f'Prefix #{k_use-i}: {v["prefix"]}')
                    attempts.append(f'Resulting Mean Score #{k_use-i}: {mean_val:.4f}')

                if self.track_stats and pool_sorted:
                    import numpy as _np
                    vals = [(v["rs"]["mean"] if "rs" in v else v["score"]) for v in pool_sorted]
                    rb = results_best[id(p_base)]
                    rb["depth"].append(d)
                    rb["max"].append(max(vals))
                    rb["average"].append(sum(vals) / len(vals))
                    rb["min"].append(min(vals))
                    rb["std"].append(_np.std(vals))

                # Propose next prompts conditioned on attempts
                if Promptwise_Generation:
                    top1 = pool_sorted[0]
                    attempts_top1 = [
                        f'Instruction #0: {top1["instruction"]}',
                        f'Prefix #0: {top1["prefix"]}',
                        f'Resulting Mean Score #0: {(top1["rs"]["mean"] if "rs" in top1 else top1["score"]):.4f}',
                    ]

                    if self.prompt_model:
                        with dspy.settings.context(lm=self.prompt_model):
                            instr = dspy.Predict(
                                GenerateInstructionGivenAttempts_Single_Generation,
                                n=self.breadth,
                                temperature=self.init_temperature,
                            )(attempted_instructions=attempts_top1)
                    else:
                        instr = dspy.Predict(
                            GenerateInstructionGivenAttempts_Single_Generation,
                            n=self.breadth,
                            temperature=self.init_temperature,
                        )(attempted_instructions=attempts_top1)
                else:
                    attempts_topk = [[] for _ in range(k_use)]   # one independent bucket per top-k item
                    for i in range(k_use - 1, -1, -1):
                        v = pool_sorted[i]
                        attempts_topk[i].append(f'Instruction #{k_use-i}: {v["instruction"]}')
                        attempts_topk[i].append(f'Prefix #{k_use-i}: {v["prefix"]}')
                        attempts_topk[i].append(f'Resulting Mean Score #{k_use-i}: {v["rs"]["mean"]:.4f}') 

                    if self.prompt_model:
                        with dspy.settings.context(lm=self.prompt_model):
                            instr = dspy.Predict(
                                GenerateInstructionGivenAttempts,
                                n=self.breadth,
                                temperature=self.init_temperature,
                            )(attempted_instructions=attempts)
                    else:
                        instr = dspy.Predict(
                            GenerateInstructionGivenAttempts,
                            n=self.breadth,
                            temperature=self.init_temperature,
                        )(attempted_instructions=attempts)

                # clear the freeze flags for next depth
                for v in evaluated_candidates[id(p)].values():
                    if "selected_for_next_gen" in v:
                        del v["selected_for_next_gen"]
                # store only tuples for new candidates
                new_pairs = _pairs_from_completions(instr.completions)
                new_candidates[id(p_base)] = list(new_pairs)
                all_candidates[id(p_base)].extend([c for c in new_pairs
                                   if c not in set(all_candidates[id(p_base)])])

            latest_candidates = new_candidates

        # ==== Final selection & return a fresh program ====
        # Flatten all evaluated candidates just for reporting/debug
        flat = []
        for predictor in module.predictors():
            flat.extend(list(evaluated_candidates[id(predictor)].values()))

        # dedupe by (instruction, prefix) keeping the best mean/score
        seen = {}
        for c in flat:
            k = (c["instruction"], c["prefix"])
            val = c["rs"]["mean"] if "rs" in c else c["score"]
            if k not in seen or (seen[k]["rs"]["mean"] if "rs" in seen[k] else seen[k]["score"]) < val:
                seen[k] = c
        candidates_flat = sorted(
            seen.values(),
            key=lambda x: (x["rs"]["mean"] if "rs" in x else x["score"]),
            reverse=True,
        )

        # Apply each predictor's own best config to the module and return it
        for p in module.predictors():
            instr, pref = best_config[id(p)]
            _apply_sig(p, instr, pref)

        # diagnostics: store only primitives
        module.candidate_programs = [
            {
                "instruction": c["instruction"],
                "prefix": c["prefix"],
                "mean": c.get("rs", {}).get("mean", c.get("score", None)),
                "n": c.get("rs", {}).get("n", 1),
            }
            for c in candidates_flat
        ]
        module.total_calls = total_calls
        if self.track_stats:
            module.results_best = results_best
            module.results_latest = results_latest
            # ---- Best prompt(s) by HIGHEST MEAN over full history ----
        def _mean_or_score(c):
            return c.get("rs", {}).get("mean", c.get("score", float("-inf")))

        best_by_mean = {}
        for idx, p in enumerate(module.predictors()):
            pid = id(p)
            pool = list(evaluated_candidates[pid].values())
            if not pool:
                continue
            ci = max(pool, key=_mean_or_score)  # highest mean
            name = predictor_names[pid]
            best_by_mean[name] = {
                "instruction": ci["instruction"],
                "prefix": ci["prefix"],
                "mean": float(ci.get("rs", {}).get("mean", ci.get("score", None))),
                "n": int(ci.get("rs", {}).get("n", 1)),
            }
        
        # 2) **Force-apply** the highest-mean prompts before returning
        #    -> ensures any external Evaluate(optimized_module) uses these prompts.
        ordered_names = [predictor_names[id(p)] for p in module.predictors()]
        for p, name in zip(module.predictors(), ordered_names):
            rec = best_by_mean.get(name)
            if rec is not None:
                _apply_sig(p, rec["instruction"], rec["prefix"])
            else:
                # fallback to best_config if we had no records (shouldn't happen normally)
                instr, pref = best_config[id(p)]
                _apply_sig(p, instr, pref)

        # 3) diagnostics & attachments
        flat = []
        for predictor in module.predictors():
            flat.extend(list(evaluated_candidates[id(predictor)].values()))

        seen = {}
        for c in flat:
            k = (c["instruction"], c["prefix"])
            val = c.get("rs", {}).get("mean", c.get("score", None))
            if (k not in seen) or (seen[k].get("rs", {}).get("mean", seen[k].get("score", -1e9)) < val):
                seen[k] = c
        candidates_flat = sorted(
            seen.values(),
            key=lambda x: (x.get("rs", {}).get("mean", x.get("score", -1e9))),
            reverse=True,
        )

        module.candidate_programs = [
            {
                "instruction": c["instruction"],
                "prefix": c["prefix"],
                "mean": c.get("rs", {}).get("mean", c.get("score", None)),
                "n": c.get("rs", {}).get("n", 1),
            }
            for c in candidates_flat
        ]
        module.total_calls = total_calls
        if self.track_stats:
            module.results_best = results_best
            module.results_latest = results_latest

        module.best_prompt_by_mean = best_by_mean
        module.best_prompt_text_by_mean = "\n\n".join(
            [
                (
                    f"### {name}\n"
                    f"Instruction:\n{rec['instruction']}\n\n"
                    f"Prefix:\n{rec['prefix']}\n\n"
                    f"MeanScore: {rec['mean']} (n={rec['n']})"
                )
                for name, rec in best_by_mean.items()
            ]
        )
        if len(module.predictors()) == 1 and best_by_mean:
            only_name = next(iter(best_by_mean))
            module.best_prompt_single = best_by_mean[only_name]

        return module

    
    def subsample(self, data, k, rng=1):
        idx = rng.choice(len(data), size=k, replace=False)
        print(f"indices: {idx.tolist()}")
        return [data[i] for i in idx]
    
    def _gumbel_top_k(self, scores, k, beta=1.0, rng=None):
        """
        scores: array of raw scores (float)
        k: number of winners
        beta: inverse temperature (larger = sharper)
        """
        if rng is None:
            rng = np.random.default_rng()

        gumbels = rng.gumbel(loc=0, scale=1.0/beta, size=len(scores))
        perturbed = scores + gumbels
        return np.argsort(-perturbed)[:k]

    def _rs_init(self, score):
        return {"n": 1, "mean": float(score), "sumsq": score**2}

    def update_stats(self, state, new_score):
        state["n"] += 1
        state["mean"] = (state["mean"] * (state["n"] - 1) + new_score) / state["n"]
        state["sumsq"] += new_score ** 2
        return state
    
    # ---- Find the best prompt(s) by HIGHEST MEAN over the full history ----
    def _mean_or_score(self, c):
        return c.get("rs", {}).get("mean", c.get("score", float("-inf")))


    def finalize_std(self, state):
        import math
        n = state["n"]
        mean = state["mean"]
        var = (state["sumsq"] / n) - (mean ** 2)
        return math.sqrt(max(var, 0.0))

    def blockwise_uniform_generation(
            self,
            breadth_predictor: dspy.Predict,   # e.g., dspy.Predict(GenerateInstructionGivenAttempts, ...)
            attempts_topk: list[list[str]],    # K items; each item is the formatted list for one attempt
            B: int = 50,                       # tokens per block
            max_tokens: int = 2000,               # what are max tokens
        ):
            blocks = []
            num_blocks = max_tokens // B
            for _ in range(num_blocks):
                attempt = random.choice(attempts_topk)         # uniform sample
                pred = breadth_predictor(
                attempted_instructions=attempt,
                config={"max_tokens": B, "temperature": self.init_temperature}
                )
            # Extract raw text from the LM completion that fills your output fields:
            # Depending on your DSPy version, you can access:
            #   pred.completions[0].text  OR  pred.completions.text
            # But for GenerateInstructionGivenAttempts, you'll likely want the fields:
            blk = {
            "attempt": attempt,
            "proposed_instruction": pred.proposed_instruction,
            "proposed_prefix_for_output_field": pred.proposed_prefix_for_output_field,
            }
            blocks.append(blk)
            return blocks