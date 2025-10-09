import logging
from collections import defaultdict
import copy
import dspy
from dspy.evaluate.evaluate import Evaluate
from dspy.signatures import Signature
from dspy.teleprompt.teleprompt import Teleprompter

logger = logging.getLogger(__name__)

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
                k = min(len(best_list), self.breadth)
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
    
    def compile_gumbel_top_k(self, student, *, trainset, minibatch_val_size=None, rng_seed = 0, eval_kwargs):
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
        module = student  # work in-place; avoid deepcopy
        
        total_calls = 0
        # Materialize once so we can sample indices reproducibly
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
        
        def subsample(data, k, seed=42):
            rng = random.Random(seed)
            idxs = list(range(len(data)))
            rng.shuffle(idxs)
            chosen = idxs[:min(k, len(data))]
            return [data[i] for i in chosen]

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
            if minibatch_val_size is None or minibatch_val_size >= len(trainset_list):
                dev_batch = trainset_list
            else:
                rng_draw = np.random.default_rng(1)
                dev_batch = subsample(trainset_list, minibatch_val_size, seed = rng_draw)
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
                    result = evaluate(module, devset=dev_batch, **eval_kwargs) #evaluate on dev_batch
                    score = result.score
                    if self.prompt_model:
                        logger.debug(f"prompt_model.inspect_history(n=1) {self.prompt_model.inspect_history(n=1)}")
                    total_calls += 1

                    # Keep the best score per (instruction, prefix) need to modify
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
                k = min(len(best_list), self.breadth)
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