import os
_zixin_api = "OPENAI_KEY_REDACTED"
os.environ["OPENAI_API_KEY"] = _zixin_api 
#pip install git+https://github.com/hendrycks/math.git
import dspy 
import os, json, time, pathlib, re
from dspy.datasets import MATH

SAMPLE_SIZE = 100
BREADTH = 2
DEPTH = 40


gpt4o_mini = dspy.LM('openai/gpt-4o-mini-2024-07-18', max_tokens=2000) #inference
gpt4o = dspy.LM('openai/gpt-4o', max_tokens=2000) #improve the prompt
dspy.configure(lm=gpt4o_mini)  # we'll use gpt-4o-mini as the default LM, unless otherwise specified


dataset = MATH(subset='algebra', sample_size = SAMPLE_SIZE)
print(len(dataset.train), len(dataset.dev))

example = dataset.train[0]
print("Question:", example.question)
print("Answer:", example.answer)

module = dspy.ChainOfThought("question -> answer")
module(question=example.question)

THREADS = 24
kwargs = dict(num_threads=THREADS, display_progress=True, display_table=5)
evaluate = dspy.Evaluate(devset=dataset.dev, metric=dataset.metric, **kwargs)

# print(evaluate(module)) #75.1%

kwargs = dict(num_threads=THREADS) #, **kwargs
optimizer = dspy.COPRO(metric=dataset.metric, student=gpt4o_mini, prompt_model = gpt4o, breadth = BREADTH, depth = DEPTH, init_temperature = 0)

# kwargs = dict(max_bootstrapped_demos=4, max_labeled_demos=4)
optimized_module = optimizer.compile(module.deepcopy(), trainset=dataset.train, eval_kwargs = kwargs)

print(evaluate(optimized_module))



runs_dir = pathlib.Path("runs"); runs_dir.mkdir(parents=True, exist_ok=True)
run_id = time.strftime("%Y%m%d_%H%M%S")
jsonl_path = runs_dir / f"math_algebra_{run_id}_b{BREADTH}_d{DEPTH}_s{SAMPLE_SIZE}.jsonl"
summary_path = runs_dir / f"summary_{run_id}_b{BREADTH}_d{DEPTH}_s{SAMPLE_SIZE}.json"

def extract_answer_line(s: str) -> str:
    s = (s or "").strip()
    m = re.search(r'Answer:\s*(.+)', s, flags=re.I)
    return (m.group(1) if m else s).strip()


# --- log dev predictions
n = 0
n_correct = 0
with open(jsonl_path, "w", encoding="utf-8") as f:
    for i, ex in enumerate(dataset.dev):
        # Call your (tuned) module
        pred = optimized_module(question=ex.question)
        pred_text = getattr(pred, "answer", "")
        pred_final = extract_answer_line(pred_text)

        # DSPy metric expects the full pred object; we also store bool
        correct = int(dataset.metric(ex, pred))

        row = {
            "i": i,
            "split": "dev",
            "question": ex.question,
            "gold": ex.answer,
            "pred_raw": pred_text,
            "pred": pred_final,
            "correct": correct,
        }
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
        n += 1
        n_correct += correct

# --- write a tiny summary JSON
summary = {
    "run_id": run_id,
    "split": "dev",
    "num_examples": n,
    "num_correct": n_correct,
    "accuracy": (n_correct / n) if n else None,
    "model_default": "openai/gpt-4o-mini-2024-07-18",
    "module": "ChainOfThought(question -> answer)",
}
with open(summary_path, "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)
print(f"Wrote {n} rows to {jsonl_path}")
print(f"Summary → {summary_path}  (acc={summary['accuracy']:.3f})")


try:
    hist = dspy.inspect_history(1)   # list of dicts
    with open(runs_dir / f"history_{run_id}_b{BREADTH}_d{DEPTH}_s{SAMPLE_SIZE}.json", "w", encoding="utf-8") as f:
        json.dump(hist, f, indent=2)
except Exception as e:
    print("inspect_history() not available here:", e)