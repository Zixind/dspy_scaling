import random, re
from datasets import load_dataset
import dspy

LETTER_MAP = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


from dspy.datasets.dataset import Dataset


def format_arc_prompt(question: str, choices: list[str]) -> str:
    lines = [f"Question: {question.strip()}", "Choices:"]
    for i, c in enumerate(choices):
        lines.append(f"{LETTER_MAP[i]}. {c.strip()}")
    lines.append("Your answer should be a single letter. The last line must be: 'Answer: X'")
    return "\n".join(lines)

def normalize_to_letter(x: str) -> str:
    """Parse model string to a single letter A–E if possible."""
    s = x.strip().upper()
    if "ANSWER:" in s:
        s = s.split("ANSWER:", 1)[1].strip()
    # keep only the last A–E/1–5 symbol that appears
    m = re.findall(r"[A-E1-5]", s)
    if not m:
        return ""
    c = m[-1]
    digits2letter = {"1":"A","2":"B","3":"C","4":"D","5":"E"}
    return digits2letter.get(c, c)


class ARC_DSPy:
    """
    Usage:
        data = ARC_DSPy(subset="ARC-Challenge", seed=0, max_train=2000, max_dev=500, max_test=500)
        data.train / data.dev / data.test are lists of dspy.Example(question=..., answer=...)
        data.metric(example, pred) -> 0/1
    """
    def __init__(self, subset="ARC-Challenge", cache_dir=None,
                 seed=0, max_train=None, max_dev=None, max_test=None):
        assert subset in ["ARC-Challenge", "ARC-Easy"]
        ds = load_dataset("allenai/ai2_arc", subset, cache_dir=cache_dir)

        def build(split):
            out = []
            for ex in ds[split]:
                q = ex["question"]
                texts = ex["choices"]["text"]           # list of options
                prompt = format_arc_prompt(q, texts)
                gold = str(ex["answerKey"]).strip().upper()  # typically A–E
                out.append(dspy.Example(question=prompt, answer=gold).with_inputs("question"))
            random.Random(seed).shuffle(out)
            return out

        train = build("train")
        dev   = build("validation")
        test  = build("test")

        if max_train: train = train[:max_train]
        if max_dev:   dev   = dev[:max_dev]
        if max_test:  test  = test[:max_test]

        self.train, self.dev, self.test = train, dev, test

    def metric(self, example, pred, trace=None):
        """Return 1 if predicted letter matches gold letter (A–E); else 0."""
        pred_letter = normalize_to_letter(getattr(pred, "answer", ""))
        gold_letter = normalize_to_letter(example.answer)
        return int(pred_letter == gold_letter)

class ARCAnswer(dspy.Signature):
    """Answer a multiple-choice science question with a single letter."""
    question: dspy.InputField()
    answer: dspy.OutputField(desc="Reply only with `Answer: X` where X is A–E.")


def quick_eval_on_dev(num=50):
    data = ARC_DSPy(subset="ARC-Challenge", seed=0, max_dev=num)
    predictor = dspy.Predict(ARCAnswer)

    correct = 0
    for ex in data.dev:
        pred = predictor(question=ex.question)
        correct += data.metric(ex, pred)
    print(f"Dev accuracy on {num} examples: {correct/num:.3f}")
