import random
import re


class MATH:
    def __init__(self, subset, sample_size = None):
        from datasets import load_dataset

        import dspy

        ds = load_dataset("DigitalLearningGmbH/MATH-lighteval", subset)

        # NOTE: Defaults to sub-splitting MATH's 'test' split into train/dev/test, presuming that current
        # LMs are trained on MATH's train. Makes no difference for gpt-4o-mini, but might for other models.

        dataset = [
            dspy.Example(
                question=example["problem"], reasoning=example["solution"], answer=extract_answer(example["solution"])
            ).with_inputs("question")
            for example in ds["test"]
        ]
        if sample_size is not None:
            assert type(sample_size) == int, "Sample size must be an integer."
            if sample_size < 350:
                trainset = dataset[:sample_size]
                sample_size_init = min(350, len(dataset) // 3)
                devset = dataset[sample_size_init: 2 * sample_size_init]  #originally 350 dev set
                testset = dataset[2 * sample_size_init:]
                
            else:
                trainset = dataset[:sample_size]
                devset = dataset[sample_size: sample_size + 350]  #originally 350 dev set
                testset = dataset[sample_size + 350:] #manually add 350
            self.train, self.dev, self.test = trainset, devset, testset
        
        else:
            size = min(350, len(dataset) // 3)
            random.Random(0).shuffle(dataset)
            self.train, self.dev, self.test = dataset[:size], dataset[size : 2 * size], dataset[2 * size :]

    def metric(self, example, pred, trace=None):
        try:
            import math_equivalence
        except ImportError:
            raise ImportError("MATH's metric requires `pip install git+https://github.com/hendrycks/math.git`")

        return math_equivalence.is_equiv(example.answer, pred.answer)


def extract_answer(s):
    start = s.find("\\boxed{")
    if start == -1:
        return None

    idx = start + len("\\boxed{")
    brace_level = 1

    answer = ""
    while idx < len(s) and brace_level > 0:
        c = s[idx]
        if c == "{":
            brace_level += 1
        elif c == "}":
            brace_level -= 1
            if brace_level == 0:
                break
        answer += c
        idx += 1

    answer = re.sub(r"\\text\{[^}]*\}", "", answer)
    answer = re.sub(r"\\!", "", answer)
    return answer.strip()


"""
NOTE: MATH's official math_equivalence.is_equiv does not seem to have perfect recall.
Consider its behavior on reference values like `left[\frac{1}{2}, \frac{4}{3}\right]`.
"""
