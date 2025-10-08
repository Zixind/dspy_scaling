import dspy 
gpt4o_mini = dspy.LM('openai/gpt-4o-mini-2024-07-18', max_tokens=500)
gpt4o = dspy.LM('openai/gpt-4o', max_tokens=500) #improve the prompt
dspy.configure(lm=gpt4o_mini)  # we'll use gpt-4o-mini as the default LM, unless otherwise specified

from dspy.datasets import MATH

dataset = MATH(subset='algebra')
print(len(dataset.train), len(dataset.dev))

example = dataset.train[0]
print("Question:", example.question)
print("Answer:", example.answer)

module = dspy.ChainOfThought("question -> answer")
module(question=example.question)

THREADS = 24
kwargs = dict(num_threads=THREADS, display_progress=True, display_table=5)
evaluate = dspy.Evaluate(devset=dataset.dev, metric=dataset.metric, **kwargs)

print(evaluate(module))

kwargs = dict(num_threads=THREADS, prompt_model = gpt4o)
optimizer = dspy.COPRO(metric=dataset.metric, student=gpt4o_mini, **kwargs)

# kwargs = dict(max_bootstrapped_demos=4, max_labeled_demos=4)
optimized_module = optimizer.compile(module, trainset=dataset.train, **kwargs)

print(evaluate(optimized_module))
