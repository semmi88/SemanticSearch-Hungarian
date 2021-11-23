from datasets import load_dataset

faq_hu = load_dataset("clips/mqa", scope="faq", language="hu")
cqa_hu = load_dataset("clips/mqa", scope="cqa", language="hu")

print(faq_hu)
print(cqa_hu)
print(faq_hu['train'][:5])
print(cqa_hu['train'][:5])