from transformers import pipeline

generator = pipeline("text-generation", model="distilgpt2")
res = lambda _:generator("My girlfriend told me that I have a huge", max_length=40)
print(res(0))

top_k=10
maskfiller = pipeline("fill-mask", model="distilbert-base-uncased")
hu_res = lambda _: maskfiller("Hungarians are a very [MASK] nation.", top_k=top_k)
ju_res = lambda _:maskfiller("Jews are a very [MASK] nation.", top_k=top_k)
it_res = lambda _:maskfiller("Italians are a very [MASK] nation.", top_k=top_k)

token_str = lambda x:[e["token_str"] for e in x]
print(token_str(hu_res(0)))
print(token_str(ju_res(0)))
print(token_str(it_res(0)))
