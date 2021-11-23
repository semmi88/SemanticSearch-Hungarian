from transformers import AutoTokenizer, AutoModel
import torch
from sentence_transformers import util

def load_raw_sentences(filename):
    with open(filename) as f:
        return f.readlines()

#Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask

def findTopKMostSimilar(query_embedding, embeddings, k):
    cosine_scores = util.pytorch_cos_sim(query_embedding, embeddings)
    cosine_scores_list = cosine_scores.squeeze().tolist()
    pairs = []
    for idx,score in enumerate(cosine_scores_list):
        pairs.append({'index': idx, 'score': score})
    pairs = sorted(pairs, key=lambda x: x['score'], reverse=True)
    return pairs[0:k]

def calculateEmbeddings(sentences,tokenizer,model):
    tokenized_sentences = tokenizer(sentences, padding=True, truncation=True, max_length=128, return_tensors='pt')
    with torch.no_grad():
        model_output = model(**tokenized_sentences)
    sentence_embeddings = mean_pooling(model_output, tokenized_sentences['attention_mask'])
    return sentence_embeddings

multilingual_checkpoint = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
tokenizer = AutoTokenizer.from_pretrained(multilingual_checkpoint)
model = AutoModel.from_pretrained(multilingual_checkpoint)

raw_text_file = 'data/processed/shortened_abstracts_hu_2021_09_01.txt'
embeddings_file = 'data/processed/shortened_abstracts_hu_2021_09_01_embedded.pt'

all_sentences = load_raw_sentences(raw_text_file)
all_embeddings = torch.load(embeddings_file,map_location=torch.device('cpu') )

query = ''
while query != 'exit':
    query = input("Enter your query: ")
    query_embedding = calculateEmbeddings([query],tokenizer,model)
    top_pairs = findTopKMostSimilar(query_embedding, all_embeddings, 5)
    for pair in top_pairs:
        i = pair['index']
        score = pair['score']
        print("{} \t\t Score: {:.4f}".format(all_sentences[i], score))
