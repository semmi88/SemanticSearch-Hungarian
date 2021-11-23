from transformers import AutoTokenizer, AutoModel
import torch
import pickle
from sentence_transformers import util
from datetime import datetime

#Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask


dt = datetime.now()
datetime_formatted = dt.strftime('%Y-%m-%d_%H:%M:%S')
batch_size = 1000
output_embeddings_file = f'data/preprocessed/embeddings_{batch_size}_batches_at_{datetime_formatted}.pkl'
def saveToDisc(embeddings):
    with open(output_embeddings_file, "ab") as f:
        pickle.dump(embeddings, f, protocol=pickle.HIGHEST_PROTOCOL)


def saveToDisc(sentences, embeddings, filename='embeddings.pkl'):
    with open(filename, "ab") as f:
        pickle.dump({'sentences': sentences, 'embeddings': embeddings}, f, protocol=pickle.HIGHEST_PROTOCOL)

def saveToDiscRaw(embeddings, filename='embeddings.pkl'):
    with open(filename, "ab") as f:
        pickle.dump(embeddings, f, protocol=pickle.HIGHEST_PROTOCOL)
        #for emb in embeddings:
        #    torch.save(emb,f)

def loadFromDiskRaw(filename='embeddings.pkl'):
    with open(filename, "rb") as f:
        stored_data = pickle.load(f)
    return stored_data

def loadFromDisk(filename='embeddings.pkl'):
    with open(filename, "rb") as f:
        stored_data = pickle.load(f)
        stored_sentences = stored_data['sentences']
        stored_embeddings = stored_data['embeddings']
    return stored_sentences, stored_embeddings

def findTopKMostSimilarPairs(embeddings, k):
    cosine_scores = util.pytorch_cos_sim(embeddings, embeddings)
    pairs = []
    for i in range(len(cosine_scores)-1):
        for j in range(i+1, len(cosine_scores)):
            pairs.append({'index': [i, j], 'score': cosine_scores[i][j]})

    pairs = sorted(pairs, key=lambda x: x['score'], reverse=True)
    return pairs[0:k]

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

raw_text_file = 'data/preprocessed/shortened_abstracts_hu_2021_09_01.txt'


concated_sentence_embeddings = None
all_sentences = []

print(datetime.now())
batch_size = 5
line = 'init'
total_read = 0
total_read_limit = 120
skip_index = 100
with open(raw_text_file) as f:
    while line and total_read < total_read_limit:
        count = 0
        sentence_batch = []
        while line and count < batch_size:
            line = f.readline()
            sentence_batch.append(line)
            count += 1
        
        all_sentences.extend(sentence_batch)
        
        if total_read >= skip_index:
            sentence_embeddings = calculateEmbeddings(sentence_batch,tokenizer,model)
            if concated_sentence_embeddings == None:
                concated_sentence_embeddings = sentence_embeddings
            else:
                concated_sentence_embeddings = torch.cat([concated_sentence_embeddings, sentence_embeddings], dim=0)
            print(concated_sentence_embeddings.size())
        #saveToDiscRaw(sentence_embeddings)
        
        total_read += count
        if total_read%5==0:
            print(f'total_read:{total_read}')
print(datetime.now())


query_embedding = calculateEmbeddings(['Melyik a legnépesebb város a világon?'],tokenizer,model)
top_pairs = findTopKMostSimilar(query_embedding, concated_sentence_embeddings, 5)

for pair in top_pairs:
    i = pair['index']
    score = pair['score']
    print("{} \t\t Score: {:.4f}".format(all_sentences[skip_index+i], score))
'''
query = ''
while query != 'exit':
    query = input("Enter your query: ")
    query_embedding = calculateEmbeddings([query],tokenizer,model)


'''