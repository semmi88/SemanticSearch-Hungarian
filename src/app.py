import streamlit as st
from transformers import AutoTokenizer, AutoModel
import transformers
import torch
from sentence_transformers import util

# explicit no operation hash functions defined, because raw sentences, embedding, model and tokenizer are not going to change


@st.cache(hash_funcs={list: lambda _: None})
def load_raw_sentences(filename):
    with open(filename) as f:
        return f.readlines()

@st.cache(hash_funcs={torch.Tensor: lambda _: None})
def load_embeddings(filename):
    with open(filename) as f:
        return torch.load(filename,map_location=torch.device('cpu') )


#Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask

def findTopKMostSimilar(query_embedding, embeddings, all_sentences, k):
    cosine_scores = util.pytorch_cos_sim(query_embedding, embeddings)
    cosine_scores_list = cosine_scores.squeeze().tolist()
    pairs = []
    for idx,score in enumerate(cosine_scores_list):
        if idx < len(all_sentences):
            pairs.append({'score': '{:.4f}'.format(score), 'text': all_sentences[idx]})
    pairs = sorted(pairs, key=lambda x: x['score'], reverse=True)
    return pairs[0:k]

def calculateEmbeddings(sentences,tokenizer,model):
    tokenized_sentences = tokenizer(sentences, padding=True, truncation=True, max_length=128, return_tensors='pt')
    with torch.no_grad():
        model_output = model(**tokenized_sentences)
    sentence_embeddings = mean_pooling(model_output, tokenized_sentences['attention_mask'])
    return sentence_embeddings

# explicit no operation hash function, because model and tokenizer are not going to change
@st.cache(hash_funcs={transformers.models.bert.tokenization_bert_fast.BertTokenizerFast: lambda _: None, transformers.models.bert.modeling_bert.BertModel: lambda _: None})
def load_model_and_tokenizer():
    multilingual_checkpoint = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
    tokenizer = AutoTokenizer.from_pretrained(multilingual_checkpoint)
    model = AutoModel.from_pretrained(multilingual_checkpoint)
    print(type(tokenizer))
    print(type(model))
    return model, tokenizer


model,tokenizer = load_model_and_tokenizer();
raw_text_file = 'data/preprocessed/shortened_abstracts_hu_2021_09_01.txt'
all_sentences = load_raw_sentences(raw_text_file)

embeddings_file = 'data/preprocessed/shortened_abstracts_hu_2021_09_01_embedded.pt'
all_embeddings = load_embeddings(embeddings_file)


st.header('Wikipedia absztrakt kereső')
st.subheader('Search Wikipedia abstracts in Hungarian!')

st.caption('[HU] Adjon meg egy tetszőleges kifejezést és a rendszer visszaadja az 5 hozzá legjobban hasonlító Wikipedia absztraktot')
st.caption('[EN] Input some search term and see the top-5 most similar wikipedia abstracts')


text_area_input_query = st.text_area('[HU] Beviteli mező - [EN] Query input',value='Mi Japán fővárosa?')

if text_area_input_query:
    query_embedding = calculateEmbeddings([text_area_input_query],tokenizer,model)
    top_pairs = findTopKMostSimilar(query_embedding, all_embeddings, all_sentences, 5)
    st.json(top_pairs)
