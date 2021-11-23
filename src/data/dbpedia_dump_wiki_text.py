from rdflib import Graph

# Downloaded from https://databus.dbpedia.org/dbpedia/text/short-abstracts
raw_data_path = 'data/raw/short-abstracts_lang=hu.ttl'
preprocessed_data_path = 'data/preprocessed/shortened_abstracts_hu_2021_09_01.txt'

g = Graph()
g.parse(raw_data_path, format='turtle')

i = 0
objects = []
with open(preprocessed_data_path, 'w') as f:
    print(len(g))
    for subject, predicate, object in g:
        objects.append(object.replace(' +/-','').replace('\n',' '))
        objects.append('\n')
        i += 1
    f.writelines(objects)