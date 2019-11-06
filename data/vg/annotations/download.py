import os

links = ['https://visualgenome.org/static/data/dataset/image_data.json.zip', 'https://visualgenome.org/static/data/dataset/objects.json.zip', 'https://visualgenome.org/static/data/dataset/relationships.json.zip', 'https://visualgenome.org/static/data/dataset/object_alias.txt', 'https://visualgenome.org/static/data/dataset/relationship_alias.txt', 'https://visualgenome.org/static/data/dataset/object_synsets.json.zip', 'https://visualgenome.org/static/data/dataset/attribute_synsets.json.zip', 'https://visualgenome.org/static/data/dataset/relationship_synsets.json.zip', 'https://visualgenome.org/static/data/dataset/relationship_synsets.json.zip', 'https://visualgenome.org/static/data/dataset/region_descriptions.json.zip', 'https://visualgenome.org/static/data/dataset/question_answers.json.zip', 'https://visualgenome.org/static/data/dataset/attributes.json.zip', 'https://visualgenome.org/static/data/dataset/region_graphs.json.zip', 'https://visualgenome.org/static/data/dataset/scene_graphs.json.zip']

for l in links:
    if not os.path.isfile(os.path.basename(l)):
        os.system('wget %s' %l)

for l in links:
    if l.endswith('.zip'):
        os.system('unzip %s' %(os.path.basename(l)))