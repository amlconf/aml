# download movies review data from https://www.eraserbenchmark.com

import json
from eraser_datasets.eraser_utils import load_flattened_documents

movies_url = "./raw_data"
documents = load_flattened_documents(movies_url, docids = None)


def read_annotations(json_file):
    anns = [json.loads(line) for line in open(json_file)]
    for a in anns:
        doc_id = a['annotation_id']
        a['document'] = " ".join(documents[doc_id])
        a['label'] = a['classification']
        del a['classification']
        a['rationale'] = []
        for evgroup in a['evidences']:
            for ev in evgroup:
                assert ev['docid'] == doc_id
                a['rationale'].append((ev['start_token'], ev['end_token']))
        del a['evidences']
        del a['query_type']
        del a['query']

    return anns


import os

os.makedirs('../data/', exist_ok = True)

for key in ['test']:
    ann = read_annotations(f'{movies_url}/' + key + '.jsonl')
    with open('data/' + key + '.jsonl', 'w') as f:
        f.write('\n'.join([json.dumps(line) for line in ann]))
