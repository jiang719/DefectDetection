import pickle
import codecs

data_path = 'D:/data/defect-detection/'


def pickle_to_text(dataset='test'):
    data = pickle.load(open(data_path + 'tokenized_' + dataset + '.pickle', 'rb'))
    print(dataset + ' loaded, ' + str(len(data)))
    wp = codecs.open(data_path + dataset + '.txt', 'w', 'utf-8')
    cnt = 0
    for _, row in data.iterrows():
        wp.write(' '.join(row.instance) + '\n')
        wp.write(' '.join(row.context_before) + '\n')
        wp.write(' '.join(row.context_after) + '\n')
        wp.write(str(row.is_buggy) + '\n')
        cnt += 1
        #if cnt % 1e4 == 0:
        #    print(cnt)
        if cnt == 1e5:
            break
    wp.close()


def get_vocab():
    vocab = {}
    fp = codecs.open(data_path + 'train.txt', 'r', 'utf-8')
    for l in fp.readlines():
        l = l.strip().split()
        for w in l:
            if w not in vocab:
                vocab[w] = 0
            vocab[w] += 1
    fp.close()
    wp = codecs.open(data_path + 'vocab.txt', 'w', 'utf-8')
    vocab = sorted(vocab.items(), key=lambda e: e[1], reverse=True)
    for (v, cnt) in vocab:
        wp.write(v + ' ' + str(cnt) + '\n')
    wp.close()


#for dataset in ('train', ):
#    pickle_to_text(dataset)

get_vocab()
