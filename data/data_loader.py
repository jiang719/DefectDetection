import codecs
import torch

from data.dictionary import Dictionary

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class DataLoader:
    def __init__(self, filepath, dictionary):
        self.dictionary = dictionary
        self.data = []
        self.load_data(filepath)

    def load_data(self, filepath):
        fp = codecs.open(filepath, 'r', 'utf-8')
        instance, context_before, context_after, y = None, None, None, None
        for i, l in enumerate(fp.readlines()):
            if i % 4 == 0:
                instance = self.dictionary.index(l.strip().split())
            elif i % 4 == 1:
                context_before = self.dictionary.index(l.strip().split())
            elif i % 4 == 2:
                context_after = self.dictionary.index(l.strip().split())
            else:
                y = int(l.strip())
                self.data.append({
                    'instance': self.split_sentence(instance),
                    'context_before': self.split_sentence(context_before),
                    'context_after': self.split_sentence(context_after),
                    'label': y
                })
            if (i + 1) % 500000 == 0:
                print(i + 1)

    def split_sentence(self, line):
        result = []
        sent = []
        for idx in line:
            sent.append(idx)
            if self.dictionary[idx] in ['{', ';']:
                result.append(sent)
                sent = []
        if sent:
            result.append(sent)
        return result

    def get_transfo_input(self, start, end):
        inputs = []
        tags = []
        labels = []
        max_length = 0
        for d in self.data[start: end]:
            l = sum([len(s) for s in d['context_before'] + d['instance'] + d['context_after']])
            if l > max_length:
                max_length = l
        max_length = min(1000, max_length)

        for d in self.data[start: end]:
            input = []
            tag = []
            for s in d['context_before']:
                if not s:
                    continue
                input += s
                tag += [1]*len(s)
            for s in d['instance']:
                if not s:
                    continue
                input += s
                tag += [2]*len(s)
            for s in d['context_after']:
                if not s:
                    continue
                input += s
                tag += [1]*len(s)
            if not input:
                continue
            input += [self.dictionary.eos()]
            tag += [1]
            input = input[: max_length]
            tag = tag[: max_length]
            input += [0]*(max_length - len(input))
            tag += [0]*(max_length - len(tag))
            inputs.append(input)
            tags.append(tag)
            labels.append(d['label'])
        return {
            'inputs': torch.LongTensor(inputs).to(device),
            'tags': torch.LongTensor(tags).to(device),
            'labels': torch.LongTensor(labels).to(device)
        }

    def get_hatt_input(self, start, end):
        max_sent_num = max([len(d['instance']) + len(d['context_before']) + len(d['context_after'])
                            for d in self.data[start: end]])
        max_word_num = max([len(s) for d in self.data[start: end]
                            for s in d['instance'] + d['context_before'] + d['context_after']])
        max_sent_num = min(max_sent_num, 50)
        max_word_num = min(max_word_num, 80)
        inputs = []
        tags = []
        labels = []
        for d in self.data[start: end]:
            input = []
            tag = []
            for s in d['context_before']:
                if not s:
                    continue
                s += [self.dictionary.eos()]
                s = s[: max_word_num]
                s += [0] * (max_word_num - len(s))
                input.append(s)
                tag.append(1)
            for s in d['instance']:
                if not s:
                    continue
                s += [self.dictionary.eos()]
                s = s[: max_word_num]
                s += [0] * (max_word_num - len(s))
                input.append(s)
                tag.append(2)
            for s in d['context_after']:
                if not s:
                    continue
                s += [self.dictionary.eos()]
                s = s[: max_word_num]
                s += [0] * (max_word_num - len(s))
                input.append(s)
                tag.append(1)
            if not input:
                continue
            input = input[: max_sent_num]
            tag = tag[: max_sent_num]
            input += [[0] * max_word_num for _ in range(max_sent_num - len(input))]
            tag += [0] * (max_sent_num - len(tag))
            inputs.append(input)
            tags.append(tag)
            labels.append(d['label'])
        return {
            'inputs': torch.LongTensor(inputs).to(device),
            'tags': torch.LongTensor(tags).to(device),
            'labels': torch.LongTensor(labels).to(device)
        }

    def get_max_length(self):
        sent_nums = [len(d['instance']) + len(d['context_before']) + len(d['context_after'])
                     for d in self.data]
        word_nums = [len(s) for d in self.data
                     for s in d['instance'] + d['context_before'] + d['context_after']]
        return sent_nums, word_nums


if __name__ == "__main__":
    dictionary = Dictionary('D:/data/defect-detection/vocab.txt')
    print('dictionary loaded')
    train_loader = DataLoader('D:/data/defect-detection/bpe_train.txt', dictionary)
    sent_nums, word_nums = train_loader.get_max_length()
