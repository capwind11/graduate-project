'''
日志条目聚类
'''

from tqdm import tqdm
from layers.layer import Layer
import pickle
from tqdm import tqdm
import wordninja

# 是否存在数字
def hasDigit(inputString):
    return any(char.isdigit() for char in inputString)

# 比较句子集的相似度
def tolerant(source_dwords, target_dwords):
    rmt = target_dwords.copy()
    if len(source_dwords)<4: return False
    if len(target_dwords)<4: return False
    rms = set()

    for word in source_dwords:
        if word in target_dwords:
            # 在目标句子里就删去
            rmt.remove(word)
        else:
            # 不在就
            rms.add(word)
    # 得到两个集合中不同的元素
    return len(rmt)<=1 and len(rms) <=1

# 字典聚类层
class DictGroupLayer(Layer):
    def __init__(self, log_messages, dictionary_file=None):
        self.log_messages = log_messages
        self.dictionary = None
        # 载入字典
        if dictionary_file:
            with open(dictionary_file, 'rb') as f:
                self.dictionary = pickle.load(f)

    # 字典化，建议直接用后续bert的字典
    def dictionaried(self):
        result = list()
        for value in tqdm(self.log_messages, desc='dictionaried'):
            # dictionary_words = set()
            dictionary_list = list()
            for word in value['Content']:
                if hasDigit(word):
                    continue
                word = word.strip('.:*')
                if word in self.dictionary:
                    # dictionary_words.add(word)
                    dictionary_list.append(word)
                elif all(char.isalpha() for char in word):
                    # 对于可切分的复合词，拆分出来
                    splitted_words = wordninja.split(word)
                    for sword in splitted_words:
                        if len(sword) <= 2: continue
                        # dictionary_words.add(sword)
                        dictionary_list.append(sword)
            # 结果是由content、dictionary_list、lineId组成的list
            result_dict = dict(message=value['Content'], dwords=dictionary_list, LineId=value['LineId'])
            result.append(result_dict)
        return result

    def run(self) -> dict:
        dicted_list = self.dictionaried()
        dwords_group = dict()
        # 根据字典聚类
        for element in tqdm(dicted_list, desc='group by dictionary words'):
            # frozen_dwords = frozenset(element['dwords'])
            # 先排个序
            frozen_dwords = tuple(sorted(element['dwords']))
            if frozen_dwords not in dwords_group:
                dwords_group[frozen_dwords] = []
            # 直接根据set来聚类
            dwords_group[frozen_dwords].append(element)
        tot = 0;
        result_group = dict()
        diffrent_length = 0
        for key in dwords_group.keys():
            if len(key) == 0:
                # 空tuple时
                for entry in dwords_group[key]:
                    result_group[tot] = [entry]
                    tot += 1
                continue
            result_group[tot] = dwords_group[key]
            len_set = set()
            for element in result_group[tot]:
                len_set.add(len(element['message']))
            diffrent_length += len(len_set)
            tot += 1
        # print('if split by length, total: {}'.format(diffrent_length))
        print('After Dictionary Group, total: {} bin(s)'.format(len(result_group.keys())))
        return result_group
