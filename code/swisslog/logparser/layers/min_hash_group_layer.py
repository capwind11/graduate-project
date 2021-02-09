from datasketch import MinHash
from tqdm import tqdm

from layers.layer import Layer
from utils.words_utils import splitWords


class MinHashGroupLayer(Layer):
    def __init__(self, message_list: list, sim_value: float):
        self.message_list = message_list
        self.sim_value = sim_value

    def run(self) -> dict:
        sim_hash_dict = dict()
        min_hash_list = list()
        for value in tqdm(self.message_list):
            min_hash = MinHash()
            content_list = [i for i in splitWords(value)]
            for i in content_list:
                min_hash.update(i.encode('utf8'))

            min_dict = dict()
            min_dict['minhash'] = min_hash
            min_dict['message'] = value
            min_hash_list.append(min_dict)

        process_list = []
        for index, obj in enumerate(tqdm(min_hash_list)):
            if index in process_list:
                continue

            process_list.append(index)
            min_group_list = list()
            min_group_list.append(obj)
            sim_hash_dict[index] = min_group_list

            for sub_index, sub_obj in enumerate(min_hash_list):
                if index == sub_index or sub_index in process_list:
                    continue
                # define a sim_value to filter all unsimilar object
                if obj['minhash'].jaccard(sub_obj['minhash']) > self.sim_value:
                    if index in sim_hash_dict.keys():
                        sim_hash_dict[index].append(sub_obj)
                        process_list.append(sub_index)

        total_group = len(sim_hash_dict.keys())
        print("SimHash分组后数据总量:%s" % total_group)
        print("数据压缩比率为:%s" % (1 - total_group / len(self.message_list)))
        return sim_hash_dict
