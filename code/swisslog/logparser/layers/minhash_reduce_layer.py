import farmhash

from datasketch import MinHash
from tqdm import tqdm

from layers.layer import Layer
from utils.words_utils import splitWords



class MinHashReduceLayer(Layer):
    """
    SimHash聚类，合并分组之间相似的数据
    """

    def _hash_func(self, d):
        return farmhash.hash64(d)

    def __init__(self, sim_hash_dict: dict(), sim_value: float = 0.9):
        self.sim_hash_dict = sim_hash_dict
        self.sim_value = sim_value


    # 这个算法是错误的
    def run(self) -> dict():
        processed_key = list()
        prepare_delete_key = list()
        # print(len(self.sim_hash_dict.keys()))
        for key in self.sim_hash_dict.keys():
            source_message = self.sim_hash_dict[key][0]['message']
            source_min_hash = MinHash(hashfunc=self._hash_func)
            content_list = source_message
            # content_list = [i for i in splitWords(source_message)]
            for i in content_list:
                source_min_hash.update(i)
            self.sim_hash_dict[key][0]['minhash'] = source_min_hash

        for key in self.sim_hash_dict.keys():

            if key in processed_key:
                continue
            # 确定是最小的放在前面吗？【0】表示该集合中最小的hash
            source_min_hash = self.sim_hash_dict[key][0]['minhash']
            processed_key.append(key)


            for sub_key in self.sim_hash_dict.keys():
                if sub_key <= key or sub_key in processed_key:
                    continue
                
                # 找到其他集合中的最小hash值。
                target_min_hash = self.sim_hash_dict[sub_key][0]['minhash']
                if source_min_hash.jaccard(target_min_hash) > self.sim_value:
                    processed_key.append(sub_key)
                    self.sim_hash_dict[key].extend(self.sim_hash_dict[sub_key])
                    prepare_delete_key.append(sub_key)
        for value in prepare_delete_key:
            del self.sim_hash_dict[value]
        print('After Minhash Reduce, total: %s bin(s)' % len(self.sim_hash_dict.keys()))
        return self.sim_hash_dict
