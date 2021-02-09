import difflib

from tqdm import tqdm

from layers.layer import Layer


class LevenLayer(Layer):

    def __init__(self, sim_hash_dict: dict, leven_threshold: float = 0.3, reduce_field: str = 'message'):
        self.sim_hash_dict = sim_hash_dict
        self.leven_threshold = leven_threshold
        self.reduce_field = reduce_field

    def difflib_leven(self, str1, str2):
        leven_cost = 0
        s = difflib.SequenceMatcher(None, str1, str2)
        for tag, i1, i2, j1, j2 in s.get_opcodes():
            if tag == 'replace':
                leven_cost += max(i2 - i1, j2 - j1)
            elif tag == 'insert':
                leven_cost += (j2 - j1)
            elif tag == 'delete':
                leven_cost += (i2 - i1)
        return leven_cost

    def run(self) -> dict:
        processed_key = []
        prepare_delete_key = []
        for key in tqdm(self.sim_hash_dict.keys()):
            if key in processed_key:
                continue

            # 这里有点问题：为什么是直接选第一个来比较他们的相似度，应该随机选取，准确度会更高一点。

            source_message = self.sim_hash_dict[key][0][self.reduce_field]
            processed_key.append(key)
            for sub_key in self.sim_hash_dict.keys():
                if sub_key == key or sub_key in processed_key:
                    continue
                target_message = self.sim_hash_dict[sub_key][0][self.reduce_field]
                if self.difflib_leven(source_message, target_message)/(len(source_message)+len(target_message))<self.leven_threshold:
                    processed_key.append(sub_key)
                    self.sim_hash_dict[key].extend(self.sim_hash_dict[sub_key])
                    prepare_delete_key.append(sub_key)

        for value in prepare_delete_key:
            del self.sim_hash_dict[value]
        print('After Leven distance reduce, total: %s bin(s)' % len(self.sim_hash_dict.keys()))
        return self.sim_hash_dict
