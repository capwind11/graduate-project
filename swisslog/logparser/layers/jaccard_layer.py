import jieba.posseg as pseg
from tqdm import tqdm

from layers.layer import Layer


class JaccardLayer(Layer):

    def splitWords(self, str_a):
        wordsa = pseg.cut(str_a)
        cuta = ""
        seta = set()
        for key in wordsa:
            cuta += key.word + " "
            seta.add(key.word)

        return [cuta, seta]

    def JaccardSim(self, str_a, str_b):
        seta = self.splitWords(str_a)[1]
        setb = self.splitWords(str_b)[1]

        sa_sb = 1.0 * len(seta & setb) / len(seta | setb)

        return sa_sb

    def __init__(self, sim_hash_dict: dict, sim_value: float = 0.6, reduce_field: str = 'message'):
        self.sim_hash_dict = sim_hash_dict
        self.sim_value = sim_value
        self.reduce_field = reduce_field

    def run(self) -> dict:
        processed_key = []
        prepare_delete_key = []
        for key in tqdm(self.sim_hash_dict.keys()):
            if key in processed_key:
                continue

            source_message = self.sim_hash_dict[key][0][self.reduce_field]
            processed_key.append(key)

            for sub_key in self.sim_hash_dict.keys():
                if sub_key == key or sub_key in processed_key or sub_key in processed_key:
                    continue
                target_message = self.sim_hash_dict[sub_key][0][self.reduce_field]

                if self.JaccardSim(source_message, target_message) > self.sim_value:
                    processed_key.append(sub_key)
                    self.sim_hash_dict[key].extend(self.sim_hash_dict[sub_key])
                    prepare_delete_key.append(sub_key)

        for value in prepare_delete_key:
            del self.sim_hash_dict[value]

        return self.sim_hash_dict
