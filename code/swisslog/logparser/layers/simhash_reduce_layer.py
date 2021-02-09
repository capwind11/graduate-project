# -*- coding: utf-8 -*-

from tqdm import tqdm

from layers.layer import Layer


class SimHashReduceLayer(Layer):
    """
    SimHash聚类，合并分组之间相似的数据
    """

    def __init__(self, sim_hash_dict: dict(), sim_value: float = 0.9):
        self.sim_hash_dict = sim_hash_dict
        self.sim_value = sim_value

    def run(self) -> dict():
        processed_key = list()
        prepare_delete_key = list()
        for key in tqdm(self.sim_hash_dict.keys()):

            if key in processed_key:
                continue

            # content_list = [x['message'] for x in self.sim_hash_dict[key]]
            # simhash_layer = SimHashGroupLayer(content_list, 5)
            # sim_dict = simhash_layer.run()
            #
            # max_length = 0
            # max_group = list()
            # for key in sim_dict.keys():
            #     if len(sim_dict[key].keys()) > max_length:
            #         max_group = sim_dict[key]
            #         max_length = len(sim_dict[key].keys())
            # source_message = max_group[0]

            # TODO:应该选取一组里面占比最高的那条记录，可以选用SimHashGroup，降低分组维度
            source_message = self.sim_hash_dict[key][0]['simhash']
            processed_key.append(key)

            for sub_key in self.sim_hash_dict.keys():
                if sub_key == key or sub_key in processed_key:
                    continue
                target_message = self.sim_hash_dict[sub_key][0]['simhash']
                if source_message.similarity(target_message) > self.sim_value:
                    processed_key.append(sub_key)
                    self.sim_hash_dict[key].extend(self.sim_hash_dict[sub_key])
                    prepare_delete_key.append(sub_key)
        for value in prepare_delete_key:
            del self.sim_hash_dict[value]

        return self.sim_hash_dict
