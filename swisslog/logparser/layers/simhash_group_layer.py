from tqdm import tqdm

from hashes.simhash import simhash
from layers.layer import Layer


class SimHashGroupLayer(Layer):
    """
    SimHash分组层，用指定的SimHash空间来对海量数据进行降数量级，用于组合模型的第一层
    """

    def __init__(self, df, hashbits: int = 23, keep_same_count: int = 0):
        self.df = df
        self.hashbits = hashbits
        self.keep_same_count = keep_same_count

    def run(self) -> dict:
        sim_hash_dict = dict()
        # tqdm 是一个显示进度条的python库
        # print(self.message_list)
        # print(type(self.df))
        for idx, value in self.df.iterrows():
            # hashbits，比较的hash位数
            # print(value)
            sim = simhash(value['Content'], hashbits=self.hashbits)
            sim_dict = dict(message=value['Content'], simhash=sim, LineId=value['LineId'])
            if sim.hash in sim_hash_dict.keys():
                sim_list = sim_hash_dict[sim.hash]
                if self.keep_same_count == 0 or self.keep_same_count <= len(sim_list):
                    sim_list.append(sim_dict)
                else:
                    print("已经达到分组保存容量的最大值，跳过词条记录")
            else:
                sim_list = list()
                sim_list.append(sim_dict)
                sim_hash_dict[sim.hash] = sim_list
        total_group = len(sim_hash_dict.keys())
        print('After Simhash Reduce, total:%s bin(s)' % len(sim_hash_dict.keys()))
        
        print("数据压缩比率为:%s" % (1 - total_group / len(self.df)))
        return sim_hash_dict
