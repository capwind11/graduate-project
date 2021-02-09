from difflib import SequenceMatcher

from tqdm import tqdm

from layers.layer import Layer
import os

# 树节点结构
class TreeNode:
    def __init__(self, tag=-1):
        self.childs = dict() # token->treenode
        self.tag = tag # non -1 denotes the id of the cluster

# 树结构
class Trie:
    def __init__(self):
        self.root = TreeNode()

    # 插入操作
    def insert(self, template_list, cluster_id):
        now = self.root
        # 从树里面搜索和寻找
        for token in template_list:
            if token not in now.childs:
                now.childs[token] = TreeNode()
            now = now.childs[token]
        now.tag = cluster_id

    # 查找对应节点
    def find(self, template_list):
        now = self.root
        wd_count = 0
        for token in template_list:
            if token in now.childs:
                now = now.childs[token]
            # 变量类型
            elif '<*>' in now.childs:
                wd_count += 1
                now = now.childs['<*>']
            else:
                return -1
        if template_list and wd_count/len(template_list) > 0.5:
            return -1
        return now.tag

# 去掉变量的掩码
def maskdel(template):
    temp = []
    for token in template:
        if token == '<*>':
            temp.append('')
        else:
            temp.append(token)
    return temp

# 掩码层
class MaskLayer(Layer):
    def __init__(self, sim_hash_dict: dict, max_mask_loop: int = 0):
        self.sim_hash_dict = sim_hash_dict
        self.max_mask_loop = max_mask_loop
        # self.output_file = output_file

    # 替换指定位置的单词
    def replace_char(self, str, char, index):
        string = list(str)
        string[index] = char
        return ''.join(string)

    # 拿到日志条目的模板，long common sequence
    def getTemplate(self, lcs, seq):
        retVal = []
        if not lcs:
            return retVal

        # 逆转数组，当做栈用
        lcs = lcs[::-1]
        i = 0
        for token in seq:
            i += 1
            if token == lcs[-1]:
                retVal.append(token)
                lcs.pop()
            else:
                retVal.append('<*>')
            if not lcs:
                break
        while i < len(seq):
            retVal.append('<*>')
            i += 1
        return retVal

    # 每两条数据之间都要对比，时间复杂度太高
    # 最长公共子序列算
    def LCS(self,seq1, seq2):
        lengths = [[0 for j in range(len(seq2)+1)] for i in range(len(seq1)+1)]
        # row 0 and column 0 are initialized to 0 already
        # 先返回最大公共子序列长度
        for i in range(len(seq1)):
            for j in range(len(seq2)):
                if seq1[i] == seq2[j]:
                    lengths[i+1][j+1] = lengths[i][j] + 1
                else:
                    lengths[i+1][j+1] = max(lengths[i+1][j], lengths[i][j+1])
        for i in range(len(seq1)+1):
            print(lengths[i])
        # read the substring out from the matrix
        result = []
        # 从后往前搜索执行
        # 匹配了相同数目的
        lenOfSeq1, lenOfSeq2 = len(seq1), len(seq2)
        while lenOfSeq1 != 0 and lenOfSeq2 != 0:
            if lengths[lenOfSeq1][lenOfSeq2] == lengths[lenOfSeq1-1][lenOfSeq2]:
                lenOfSeq1 -= 1
            elif lengths[lenOfSeq1][lenOfSeq2] == lengths[lenOfSeq1][lenOfSeq2-1]:
                lenOfSeq2 -= 1
            else:
                assert seq1[lenOfSeq1-1] == seq2[lenOfSeq2-1]
                result.append(seq1[lenOfSeq1-1])
                lenOfSeq1 -= 1
                lenOfSeq2 -= 1
        result = result[::-1]
        print(result)
        return result

    def mask_simple(self, seq1, seq2):
        res = []
        for i in range(len(seq1)):
            if seq1[i] == seq2[i]:
                res.append(seq1[i])
            else:
                res.append('<*>')
        return res

    # sim_hash_dict格式为id-->list<content、dictionary_list、lineId>
    def run(self):
        templates = dict()
        template_list = list()
        template_dict = dict()
        # import pdb; pdb.set_trace()
        for key in tqdm(self.sim_hash_dict, desc='templates initialization'):
            # import pdb; pdb.set_trace()
            sorted_list = [[x['message'], x['LineId']] for x in self.sim_hash_dict[key]]
            # 取同类型第一个作template即可
            template = sorted_list[0][0]
            for index, rs in enumerate(sorted_list):
                if index == 0:
                    continue
                # print('TEP: {}'.format(template))
                # print('ORG: {}'.format(rs[0]))
                # print('LCS: {}'.format(self.LCS(rs[0], template)))
                temp_template = self.getTemplate(self.LCS(rs[0], template), template)
                # temp_template = self.mask_simple(rs[0], template)
                # 一条一条日志叠起来
                if temp_template != '':
                    template = temp_template
            # 建立list和从key ——> template的映射
            template_list.append([key, template])
            template_dict[key] = template
        # import pdb; pdb.set_trace()
        template_list = sorted(template_list, key=lambda entry: maskdel(entry[1]))
        trie = Trie()
        # 构建树
        for entry in tqdm(template_list, desc='trie group'):
            tag = trie.find(entry[1])
            if tag == -1:
                # 没找到就插入
                trie.insert(entry[1], entry[0])
            else:
                self.sim_hash_dict[tag].extend(self.sim_hash_dict[entry[0]])
                self.sim_hash_dict.pop(entry[0])
        # import pdb; pdb.set_trace()
        for key in tqdm(self.sim_hash_dict, desc='generate output'):
            sorted_list = [[x['message'], x['LineId']] for x in self.sim_hash_dict[key]]
            clustIDs = list()
            for index, rs in enumerate(sorted_list):
                clustIDs.append(rs[1])
            template = ' '.join(template_dict[key])
            # print(len(clustIDs))
            templates[template] = clustIDs
            # print(templates)
            for index, rs in enumerate(self.sim_hash_dict[key]):
                rs['template'] = template
        print('After mask layer finish, tot, total: {} bin(s)'.format(len(self.sim_hash_dict)))
        return self.sim_hash_dict, templates
