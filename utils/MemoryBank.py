import os, sys
import torch
import numpy as np

class MemoryBank(object):
    """
    This is a general type of MIL memory bank.
    The memory bank could be a general two-level container (bag-level and instance-level).
    I recommend using structured memory bank (`torch.tensor` or `numpy.array`).
    See `TensorMemoryBank` for more details.

    Notes:
        1. update(): put values into memory bank.
        2. get_rank(): get the relative rank (0~1) 
        3. get_weight(): get the loss weight given relative rank and ratio.
        4. different pooling method (max and average)
        5. plug-and-play cal weight function enabled (updated 2020.2.23)
            currently, weight_func must accept (bag_index, inner_index, ranks, nodule_ratios)
            as input argument.
    
    Args:
        dictionary: (obj) a two-level container (bag level and ins level)
        mmt: (float) momentum for updating the dictionary
        weight_func: (callable) weight calculating function.
    """
    def __init__(self, dictionary={}, mmt=0.75, weight_func=None):
        super(MemoryBank, self).__init__()
        self.dictionary = dictionary
        self.mmt = mmt
        self.weight_func = None

    def state_dict(self):
        return self.dictionary

    def load(self, load_dir=None, resume=None):
        """
        Load memory bank according to resume config and logger.
        Note that logger is used to help loading (logger.load() function should be implemented) (TODO)
        """
        if load_dir is not None:
            if resume > 0:
                self.dictionary = torch.load(os.path.join(load_dir, "res{}.pth".format(resume)))
                ##BC code, if dictionary, find "memory_bank" key
                if isinstance(self.dictionary, dict):
                    self.dictionary = self.dictionary["memory_bank"]


    def update(self, bag_index, inner_index, instance_preds):
        """
        Since this is a general type memory bank, should check the type of
        memory bank and use different indexing operation (TODO)

        Args:
            bag_index: (list) bag key
            inner_index: (list) instance key
            instance_preds: (torch.Tensor) result of prediction
        """
        result = instance_preds.cpu().detach().view(-1)
        ##if is a dict or list, use for loop
        if isinstance(self.dictionary, dict) or isinstance(self.dictionary, list):
            for k in range(len(inner_index)):
                self.dictionary[bag_index[k]][inner_index[k]] = self.mmt * \
                    self.dictionary[bag_index[k]][inner_index[k]] + \
                    (1 - self.mmt) * result[k]
        elif isinstance(self.dictionary, torch.Tensor):
            self.dictionary[bag_index, inner_index] = self.mmt * \
                    self.dictionary[bag_index, inner_index] + \
                    (1 - self.mmt) * result

    def _sort(self):
        raise NotImplementedError

    def get_rank(self, bag_index, inner_index):
        """
        get relative rank of instances inside a bag

        relative rank = absolute rank / (length - 1)
        
        Example:
            self.dictionary is [[0.1,0.2,0.3,0.05],
                                [0.3,0.6,0.2,0.1]]
            
            get_rank([0,1], [0,0]) return [0.33, 0.66]
            since 0.1 in first row is the second lowest element (absolute rank=1) 
            and 0.3 is the third lowest element in the second row.
        
        Notes:
            only implemented in sub-class. (2020.2.4)
        """
        raise NotImplementedError

    def get_weight(self, bag_index, inner_index, nodule_ratios, **kwargs):
        """
        This function calls get_rank() and cal_weight() to get the weight.
        """
        ranks = self.get_rank(bag_index, inner_index)
        if self.weight_func is None:
            weights = self.cal_weight(ranks, nodule_ratios)
        else:
            weights = self.weight_func(ranks, nodule_ratios, bag_index, inner_index, **kwargs)
        return weights

    def cal_weight(self, ranks, nodule_ratios):
        """
        This is a plug-in function, could be modified/replaced in the future.
        Args:
            ranks: (torch.Tensor) [M, ] relative ranks of the prediction.
            nodule_ratios: (torch.Tensor) [M, ] nodule ratios to calculate the RCE weight.
            (Reference: Rectified Cross Entropy Loss)

        Notes:
            1. Currently we use threshold linear function
            2. It is promised that if nodule ratio is zero, the weight is approximately zero.
        """
        weights = (ranks / (1 - nodule_ratios.mean())).clamp(max=1.0, min=0.0)
        weights[nodule_ratios<1e-6] = 1.0
        return weights

    def max_pool(self):
        return_list = []
        for bag in self.dictionary:
            pred = max(bag)
            return_list.append(pred)
        
        return return_list

    def avg_pool(self, bag_lengths):
        return_list = []
        for idx, bag in enumerate(self.dictionary):
            pred = sum(bag[:bag_lengths[idx]]) / bag_lengths[idx]
            return_list.append(pred)
        
        return return_list

    def to_list(self, bag_lengths):
        return_list = []
        for idx, bag in enumerate(self.dictionary):
            return_list.extend(list(bag[:bag_lengths[idx]]))
        return return_list

class TensorMemoryBank(MemoryBank):
    """
    torch.Tensor version of memory bank. Since it is structured data, something is different:
        1. A global rank Tensor (sized [N, M]) is maintained, where N is bag number
         and M is the max bag size
        2. Tensor based implementation of ranking.
        3. An initialize function for the memory bank embedded.
    
    Args:
        bag_num: (int) the total number of bag
        max_ins_num: (int) maximum bag length. This could be equal to 
            max(bag_lens)
        bag_lens: (list of int) bag length for each bag, sized [bag_num, ]
    """
    def __init__(self, bag_num, max_ins_num=None, bag_lens=None, mmt=0.75, weight_func=None):
        self.bag_lens = torch.tensor(bag_lens).view(bag_num, 1)
        self.max_ins_num = max_ins_num
        if self.max_ins_num is None and self.bag_lens is None:
            raise Exception
        elif self.bag_lens is None:
            self.bag_lens = torch.tensor([max_ins_num] * bag_num).view(bag_num, 1)
        elif self.max_ins_num is None:
            self.max_ins_num = max(self.bag_lens)

        init_bank = self.init_bank(bag_num, max_ins_num)
        super(TensorMemoryBank, self).__init__(init_bank, mmt, weight_func)
        self.rank_tensor = self.update_rank()

    def init_bank(self, bag_num, max_ins_num):
        """
        Since each bag(a row) has a valid length, elements
        that are out of index are marked as a pre-defined value (-1.0 since no score < 0)
        """
        init_tensor = torch.ones([bag_num, max_ins_num]).float()
        for idx, bag_len in enumerate(self.bag_lens):
            init_tensor[idx, bag_len:] = -1.0

        return init_tensor
            
    def update_rank(self):
        """
        Thanks qsf for the idea of implementing this!
        This function update the relative rank for memory bank (sized [M,N]).
        Notice that for each bag the bag length is not the same. The last few
        elements in each row would be negative.

        """
        abs_ranks = self.dictionary.argsort(dim=1).argsort(dim=1)
        ##ignore last few elements by setting them as minors
        abs_ranks = abs_ranks - self.max_ins_num + self.bag_lens
        relative_ranks = abs_ranks.float() / (self.bag_lens-1).float()
        return relative_ranks
        
    def get_rank(self, bag_index, inner_index):
        return self.rank_tensor[bag_index, inner_index]


class MRCETensorMemoryBank(TensorMemoryBank):
    """
    re-implement the cal_weight function (provided by ltc).
    """
    def __init__(self, bag_num, max_ins_num=None, bag_lens=None, mmt=0.75):
        super(MRCETensorMemoryBank, self).__init__(bag_num, max_ins_num, bag_lens, mmt)

    def get_weight(self, bag_index, inner_index, nodule_ratios, preds=None):
        ranks = self.get_rank(bag_index, inner_index)
        weights = self.cal_weight(bag_index, inner_index, ranks, nodule_ratios, preds)
        return weights

    def cal_weight(self, bag_index, inner_index, ranks, nodule_ratios, preds):
        """
        Our weight calculating functtion by ltc.
        """
        weights = torch.ones_like(ranks).float()
        threshold = torch.stack((1-nodule_ratios, ((self.dictionary<0.5) & (self.dictionary>=0)).float().mean(dim=1)[bag_index])).max(dim=0)[0]
        mask1 = (ranks < threshold) & (self.dictionary[bag_index, inner_index] < 0.5)
        weights[mask1] = 0.0
        weights[nodule_ratios<1e-6] = 1.0
        return weights


class MRCEV3TensorMemoryBank(TensorMemoryBank):
    """
    re-implement the cal_weight function (provided by ltc).
    """
    def __init__(self, bag_num, max_ins_num=None, bag_lens=None, mmt=0.75):
        super(MRCEV3TensorMemoryBank, self).__init__(bag_num, max_ins_num, bag_lens, mmt)

    def get_weight(self, bag_index, inner_index, nodule_ratios, preds=None):
        ranks = self.get_rank(bag_index, inner_index)
        weights = self.cal_weight(bag_index, inner_index, ranks, nodule_ratios, preds)
        return weights

    def cal_weight(self, bag_index, inner_index, ranks, nodule_ratios, preds):
        """
        Our weight calculating function (v3)

        """
        weights = torch.ones_like(ranks).float()
        th1 = 1- nodule_ratios
        th2 = ((self.dictionary<0.5) & (self.dictionary>=0)).float().mean(dim=1)
        weights[(ranks>=th2) & (ranks<th1)] = (ranks - th2) / (th1 - th2)
        weights[ranks<th2] = 0.0
        weights[nodule_ratios<1e-6] = 1.0
        return weights