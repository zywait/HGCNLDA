import os
import sys
import logging
import torch
import pandas as pd
import numpy as np

logger = logging.getLogger("SGALDA")

class NoParsingFilter(logging.Filter):
    def filter(self, record):
        if record.funcName=="summarize" and record.levelno==20:
            return False
        if record.funcName=="_info" and record.funcName=="distributed.py" and record.lineno==20:
            return False
        return True

def init_logger(log_dir):
    lightning_logger = logging.getLogger("pytorch_lightning.core.lightning")
    lightning_logger.addFilter(NoParsingFilter())
    distributed_logger = logging.getLogger("pytorch_lightning.utilities.distributed")
    distributed_logger.addFilter(NoParsingFilter())
    format = '%Y-%m-%d %H-%M-%S'
    fm = logging.Formatter(fmt="[%(asctime)s][%(levelname)s] %(name)s: %(lineno)4d: %(message)s",
                           datefmt="%m/%d %H:%M:%S")
    ch = logging.StreamHandler()
    ch.setFormatter(fm)
    logger.addHandler(ch)
    logger.setLevel(logging.INFO)
    logger.info(f"terminal cmd: python {' '.join(sys.argv)}")
    if len(logger.handlers)==1:
        import time
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        else:
            logger.warning(f"error file exist! {log_dir}")
            logger.warning("please init new 'comment' value")
            # exit(0)
        logger.propagate = False
        log_file = os.path.join(log_dir, f"{time.strftime(format, time.localtime())}.log")
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(fm)
        logger.addHandler(file_handler)
        logger.info(f"log file: {log_file}")
    else:
        logger.warning("init_logger fail")
    return logger

def select_topk(data, k=-1):
    if k<=0:
        return data
    assert k<=data.shape[1]
    val, col = torch.topk(data ,k=k)
    col = col.reshape(-1)
    row = torch.ones(1, k, dtype=torch.int)*torch.arange(data.shape[0]).view(-1, 1)
    row = row.view(-1).to(device=data.device)
    new_data = torch.zeros_like(data)
    new_data[row, col] = data[row, col]
    # new_data[row, col] = 1.0
    return new_data


def disease_group(RNA_idx, RDA:pd.DataFrame):
    group_mask = (RDA.values[RNA_idx, :] == 1)
    disease_idxs = np.arange(len(group_mask))[group_mask]
    return disease_idxs


def disease_group_similarity(disease_similarity:pd.DataFrame, disease_idxs1,disease_idxs2):
    dis_num1 = len(disease_idxs1)
    dis_num2 = len(disease_idxs2)
    dis_group_similarity = np.zeros((dis_num1, dis_num2))
    for i in range(dis_num1):
        for j in range(dis_num2):
            dis_group_similarity[i][j] = disease_similarity.iloc[disease_idxs1[i], disease_idxs2[j]]
    return dis_group_similarity


def RNA_similarity(RDA:pd.DataFrame, disease_similarity:pd.DataFrame, RNA1_idx, RNA2_idx):
    group1_idxs = disease_group(RNA1_idx, RDA)
    group2_idxs = disease_group(RNA2_idx, RDA)
    len1 = len(group1_idxs)
    len2 = len(group2_idxs)
    if len1 == 0 or len2 == 0:
        return 0

    s = disease_group_similarity(disease_similarity, group1_idxs, group2_idxs)
    sum1=0
    sum2=0

    if len2 > 0:
        for i in range(len1):
           sum1 += (max(s[i, :]))
    if len1 > 0:
        for i in range(len2):
            sum2 += (max(s[:, i]))
    return (sum1 + sum2) / (len1 + len2)


def PBPA(RNA_i, RNA_j, di_sim, rna_di):
    diseaseSet_i = rna_di[RNA_i] > 0
    diseaseSet_j = rna_di[RNA_j] > 0
    diseaseSim_ij = di_sim[diseaseSet_i][:, diseaseSet_j]
    ijshape = diseaseSim_ij.shape
    if ijshape[0] == 0 or ijshape[1] == 0:
        return 0
    return (sum(np.max(diseaseSim_ij, axis=0)) + sum(np.max(diseaseSim_ij, axis=1))) / (ijshape[0] + ijshape[1])


def rna_func_sim(RDA, disease_similarity, size):
    rna_similarity = np.zeros((size, size))
    for i in range(size):
        for j in range(i+1, size):
            rna_similarity[i, j] = rna_similarity[j, i] = PBPA(i, j, disease_similarity, RDA)
    return rna_similarity


def csv_to_excel(from_file, to_file):
    csv = pd.read_csv(from_file, encoding='utf-8', index_col=0)
    csv.to_excel(to_file, header=None, index=False)

