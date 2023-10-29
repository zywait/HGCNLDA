"""https://github.com/storyandwine/LAGCN
Predicting Drug-Disease Associations through Layer Attention Graph Convolutional Networks
"""

import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
import scipy.io as scio
import os

def get_metrics(real_score, predict_score):
    sorted_predict_score = np.array(
        sorted(list(set(np.array(predict_score).flatten()))))
    sorted_predict_score_num = len(sorted_predict_score)
    thresholds = sorted_predict_score[np.int32(
        sorted_predict_score_num*np.arange(1, 1000)/1000)]
    thresholds = np.mat(thresholds)
    thresholds_num = thresholds.shape[1]

    predict_score_matrix = np.tile(predict_score, (thresholds_num, 1))
    negative_index = np.where(predict_score_matrix < thresholds.T)
    positive_index = np.where(predict_score_matrix >= thresholds.T)
    predict_score_matrix[negative_index] = 0
    predict_score_matrix[positive_index] = 1
    TP = predict_score_matrix.dot(real_score.T)
    FP = predict_score_matrix.sum(axis=1)-TP
    FN = real_score.sum()-TP
    TN = len(real_score.T)-TP-FP-FN

    fpr = FP/(FP+TN)
    tpr = TP/(TP+FN)
    ROC_dot_matrix = np.mat(sorted(np.column_stack((fpr, tpr)).tolist())).T
    ROC_dot_matrix.T[0] = [0, 0]
    ROC_dot_matrix = np.c_[ROC_dot_matrix, [1, 1]]
    x_ROC = ROC_dot_matrix[0].T
    y_ROC = ROC_dot_matrix[1].T
    auc = 0.5*(x_ROC[1:]-x_ROC[:-1]).T*(y_ROC[:-1]+y_ROC[1:])

    recall_list = tpr
    precision_list = TP/(TP+FP)
    PR_dot_matrix = np.mat(sorted(np.column_stack(
        (recall_list, precision_list)).tolist())).T
    PR_dot_matrix.T[0] = [0, 1]
    PR_dot_matrix = np.c_[PR_dot_matrix, [1, 0]]
    x_PR = PR_dot_matrix[0].T
    y_PR = PR_dot_matrix[1].T
    aupr = 0.5*(x_PR[1:]-x_PR[:-1]).T*(y_PR[:-1]+y_PR[1:])

    f1_score_list = 2*TP/(len(real_score.T)+TP-TN)
    accuracy_list = (TP+TN)/len(real_score.T)
    specificity_list = TN/(TN+FP)

    max_index = np.argmax(f1_score_list)
    f1_score = f1_score_list[max_index]
    accuracy = accuracy_list[max_index]
    specificity = specificity_list[max_index]
    recall = recall_list[max_index]
    precision = precision_list[max_index]

    return [aupr[0, 0], auc[0, 0], f1_score, accuracy, recall, specificity, precision]


def evaluate(predict, label, is_final=False):
    if not is_final:
        res = get_metrics(real_score=label, predict_score=predict)
    else:
        res = [None]*7
    aupr = metrics.average_precision_score(y_true=label, y_score=predict)
    auroc = metrics.roc_auc_score(y_true=label, y_score=predict)
    result = {"aupr":aupr,
              "auroc":auroc,
              "lagcn_aupr":res[0],
              "lagcn_auc":res[1],
              "lagcn_f1_score":res[2],
              "lagcn_accuracy":res[3],
              "lagcn_recall":res[4],
              "lagcn_specificity":res[5],
              "lagcn_precision":res[6]}
    return result

def roc_data(predict_score, real_score):
    sorted_predict_score = np.array(sorted(list(set(np.array(predict_score).flatten()))))
    sorted_predict_score_num = len(sorted_predict_score)
    thresholds = sorted_predict_score[np.int32(
        sorted_predict_score_num * np.arange(1, 1000) / 1000)]
    thresholds = np.mat(thresholds)
    thresholds_num = thresholds.shape[1]

    predict_score_matrix = np.tile(predict_score, (thresholds_num, 1))
    negative_index = np.where(predict_score_matrix < thresholds.T)
    positive_index = np.where(predict_score_matrix >= thresholds.T)
    predict_score_matrix[negative_index] = 0
    predict_score_matrix[positive_index] = 1
    TP = predict_score_matrix.dot(real_score.T)
    FP = predict_score_matrix.sum(axis=1) - TP
    FN = real_score.sum() - TP
    TN = len(real_score.T) - TP - FP - FN

    fpr = FP / (FP + TN)
    tpr = TP / (TP + FN)
    ROC_dot_matrix = np.mat(sorted(np.column_stack((fpr, tpr)).tolist())).T
    ROC_dot_matrix.T[0] = [0, 0]
    ROC_dot_matrix = np.c_[ROC_dot_matrix, [1, 1]]
    x_ROC = ROC_dot_matrix[0].T
    y_ROC = ROC_dot_matrix[1].T
    auc = 0.5 * (x_ROC[1:] - x_ROC[:-1]).T * (y_ROC[:-1] + y_ROC[1:])

    recall_list = tpr
    precision_list = TP / (TP + FP)
    PR_dot_matrix = np.mat(sorted(np.column_stack(
        (recall_list, precision_list)).tolist())).T
    PR_dot_matrix.T[0] = [0, 1]
    PR_dot_matrix = np.c_[PR_dot_matrix, [1, 0]]
    x_PR = PR_dot_matrix[0].T
    y_PR = PR_dot_matrix[1].T
    aupr = 0.5 * (x_PR[1:] - x_PR[:-1]).T * (y_PR[:-1] + y_PR[1:])
    f1_score_list = 2 * TP / (len(real_score.T) + TP - TN)
    accuracy_list = (TP + TN) / len(real_score.T)
    specificity_list = TN / (TN + FP)

    max_index = np.argmax(f1_score_list)
    # f1_score = f1_score_list[max_index]
    # accuracy = accuracy_list[max_index]
    # specificity = specificity_list[max_index]
    recall = recall_list[max_index]
    precision = precision_list[max_index]
    # auc = metrics.auc(fpr, tpr)
    # aupr = metrics.average_precision_score(y_true=real_score, y_score=predict_score)
    # auroc = metrics.roc_auc_score(y_true=real_score, y_score=predict_score)
    return {"auc": auc[0, 0],
            "aupr": aupr[0, 0],
            "fpr": fpr,
            "tpr": tpr,
            "recall": recall_list,
            "precision": precision_list}


def draw_roc_pr(alg_names, predict_scores, real_scores, fold_num=5):
    colorlist = plt.cm.cool(np.linspace(0, 1, fold_num))
    fig = plt.figure(figsize=(12, 6))
    ax_roc = fig.add_subplot(121)
    ax_roc.set_xlabel('False Positive Rate')
    ax_roc.set_ylabel('Ture Positive Rate')
    ax_roc.set_title("ROC Curves")
    ax_pr = fig.add_subplot(122)
    ax_pr.set_xlabel('Recall')
    ax_pr.set_ylabel('Precision')
    ax_pr.set_title("PR Curves")
    for alg, predict_score, real_score, color in zip(alg_names, predict_scores, real_scores, colorlist):
        data = roc_data(predict_score, real_score)
        # ax_roc.plot(data['fpr'], data['tpr'], color=color, linestyle='-', label=f'{alg}(AUC=%0.4f)' % data['auc'])
        # ax_pr.plot(data['recall'], data['precision'], color=color, linestyle='-', label=f'{alg}(AUPR=%0.4f)' % data['aupr'])
        ax_roc.plot(data['fpr'], data['tpr'], label=f'{alg}(AUC=%0.4f)' % data['auc'])
        ax_pr.plot(data['recall'], data['precision'], label=f'{alg}(AUPR=%0.4f)' % data['aupr'])
    ax_roc.legend()
    ax_pr.legend()
    plt.show()

def draw_roc(alg_names, predict_scores, real_scores, fold_num=5):
    colorlist = plt.cm.cool(np.linspace(0, 1, fold_num))
    plt.figure(figsize=(6, 6))
    plt.xlabel('False Positive Rate')
    plt.ylabel('Ture Positive Rate')
    plt.title("ROC Curves")
    for alg, predict_score, real_score, color in zip(alg_names, predict_scores, real_scores, colorlist):
        data = roc_data(predict_score, real_score)
        # plt.plot(data['fpr'], data['tpr'], color=color, linestyle='-', label=f'{alg}(AUC=%0.4f)' % data['auc'])
        plt.plot(data['fpr'], data['tpr'], label=f'{alg}(AUC=%0.4f)' % data['auc'])
    plt.legend()
    plt.show()

def draw_pr(alg_names, predict_scores, real_scores, fold_num=5):
    colorlist = plt.cm.cool(np.linspace(0, 1, fold_num))
    plt.figure(figsize=(6, 6))
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title("PR Curves")
    for alg, predict_score, real_score, color in zip(alg_names, predict_scores, real_scores, colorlist):
        data = roc_data(predict_score, real_score)
        # plt.plot(data['recall'], data['precision'], color=color, linestyle='-', label=f'{alg}(AUPR=%0.4f)' % data['aupr'])
        plt.plot(data['recall'], data['precision'], label=f'{alg}(AUPR=%0.4f)' % data['aupr'])
    plt.legend()
    plt.show()

if __name__ == '__main__':
    file_dir = "D:/Projects/Python/SGALDA/results/cv/MPSGLGCN/2022-05-12_21-18-27"
    alg_names = ['SGALDA fold1', 'SGALDA fold2', 'SGALDA fold3',
                 'SGALDA fold4', 'SGALDA fold5']
    predict_scores = []
    real_scores = []
    for filename in os.listdir(file_dir):
        if filename.startswith('RFLDA'):
            data = scio.loadmat(os.path.join(file_dir, filename))
            predict_score = data['score'].reshape(-1)
            real_score = data['label'].reshape(-1)
            predict_scores.append(predict_score)
            real_scores.append(real_score)
    draw_roc(alg_names, predict_scores, real_scores)
    draw_pr(alg_names, predict_scores, real_scores)
    draw_roc_pr(alg_names, predict_scores, real_scores)
