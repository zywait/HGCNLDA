from datetime import datetime
import os
import time
import argparse
import torch
import pytorch_lightning as pl
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.utilities import move_data_to_device
from pytorch_lightning.loggers import TensorBoardLogger
from tqdm import tqdm
import numpy as np
import pandas as pd
import scipy.io as scio
from pprint import pformat

from src import metric_fn
from src.utils import init_logger, logger
from src.dataloader import CVDataset, RDDataset
from src.HGCNLDA.model import SGALDA, HGCNLDA
from src.HGCNLDA.abalation_model import *


@torch.no_grad()
def train_test_fn(model, train_loader, val_loader, save_file_format=None):
    device = model.device
    state = model.training
    model.eval()
    scores, labels, edges = [], [], []
    for batch in train_loader:
        model.train_step(batch)
    for batch in val_loader:
        batch = move_data_to_device(batch, device)
        output = model.test_step(batch)
        label, score = output["label"], output["predict"]
        edge = batch.interaction_pair[:, batch.valid_mask.reshape(-1)]
        scores.append(score.detach().cpu())
        labels.append(label.cpu())
        edges.append(edge.cpu())
    model.train(state)
    scores = torch.cat(scores).numpy()
    labels = torch.cat(labels).numpy()
    edges = torch.cat(edges, dim=1).numpy()
    eval_star_time_stamp = time.time()
    metric = metric_fn.evaluate(predict=scores, label=labels)
    eval_end_time_stamp = time.time()
    logger.info(f"eval time cost: {eval_end_time_stamp - eval_star_time_stamp}")
    if save_file_format is not None:
        save_file = save_file_format.format(aupr=metric["aupr"], auroc=metric["auroc"])
        scio.savemat(save_file, {"row": edges[0],
                                 "col": edges[1],
                                 "score": scores,
                                 "label": labels,
                                 })
        logger.info(f"save time cost: {time.time() - eval_end_time_stamp}")
    return scores, labels, edges, metric


@torch.no_grad()
def test_fn(model, val_loader, save_file_format=None, save_emb_path=None):
    device = model.device
    state = model.training
    model.eval()
    scores, labels, edges, lnc_embeddings, dis_embeddings = [], [], [], [], []
    for batch in val_loader:
        batch = move_data_to_device(batch, device)
        output = model.step(batch)
        label, score = output["label"], output["predict"]
        embedding = output['embedding'].cpu()
        edge = batch.interaction_pair[:, batch.valid_mask.reshape(-1)]
        scores.append(score.detach().cpu())
        labels.append(label.cpu())
        edges.append(edge.cpu())
        lnc_size = batch.valid_mask.shape[0]
        dis_size = batch.valid_mask.shape[1]
        lnc_embedding = embedding[:lnc_size, :]
        dis_embedding = embedding[lnc_size:lnc_size+dis_size, :]
        lnc_embeddings.append(lnc_embedding)
        dis_embeddings.append(dis_embedding)
    model.train(state)
    scores = torch.cat(scores).numpy()
    labels = torch.cat(labels).numpy()
    edges = torch.cat(edges, dim=1).numpy()
    lnc_embeddings = torch.cat(lnc_embeddings, dim=0).numpy()
    dis_embeddings = torch.cat(dis_embeddings, dim=0).numpy()
    eval_star_time_stamp = time.time()
    metric = metric_fn.evaluate(predict=scores, label=labels)
    eval_end_time_stamp = time.time()
    logger.info(f"eval time cost: {eval_end_time_stamp-eval_star_time_stamp}")
    if save_file_format is not None:
        save_file = save_file_format.format(aupr=metric["aupr"], auroc=metric["auroc"])
        scio.savemat(save_file, {"row": edges[0],
                      "col": edges[1],
                      "score": scores,
                      "label": labels,
                      })
        logger.info(f"save time cost: {time.time()-eval_end_time_stamp}")
    if save_emb_path is not None:
        save_file = save_emb_path + '-lnc_embeddings.npy'
        np.save(save_file, np.array(lnc_embeddings))
        save_file = save_emb_path + '-dis_embeddings.npy'
        np.save(save_file, np.array(dis_embeddings))
    return scores, labels, edges, metric


def train_fn(config, model, train_loader, val_loader):
    checkpoint_callback = ModelCheckpoint(monitor="val/auroc",
                                          mode="max",
                                          save_top_k=1,
                                          verbose=False,
                                          save_last=True)
    lr_callback = pl.callbacks.LearningRateMonitor("epoch")
    now = datetime.now()
    logger = TensorBoardLogger(f'tb_logs_' + now.strftime('%Y-%m-%d'), name=f'my_model_{config.time_stamp}')
    trainer = Trainer(max_epochs=config.epochs,
                      default_root_dir=config.log_dir,
                      profiler=config.profiler,
                      fast_dev_run=False,
                      # checkpoint_callback=checkpoint_callback,
                      callbacks=[lr_callback],
                      gpus=config.gpus,
                      check_val_every_n_epoch=1,
                      logger=logger
                      )
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    if not hasattr(config, "dirpath"):
        config.dirpath = trainer.checkpoint_callback.dirpath
    # checkpoint and add path
    # checkpoint = torch.load("lightning_logs/version_7/checkpoints/epoch=85.ckpt")
    # trainer.on_load_checkpoint(checkpoint)
    print(model.device)


def train(config):
    time_stamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    if config.model is not None:
        model_cls = MODEL_REGISTRY.get(config.model)
    datasets = RDDataset(dataset_name=config.dataset_name, lnc_neighbor_num=config.lnc_neighbor_num,
                         disease_neighbor_num=config.disease_neighbor_num, mi_neighbor_num=config.mi_neighbor_num)
    log_dir = os.path.join(f"{config.comment}", f"{config.split_mode}-{config.n_splits}-fold", f"{config.dataset_name}",
                           f"{config.seed}", f"{model_cls.__name__}", f"{time_stamp}")
    config.log_dir = log_dir
    config.n_lnc = datasets.lnc_num
    config.n_disease = datasets.disease_num
    config.n_mi = datasets.mi_num

    config.size_u = datasets.lnc_num
    config.size_v = datasets.disease_num
    config.size_w = datasets.mi_num

    config.gpus = 1 if torch.cuda.is_available() else 0
    config.pos_weight = datasets.pos_weight
    config.time_stamp = time_stamp
    logger = init_logger(log_dir)
    logger.info(pformat(vars(config)))
    config.dataset_type = config.dataset_dype if config.dataset_type is not None else model_cls.DATASET_TYPE
    cv_spliter = CVDataset(datasets, split_mode=config.split_mode, n_splits=config.n_splits,
                           lnc_idx=config.lnc_idx, disease_idx=config.disease_idx,
                           train_fill_unknown=config.train_fill_unknown,
                           global_test_all_zero=config.global_test_all_zero, seed=config.seed,
                           dataset_type=config.dataset_type)
    pl.seed_everything(config.seed)
    scores, labels, edges, split_idxs = [], [], [], []
    metrics = {}
    start_time_stamp = time.time()
    aurocs = []
    auprs = []
    for split_id, datamodule in enumerate(cv_spliter):
        # if split_id not in [4, 5]:
        #     continue
        config.split_id = split_id
        split_start_time_stamp = time.time()

        datamodule.prepare_data()
        datamodule.setup()
        train_loader = datamodule.train_dataloader()
        val_loader = datamodule.val_dataloader()
        config.pos_weight = train_loader.dataset.pos_weight
        model = model_cls(**vars(config))
        model = model.cuda() if config.gpus else model

        if split_id==0:
            logger.info(model)
        logger.info(f"begin train fold {split_id}/{len(cv_spliter)}")
        train_fn(config, model, train_loader=train_loader, val_loader=val_loader)
        logger.info(f"end train fold {split_id}/{len(cv_spliter)}")
        save_file_format = os.path.join(config.log_dir,
                                        f"{config.dataset_name}-{config.split_id} fold-{{auroc}}-{{aupr}}.mat")
        # save_emb_path = os.path.join(config.log_dir, f"{config.dataset_name}-{config.split_id} fold")
        # score, label, edge, metric = test_fn(model, val_loader, save_file_format, save_emb_path)
        score, label, edge, metric = test_fn(model, val_loader, save_file_format)
        # score, label, edge, metric = train_test_fn(model, train_loader, val_loader, save_file_format)
        metrics[f"{split_id}"] = metric
        aurocs.append(metric['auroc'])
        auprs.append(metric['aupr'])
        scores.append(score)
        labels.append(label)
        edges.append(edge)
        split_idxs.append(np.ones(len(score), dtype=int)*split_id)
        logger.info(f"{split_id}/{len(cv_spliter)} folds: {metric}")
        logger.info(f"{split_id}/{len(cv_spliter)} folds time cost: {time.time()-split_start_time_stamp}")

        if config.debug:
            break
    end_time_stamp = time.time()
    logger.info(f"total time cost:{end_time_stamp-start_time_stamp}")
    scores = np.concatenate(scores, axis=0)
    labels = np.concatenate(labels, axis=0)
    edges = np.concatenate(edges, axis=1)
    split_idxs = np.concatenate(split_idxs, axis=0)
    final_metric = metric_fn.evaluate(predict=scores, label=labels, is_final=True)
    metrics["final"] = final_metric
    metrics["std"] = {'aupr': np.std(auprs, ddof=1), 'auroc': np.std(aurocs, ddof=1)}
    metrics = pd.DataFrame(metrics).T
    metrics.index.name = "split_id"
    metrics["seed"] = config.seed
    logger.info(f"final {config.dataset_name}-{config.split_mode}-{config.n_splits}-fold-auroc={final_metric['auroc']}-aupr={final_metric['aupr']}")
    output_file_name = f"final-{config.dataset_name}-{config.split_mode}-{config.n_splits}-auroc={final_metric['auroc']}-aupr={final_metric['aupr']}-fold"
    scio.savemat(os.path.join(log_dir, f"{output_file_name}.mat"),
                 {"row": edges[0],
                  "col": edges[1],
                  "score": scores,
                  "label": labels,
                  "split_idx":split_idxs}
                 )
    with pd.ExcelWriter(os.path.join(log_dir, f"{output_file_name}.xlsx")) as f:
        metrics.to_excel(f, sheet_name="metrics")
        params = pd.DataFrame({key:str(value) for key, value in vars(config).items()}, index=[str(time.time())])
        for key, value in final_metric.items():
            params[key] = value
        params["file"] = output_file_name
        params.to_excel(f, sheet_name="params")

    logger.info(f"save final results to r'{os.path.join(log_dir, output_file_name)}.mat'")
    logger.info(f"final results: {final_metric}")


def parse(print_help=False):
    parser = argparse.ArgumentParser(conflict_handler="resolve")
    parser.add_argument("--model", default="HGCNLDA", type=str)
    parser.add_argument("--epochs", default=300, type=int)
    parser.add_argument("--debug", default=False, action="store_true")
    parser.add_argument("--profiler", default=False, type=str)
    parser.add_argument("--comment", default="runs", type=str, help="experiment name")
    parser = RDDataset.add_argparse_args(parser)
    parser = CVDataset.add_argparse_args(parser)
    parser = HGCNLDA.add_model_specific_args(parser)
    args = parser.parse_args()
    if print_help:
        parser.print_help()
    return args


def pathIterator(comment):
    split_modes = sorted(os.listdir(comment))
    for split_mode in split_modes:
        for dataset in os.listdir(os.path.join(comment, split_mode)):
            for model in os.listdir(os.path.join(comment, split_mode, dataset)):
                for seed in os.listdir(os.path.join(comment, split_mode, dataset, model)):
                    for time_stamp in os.listdir(os.path.join(comment, split_mode, dataset, model, seed)):
                        yield comment, split_mode, dataset, model, seed, time_stamp


def report(comment, collect_parma=True):
    metrics = []
    params = []
    for comment, split_mode, dataset, model, seed, time_stamp in pathIterator(comment):
        dataset_dir = os.path.join(comment, split_mode, dataset, model, seed, time_stamp)
        valid_files = sorted([ file for file in os.listdir(dataset_dir) if file.endswith(".xlsx") and file.startswith("final")])
        for file in tqdm(valid_files):
            with pd.ExcelFile(os.path.join(dataset_dir, file)) as reader:
                metric = reader.parse(reader.sheet_names[0], index_col=0)
                if collect_parma:
                    param = reader.parse("params", index_col=0)
                    params.append(param)
                    metric["index"] = param.index[0]
                    metric["seed"] = param["seed"].iloc[0]
                metric["model"] = model
                metric["split_mode"] = split_mode
                metric["dataset"] = dataset
                metric["comment"] = comment
                metrics.append(metric)
    metrics = pd.concat(metrics)
    metrics.index.name = "split_id"
    if collect_parma:
        params = pd.concat(params)
    with pd.ExcelWriter(f"{comment}.xlsx") as writer:
        metrics.to_excel(writer, sheet_name="metrics")
        if collect_parma:
            params.to_excel(writer, sheet_name="params")


if __name__=="__main__":
    args = parse(print_help=True)
    train(args)
