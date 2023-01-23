from __future__ import absolute_import, division, print_function
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # del
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import argparse
import random
import sys
import re
import json
import jsonlines
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler, TensorDataset)
from tqdm import tqdm, trange
import torch.nn.functional
from pytorch_pretrained_bert.file_utils import WEIGHTS_NAME
from transformers import RobertaTokenizer, RobertaConfig
from pytorch_pretrained_bert.optimization import BertAdam
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from tensorboardX import SummaryWriter
from model import  RobertaMoEForSequenceClassification
import nltk
import logging
import pandas as pd
from tfidf_similarity import TfIdfSimilarity

logger = logging.getLogger(__name__)


LABELS = {"SUPPORTS":0, "REFUTES":1}

class InputExample(object):
    def __init__(self, idx, text_a, text_b=None, label=None,priori=None):
        '''
        Args:
            idx:   unique id
            text_a: response/claim
            text_b: context+evidence
            label:  positive / negative / NEI
            priori: priori distribution over experts based on rules
        '''
        self.idx = idx
        self.text_a = text_a
        self.text_b = text_b
        self.label = label
        self.priori = priori


class InputFeatures(object):
    def __init__(self, input_ids, input_mask, segment_ids, label_id, priori):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.priori = priori


class DataProcessor(object):
    def get_examples(self, data_dir, dataset=None):
        logger.info('Get examples from: {}.jsonl'.format(dataset))
        return self._create_examples(self._get_json_lines(os.path.join(data_dir, "{}.jsonl".format(dataset))))

    def get_labels(self):
        return [0, 1], len([0, 1])

    def _get_json_lines(cls,inp_file):
        lines = []
        with jsonlines.open(inp_file) as reader:
            for obj in reader:
                lines.append(obj)
                
        return lines

    def _create_examples(self, lines, max_evidences=1):
        examples = []
        obj = TfIdfSimilarity()
        for i, datapoint in enumerate(tqdm(lines)):
            if 'gold_evidence' in datapoint.keys():
                # evidence = datapoint['positive_evidence']
                # evidence_text = 'title: ' + evidence['title'] + ' content: ' + evidence['text']
                # datapoint['evidence_touse'] = evidence_text

                all_evidences = datapoint['gold_evidence'][:max_evidences]
                all_evidence_texts = ['title: ' + x['title'] + ' content: ' + x['text'] for x in all_evidences]
                evidence_text = ' '.join(all_evidence_texts)
                datapoint['evidence_touse'] = evidence_text

            # if args.claim_only:
            #     datapoint['evidence_touse'] = ''

            #  sent1 = '[CONTEXT]: ' + ' [EOT] '.join(example['context'][-2:]) + ' [RESPONSE]: ' + sent1
            primi_idx = datapoint['id']
            text_a = ""
            if 'claim' in datapoint.keys():
                text_a = datapoint['claim']
            text_b =  datapoint['question'] + datapoint['evidence_touse']
            if 'label' in datapoint.keys():
                label = LABELS[datapoint['label']]

            priori = get_priori(obj, text_a, datapoint['question'] ,datapoint['evidence_touse'], T = 1)
            examples.append((InputExample(idx=primi_idx, text_a=text_a, text_b=text_b, label=label, priori=priori)))
        return examples


def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer):
    label_map = {label: i for i, label in enumerate(label_list)}

    features = []

    for (ex_index, example) in enumerate(tqdm(examples, desc="convert to features")):

        label_id = label_map[example.label]

        tokens_a = tokenizer.tokenize(example.text_a)
        tokens_b = tokenizer.tokenize(example.text_b)
        _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)

        tokens = ["<s>"] + tokens_a + ["</s>"]
        segment_ids = [0] * (len(tokens_a) + 2)
        tokens += tokens_b + ["</s>"]
        segment_ids += [1] * (len(tokens_b) + 1)
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_ids)

        padding = [1] * (max_seq_length - len(input_ids))
        input_mask += [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        segment_ids += padding
        #print(len(input_ids))
        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        if ex_index < 1:
            logger.info("*** Example ***")
            logger.info("tokens: %s" % " ".join([str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            logger.info("label: %s (id = %d)" % (example.label, label_id))

        features.append(InputFeatures(input_ids=input_ids,
                                      input_mask=input_mask,
                                      segment_ids=segment_ids,
                                      label_id=label_id,
                                      priori=example.priori))
    return features


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()

def get_priori(obj, response, context, evidence, T = 1):
    score = [0.2,0.2,0.6]
    try:
        res_ctx_score = 0.2*(1 - obj.cal_consine_similarities(response, context))
        res_evi_score = 0.2*(1 - obj.cal_consine_similarities(response, evidence))
    except:
        res_ctx_score = 0
        res_evi_score = 0
    score[0] += res_ctx_score
    score[1] += res_evi_score
    score = softmax(score,T)
    return score


def eval_1(preds, labels):
    TP = ((preds == 1) & (labels == 1)).sum()
    FN = ((preds == 0) & (labels == 0)).sum()
    TN = ((preds == 0) & (labels == 1)).sum()
    FP = ((preds == 1) & (labels == 0)).sum()
    precision = TP / (TP + FP + 0.001)
    recall = TP / (TP + FN + 0.001)
    success = TP + FN
    fail = TN + FP
    acc = success / (success + fail + 0.001)
    return TP, TN, FN, FP, precision, recall, success, fail, acc


def eval_2(mapping):
    success = 0
    fail = 0
    for idx in mapping.keys():
        similarity, prog_label, fact_label, gold_label = mapping[idx]
        if prog_label == fact_label:
            success += 1
        else:
            fail += 1
    acc = success / (success + fail + 0.001)

    return success, fail, acc


def compute_metrics(preds, labels):
    assert len(preds) == len(labels)
    TP, TN, FN, FP, precision, recall, success, fail, acc = eval_1(preds, labels)
    result = {"TP": TP, "TN": TN, "FN": FN, "FP": FP,
              "precision": precision, "recall": recall, "success": success, "fail": fail, "acc": acc}

    return result

def compute_metrics_fn(preds, labels):
    # preds = np.argmax(p.predictions, axis=1)
    assert len(preds) == len(labels)
#     acc = (preds == labels).mean()
    f1 = f1_score(y_true= labels, y_pred=preds, average="macro", labels=np.unique(labels))
    acc = accuracy_score(y_true= labels, y_pred=preds)
    p = precision_score(y_true= labels, y_pred=preds, average="macro", labels=np.unique(labels))
    r = recall_score(y_true= labels, y_pred=preds, average="macro", labels=np.unique(labels))
    return {
        "p": p,
        "acc": acc,
        "macro_f1": f1,
        "macro_recall":r
    }

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def get_dataLoader(args, processor, tokenizer, phase=None):
    dataset_dict = {"train": args.train_set, "dev": args.dev_set, "test": args.test_set}
    label_list, _ = processor.get_labels()

    examples = processor.get_examples(args.data_dir, dataset_dict[phase])
    features = convert_examples_to_features(examples, label_list, args.max_seq_length, tokenizer)

    batch_size = args.train_batch_size if phase == "train" else args.eval_batch_size
    epoch_num = args.num_train_epochs if phase == "train" else 1
    num_optimization_steps = int(len(examples) / batch_size / args.gradient_accumulation_steps) * epoch_num
    logger.info("Examples#: {}, Batch size: {}".format(len(examples), batch_size * args.gradient_accumulation_steps))
    logger.info("Total num of steps#: {}, Total num of epoch#: {}".format(num_optimization_steps, epoch_num))

    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)
    all_priori = torch.tensor([f.priori for f in features], dtype=torch.float)

    all_data = TensorDataset(all_input_ids, all_input_mask, all_label_ids, all_priori)
    if args.do_train_eval:
        sampler = SequentialSampler(all_data)
    else:
        sampler = RandomSampler(all_data) if phase == "train" else SequentialSampler(all_data)
    dataloader = DataLoader(all_data, sampler=sampler, batch_size=batch_size)

    return dataloader, num_optimization_steps, examples


def save_model(model_to_save):
    save_model_dir = os.path.join(args.output_dir, 'saved_model')
    mkdir(save_model_dir)
    output_model_file = os.path.join(save_model_dir, WEIGHTS_NAME)
    torch.save(model_to_save.state_dict(), output_model_file, _use_new_zipfile_serialization=False)

def softmax(input,T=1):
    output = [np.exp(i/T) for i in input]
    output_sum = sum(output)
    final = [i/output_sum for i in output]
    return final


def run_train(device, processor, tokenizer, model, writer, phase="train"):
    logger.info("\n************ Start Training *************")

    tr_dataloader, tr_num_steps, tr_examples = get_dataLoader(args, processor, tokenizer, phase="train")

    model.train()

    loss_fct = torch.nn.KLDivLoss(reduction='batchmean')

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = \
        [{'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
         {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]
    optimizer = BertAdam(optimizer_grouped_parameters,
                                 lr=args.learning_rate,
                                 warmup=args.warmup_proportion,
                                 t_total=tr_num_steps)
    optimizer.zero_grad()

    global_step = 0
    best_acc = 0.0
    n_gpu = torch.cuda.device_count()

    for ep in trange(args.num_train_epochs, desc="Training"):
        for step, batch in tqdm(enumerate(tr_dataloader)):

            batch = tuple(t.to(device) for t in batch)
            input_ids, input_mask, label_ids, priori = batch
            logits, loss, final_out_logits, origin_gates = model(input_ids=input_ids, attention_mask=input_mask, labels=label_ids)
            guide_loss = loss_fct(torch.nn.functional.log_softmax(origin_gates, dim=1), priori)
            loss += args.lmd * guide_loss
            if n_gpu > 1:
                loss = loss.mean()
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            writer.add_scalar('{}/loss'.format(phase), loss.item(), global_step)

            loss.backward()
            del loss

            if (step + 1) % args.gradient_accumulation_steps == 0:  # optimizer
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

            model.eval()
            torch.set_grad_enabled(False)

            if args.do_eval and (((step + 1) % args.gradient_accumulation_steps == 0 and global_step % args.period == 0) or (ep==0 and step==0)):
                model_to_save = model.module if hasattr(model, 'module') else model

                dev_acc, dev_recall = run_eval(device, processor, tokenizer, model, writer, global_step, tensorboard=True,
                                   phase="dev")
                if dev_acc > best_acc:
                    wait_step = 0
                    stop_training = False
                    best_acc = dev_acc
                    logger.info(">> Save model. Best acc: {:.4}. Epoch {}".format(best_acc, ep))
                    save_model(model_to_save)  # save model
                    logger.info(">> Now the best acc is {:.4}\n, recall is {:.4}".format(dev_acc, dev_recall))
                else:
                    wait_step += 1
                    if wait_step >= args.wait_step:
                        stop_training = True
                        break

            model.train()
            torch.set_grad_enabled(True)
        
        if stop_training:
            break

    return global_step


def run_eval(device, processor, tokenizer, model, writer, global_step, tensorboard=False,
             phase=None):
    sys.stdout.flush()
    logger.info("\n************ Start {} *************".format(phase))

    model.eval()

    loss_fct = torch.nn.KLDivLoss(reduction='batchmean')
    cross_entropy = nn.CrossEntropyLoss(reduction='none')

    dataloader, num_steps, examples = get_dataLoader(args, processor, tokenizer, phase=phase)

    eval_loss = 0.0
    eval_guide_loss = 0.0
    num_steps = 0
    preds = []
    preds_0, preds_1, preds_2 = [],[],[]
    all_labels = []
    mapping = []
    for step, batch in enumerate(tqdm(dataloader, desc=phase)):
        batch = tuple(t.to(device) for t in batch)
        input_ids, input_mask, label_ids, priori = batch
        num_steps += 1

        with torch.no_grad():

            logits, tmp_loss, final_out_logits, origin_gates = model(input_ids=input_ids, attention_mask=input_mask, labels=label_ids)
            guide_loss = loss_fct(torch.nn.functional.log_softmax(origin_gates, dim=1), priori)

            eval_loss += tmp_loss.mean().item()
            eval_guide_loss += guide_loss.mean().item()
            logits_sigmoid = final_out_logits
            loss = []
            for l in logits:
                loss.append(cross_entropy(l.squeeze(1), label_ids.view(-1)).view(-1,1))
            if len(loss) == 1:
                loss_mat = loss[0].view(-1,1)
            else:
                loss_mat = torch.cat(loss, dim=1) # bsz * # of experts
            logits_sigmoid_0 = torch.nn.functional.softmax(logits[0].squeeze(1), dim=1)
            logits_sigmoid_1 = torch.nn.functional.softmax(logits[1].squeeze(1), dim=1)
            logits_sigmoid_2 = torch.nn.functional.softmax(logits[2].squeeze(1), dim=1)
            if len(preds) == 0:
                preds.append(logits_sigmoid.detach().cpu().numpy())
                preds_0.append(logits_sigmoid_0.detach().cpu().numpy())
                preds_1.append(logits_sigmoid_1.detach().cpu().numpy())
                preds_2.append(logits_sigmoid_2.detach().cpu().numpy())
            else:
                preds[0] = np.append(preds[0], logits_sigmoid.detach().cpu().numpy(), axis=0)
                preds_0[0] = np.append(preds_0[0], logits_sigmoid_0.detach().cpu().numpy(), axis=0)
                preds_1[0] = np.append(preds_1[0], logits_sigmoid_1.detach().cpu().numpy(), axis=0)
                preds_2[0] = np.append(preds_2[0], logits_sigmoid_2.detach().cpu().numpy(), axis=0)

            labels = label_ids.detach().cpu().numpy().tolist()

            start = step * args.eval_batch_size if not args.do_train_eval else step * args.train_batch_size
            end = start + len(labels)
            batch_range = list(range(start, end))

            idx = [examples[i].idx for i in batch_range]
            labels = label_ids.detach().cpu().numpy().tolist()
            all_labels.extend(labels)
            loss_mat_cpu = loss_mat.detach().cpu().numpy().tolist()
            for i, t_name in enumerate(idx):
                mapping.append([str(loss_mat_cpu[i][0]), str(loss_mat_cpu[i][1]), str(loss_mat_cpu[i][2])])

    result = {}
    result['acc'] = 0
    eval_loss /= num_steps
    eval_guide_loss /= num_steps
    preds = np.argmax(preds[0], axis=1)
    preds_0 = np.argmax(preds_0[0], axis=1)
    preds_1 = np.argmax(preds_1[0], axis=1)
    preds_2 = np.argmax(preds_2[0], axis=1)
    pred_for_test, label_for_test = [] ,[]
    for pred, label in zip(preds,all_labels):
        pred_for_test.append(pred)
        label_for_test.append(label)
            
    result = compute_metrics(np.asarray(pred_for_test), np.asarray(label_for_test))
    result_0 = compute_metrics(np.asarray(preds_0), np.asarray(all_labels))
    result_1 = compute_metrics(np.asarray(preds_1), np.asarray(all_labels))
    result_2 = compute_metrics(np.asarray(preds_2), np.asarray(all_labels))
    result['acc_0'] = result_0['acc']
    result['acc_1'] = result_1['acc']
    result['acc_2'] = result_2['acc']
    result['{}_loss'.format(phase)] = eval_loss
    result['{}_guide_loss'.format(phase)] = eval_guide_loss
    result['global_step'] = global_step
    logger.info(result)
    if tensorboard and writer is not None:
        for key in sorted(result.keys()):
            writer.add_scalar('{}/{}'.format(phase, key), result[key], global_step)
    json.dump(mapping, open('./{}_moe_roberta_lmd_0.1.json'.format(phase),'w', encoding='utf8'))
        
    model.train()
    return result['acc'], result['recall']


def main():
    mkdir(args.output_dir)

    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps
    writer = SummaryWriter(os.path.join(args.output_dir, 'events'))
    cache_dir = args.cache_dir

    device = torch.device("cuda")
    n_gpu = torch.cuda.device_count()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    save_code_log_path = args.output_dir

    logging.basicConfig(format='%(message)s', datefmt='%m/%d/%Y %H:%M', level=logging.INFO,
                        handlers=[logging.FileHandler("{0}/{1}.log".format(save_code_log_path, 'output')),
                                  logging.StreamHandler()])
    logger.info(args)
    logger.info("Command is: %s" % ' '.join(sys.argv))
    logger.info("Device: {}, n_GPU: {}".format(device, n_gpu))
    logger.info("Datasets are loaded from {}\nOutputs will be saved to {}\n".format(args.data_dir, args.output_dir))

    processor = DataProcessor()

    tokenizer = RobertaTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)

    load_dir = args.load_dir if args.load_dir else args.bert_model
    logger.info('Model is loaded from %s' % load_dir)
    label_list = processor.get_labels()
    config = RobertaConfig.from_json_file(os.path.join(args.bert_model,'config.json'))
    model = RobertaMoEForSequenceClassification(config, num_public_layers=12, num_experts=3,num_labels=2, num_gate_layer=2)
    model.load_roberta(args.bert_model)
    if args.load_dir:
        model.load_state_dict(torch.load(load_dir+'/pytorch_model.bin'))
        print('parameters loaded successfully.')
    model.to(device)

    if n_gpu > 1:
        model = torch.nn.DataParallel(model,device_ids=[0, 1])

    if args.do_train:
        run_train(device, processor, tokenizer, model, writer, phase="train")

    if args.do_eval:
        run_eval(device, processor, tokenizer, model, writer, global_step=0, tensorboard=False,
                 phase="dev")
        run_eval(device, processor, tokenizer, model, writer, global_step=0, tensorboard=False,
                 phase="test")

    if args.do_test:
        run_eval(device, processor, tokenizer, model, writer, global_step=0, tensorboard=False,
                 phase="test")

    if args.do_train_eval:
        run_eval(device, processor, tokenizer, model, writer, global_step=0, tensorboard=False,
                 phase="train")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--do_train", action='store_true')
    parser.add_argument("--do_eval", action='store_true')
    parser.add_argument("--do_test", action='store_true')
    parser.add_argument("--do_train_eval", action='store_true')
    parser.add_argument("--add_unk", action='store_true')
    parser.add_argument("--load_dir", help="load model checkpoints")
    parser.add_argument("--data_dir", help="path to data", default='../data/faviq/gold/faviq_a_set')
    parser.add_argument("--train_set", default="gold_train")
    parser.add_argument("--dev_set", default="gold_dev")
    parser.add_argument("--test_set", default="gold_test")
    parser.add_argument("--output_dir", default='./outputs_faviqlmoe_gold')
    parser.add_argument("--cache_dir", default="./roberta", type=str, help="store downloaded pre-trained models")
    parser.add_argument('--period', type=int, default=1000)
    parser.add_argument("--bert_model", default="../roberta_large", type=str)
    parser.add_argument("--do_lower_case", default=True, help="Set this flag if you are using an uncased model.")
    parser.add_argument("--task_name", default="LPA", type=str)
    parser.add_argument('--response_tag', type=str, help='tag', default='response')
    parser.add_argument("--max_seq_length", default=512)
    parser.add_argument("--train_batch_size", default=16)
    parser.add_argument("--eval_batch_size", default=16)
    parser.add_argument('--debug_mode', action='store_true')
    parser.add_argument("--learning_rate", default=2e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs", default=10000)
    parser.add_argument('--wait_step', type=int, default=10)
    parser.add_argument("--lmd",default=0.1, type=float, help="the ratio of guide loss in the ttl loss")
    parser.add_argument("--warmup_proportion", default=0.3, type=float, help="0.1 = 10%% of training.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=2)
    parser.add_argument('--seed', type=int, default=42, help="random seed")

    args = parser.parse_args()
    main()
    
    # TRANSFORMERS_OFFILNE=1 python run_faviqmoe_gold.py --do_train --do_eval
    # TRANSFORMERS_OFFILNE=1 CUDA_VISIBLE_DEVICES=3 nohup python -u run_faviqmoe_gold.py --do_train --do_eval --load_dir ./outputs_faviqlmoe_gold/saved_model > faviq_log_gold.file 2>&1 &
    # python run_dialmoe_gold.py --do_eval --do_test --load_dir [modeldir]