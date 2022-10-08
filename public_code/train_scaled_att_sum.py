import argparse
import csv
import os
import sys
import json
from attr import attr

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score
from transformers import AdamW, AutoModel, AutoTokenizer, AutoConfig
from captum.attr import Saliency, InputXGradient, IntegratedGradients,GuidedBackprop,Occlusion, DeepLift
from tqdm import tqdm
from captum._utils.common import _run_forward
from typing import Any, Callable, Union, Tuple
from torch import Tensor

csv.field_size_limit(sys.maxsize)


parser = argparse.ArgumentParser()
parser.add_argument('--device', type=int, default=0, help='CUDA device')
parser.add_argument('--model', type=str, help='pre-trained model (bert-base-uncased, roberta-base)')
parser.add_argument('--task', type=str, help='task name (SNLI, MNLI, QQP, TwitterPPDB, SWAG, HellaSWAG)')
parser.add_argument('--max_seq_length', type=int, default=256, help='max sequence length')
parser.add_argument('--ckpt_path', type=str, help='model checkpoint path')
parser.add_argument('--output_path', type=str, help='model output path')
parser.add_argument('--train_path', type=str, help='train dataset path')
parser.add_argument('--dev_path', type=str, help='dev dataset path')
parser.add_argument('--test_path', type=str, help='test dataset path')
parser.add_argument('--epochs', type=int, default=3, help='number of epochs')
parser.add_argument('--seed', type=int, default=42, help='random seed')
parser.add_argument('--batch_size', type=int, default=8, help='batch size')
parser.add_argument('--learning_rate', type=float, default=2e-5, help='learning rate')
parser.add_argument('--weight_decay', type=float, default=0., help='weight decay')
parser.add_argument('--label_smoothing', type=float, default=-1., help='label smoothing \\alpha')
parser.add_argument('--max_grad_norm', type=float, default=1., help='gradient clip')
parser.add_argument('--lamb', type=float, default=1.0, help='reg-curr lambda')
parser.add_argument('--do_train', action='store_true', default=False, help='enable training')
parser.add_argument('--do_expl_selective_loss', action='store_true', default=False, help='enable explanation selective tuning loss')
parser.add_argument('--do_evaluate', action='store_true', default=False, help='enable evaluation')
args = parser.parse_args()
print(args)
seed = args.seed
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.manual_seed(int(seed))
np.random.seed(int(seed))
torch.cuda.manual_seed(int(seed))

assert args.task in ('SNLI', 'MNLI', 'QQP', 'TwitterPPDB', 'SWAG', 'HellaSWAG')
assert args.model in ('bert-base-uncased', 'roberta-base')


if args.task in ('SNLI', 'MNLI'):
    n_classes = 3
elif args.task in ('QQP', 'TwitterPPDB'):
    n_classes = 2
elif args.task in ('SWAG', 'HellaSWAG'):
    n_classes = 1


def cuda(tensor):
    """Places tensor on CUDA device."""

    return tensor.to(args.device)


def load(dataset, batch_size, shuffle):
    """Creates data loader with dataset and iterator options."""

    return DataLoader(dataset, batch_size, shuffle=shuffle)


def adamw_params(model):
    """Prepares pre-trained model parameters for AdamW optimizer."""

    no_decay = ['bias', 'LayerNorm.weight']
    params = [
        {
            'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            'weight_decay': args.weight_decay
        },
        {
            'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            'weight_decay': 0.0
        },
    ]
    return params


def encode_pair_inputs(sentence1, sentence2):
    """
    Encodes pair inputs for pre-trained models using the template
    [CLS] sentence1 [SEP] sentence2 [SEP]. Used for SNLI, MNLI, QQP, and TwitterPPDB.
    Returns input_ids, segment_ids, and attention_mask.
    """

    inputs = tokenizer.encode_plus(
        sentence1, sentence2, add_special_tokens=True, max_length=args.max_seq_length
    )
    input_ids = inputs['input_ids']
    segment_ids = inputs['token_type_ids']
    attention_mask = inputs['attention_mask']
    padding_length = args.max_seq_length - len(input_ids)
    input_ids += [tokenizer.pad_token_id] * padding_length
    segment_ids += [0] * padding_length
    attention_mask += [0] * padding_length
    for input_elem in (input_ids, segment_ids, attention_mask):
        assert len(input_elem) == args.max_seq_length
    return (
        cuda(torch.tensor(input_ids)).long(),
        cuda(torch.tensor(segment_ids)).long(),
        cuda(torch.tensor(attention_mask)).long(),
    )


def encode_mc_inputs(context, start_ending, endings):
    """
    Encodes multiple choice inputs for pre-trained models using the template
    [CLS] context [SEP] ending_i [SEP] where 0 <= i < len(endings). Used for
    SWAG and HellaSWAG. Returns input_ids, segment_ids, and attention_masks.
    """

    context_tokens = tokenizer.tokenize(context)
    start_ending_tokens = tokenizer.tokenize(start_ending)
    all_input_ids = []
    all_segment_ids = []
    all_attention_masks = []
    for ending in endings:
        ending_tokens = start_ending_tokens + tokenizer.tokenize(ending)
        inputs = tokenizer.encode_plus(
            context_tokens, ending_tokens, add_special_tokens=True, max_length=args.max_seq_length
        )
        # import ipdb; ipdb.set_trace()

        input_ids = inputs['input_ids']
        segment_ids = inputs['token_type_ids']
        attention_mask = inputs['attention_mask']
        padding_length = args.max_seq_length - len(input_ids)
        input_ids += [tokenizer.pad_token_id] * padding_length
        segment_ids += [0] * padding_length
        attention_mask += [0] * padding_length
        for input_elem in (input_ids, segment_ids, attention_mask):
            assert len(input_elem) == args.max_seq_length
        all_input_ids.append(input_ids)
        all_segment_ids.append(segment_ids)
        all_attention_masks.append(attention_mask)
    return (
        cuda(torch.tensor(all_input_ids)).long(),
        cuda(torch.tensor(all_segment_ids)).long(),
        cuda(torch.tensor(all_attention_masks)).long(),
    )


def encode_label(label):
    """Wraps label in tensor."""

    return cuda(torch.tensor(label)).long()


class SNLIProcessor:
    """Data loader for SNLI."""

    def __init__(self):
        self.label_map = {'entailment': 0, 'contradiction': 1, 'neutral': 2}

    def valid_inputs(self, sentence1, sentence2, label):
        return len(sentence1) > 0 and len(sentence2) > 0 and label in self.label_map

    def load_samples(self, path):
        samples = []
        with open(path, newline='') as f:
            reader = csv.reader(f, delimiter='\t')
            next(reader)  # skip header
            desc = f'loading \'{path}\''
            for row in tqdm(reader, desc=desc):
                try:
                    sentence1 = row[7]
                    sentence2 = row[8]
                    label = row[-1]
                    if self.valid_inputs(sentence1, sentence2, label):
                        label = self.label_map[label]
                        samples.append((sentence1, sentence2, label))
                except:
                    pass
        return samples


class MNLIProcessor(SNLIProcessor):
    """Data loader for MNLI."""

    def load_samples(self, path):
        samples = []
        with open(path, newline='') as f:
            reader = csv.reader(f, delimiter='\t')
            next(reader)  # skip header
            desc = f'loading \'{path}\''
            for row in tqdm(reader, desc=desc):
                try:
                    sentence1 = row[8]
                    sentence2 = row[9]
                    label = row[-1]
                    if self.valid_inputs(sentence1, sentence2, label):
                        label = self.label_map[label]
                        samples.append((sentence1, sentence2, label))
                except:
                    pass
        return samples


class QQPProcessor:
    """Data loader for QQP."""

    def valid_inputs(self, sentence1, sentence2, label):
        return len(sentence1) > 0 and len(sentence2) > 0 and label in ('0', '1')

    def load_samples(self, path):
        samples = []
        with open(path, newline='') as f:
            reader = csv.reader(f, delimiter='\t')
            next(reader)  # skip header
            desc = f'loading \'{path}\''
            for row in tqdm(reader, desc=desc):
                try:
                    sentence1 = row[3]
                    sentence2 = row[4]
                    label = row[5]
                    if self.valid_inputs(sentence1, sentence2, label):
                        label = int(label)
                        samples.append((sentence1, sentence2, label))
                except:
                    pass
        return samples


class TwitterPPDBProcessor:
    """Data loader for TwittrPPDB."""

    def valid_inputs(self, sentence1, sentence2, label):
        return len(sentence1) > 0 and len(sentence2) > 0 and label != 3 
    
    def load_samples(self, path):
        samples = []
        with open(path, newline='') as f:
            reader = csv.reader(f, delimiter='\t')
            desc = f'loading \'{path}\''
            for row in tqdm(reader, desc=desc):
                try:
                    sentence1 = row[0]
                    sentence2 = row[1]
                    label = eval(row[2])[0]
                    if self.valid_inputs(sentence1, sentence2, label):
                        label = 0 if label < 3 else 1
                        samples.append((sentence1, sentence2, label))
                except:
                    pass
        return samples


class SWAGProcessor:
    """Data loader for SWAG."""

    def load_samples(self, path):
        samples = []
        with open(path, newline='') as f:
            reader = csv.reader(f, delimiter=',')
            desc = f'loading \'{path}\''
            for row in tqdm(reader, desc=desc):
                try:
                    context = row[4]
                    start_ending = row[5]
                    endings = row[7:11]
                    label = int(row[-1])
                    samples.append((context, start_ending, endings, label))
                except:
                    pass
        return samples


class HellaSWAGProcessor:
    """Data loader for HellaSWAG."""

    def load_samples(self, path):
        samples = []
        with open(path) as f:
            desc = f'loading \'{path}\''
            for line in f:
                try:
                    line = line.rstrip()
                    input_dict = json.loads(line)
                    context = input_dict['ctx_a']
                    start_ending = input_dict['ctx_b']
                    endings = input_dict['endings']
                    label = input_dict['label']
                    samples.append((context, start_ending, endings, label))
                except:
                    pass
        return samples


def select_processor():
    """Selects data processor using task name."""

    return globals()[f'{args.task}Processor']()


class TextDataset(Dataset):
    """
    Task-specific dataset wrapper. Used for storing, retrieving, encoding,
    caching, and batching samples.
    """

    def __init__(self, path, processor):
        self.samples = processor.load_samples(path)
        self.cache = {}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        res = self.cache.get(i, None)
        if res is None:
            sample = self.samples[i]
            if args.task in ('SNLI', 'MNLI', 'QQP', 'MRPC', 'TwitterPPDB'):
                sentence1, sentence2, label = sample
                input_ids, segment_ids, attention_mask = encode_pair_inputs(
                    sentence1, sentence2
                )
                packed_inputs = (sentence1, sentence2)
            elif args.task in ('SWAG', 'HellaSWAG'):
                context, ending_start, endings, label = sample
                input_ids, segment_ids, attention_mask = encode_mc_inputs(
                    context, ending_start, endings
                )
            label_id = encode_label(label)
            res = ((input_ids, segment_ids, attention_mask), label_id)
            self.cache[i] = res
        return res

SMALL = 1e-08


class VMASK(nn.Module):
    def __init__(self, args):
        super(VMASK, self).__init__()

        self.device = args.device
        self.mask_hidden_dim = 100
        self.activations = {'tanh': torch.tanh, 'sigmoid': torch.sigmoid, 'relu': torch.relu,
                            'leaky_relu': F.leaky_relu}
        self.activation = self.activations['tanh']
        self.embed_dim = 768
        self.linear_layer = nn.Linear(self.embed_dim, self.mask_hidden_dim)
        self.hidden2p = nn.Linear(self.mask_hidden_dim, 2)

    def forward_sent_batch(self, embeds):
        temps = self.activation(self.linear_layer(embeds))
        p = self.hidden2p(temps)  # bsz, seqlen, dim
        return p

    def forward(self, x, p, flag):
        if flag == 'train':
            r = F.gumbel_softmax(p, hard=True, dim=2)[:, :, 1:2]
            x_prime = r * x
            return x_prime
        else:
            probs = F.softmax(p, dim=2)[:, :, 1:2]  # select the probs of being 1
            x_prime = probs * x
            return x_prime

    def get_statistics_batch(self, embeds):
        p = self.forward_sent_batch(embeds)
        return p

class Model(nn.Module):
    """Pre-trained model for classification."""

    def __init__(self):
        super().__init__()
        self.config = AutoConfig.from_pretrained(args.model)
        self.config.output_attentions = True
        self.model = AutoModel.from_pretrained(args.model, config=self.config)
        # self.model.encoder.output_hidden_states = True
        self.classifier = nn.Linear(768, n_classes)
        # self.maskmodel = VMASK(args)
        self.attr_classifier = nn.Linear(args.max_seq_length, 1)
    
    
    def embed_forward(self, embed, n_choices=-1):
        e = self.model(inputs_embeds=embed)
        
        logits = self.classifier(e[1])

        probs = torch.softmax(logits, dim = -1)
        if args.task in ('SWAG', 'HellaSWAG'):
            probs = probs.view(-1, n_choices)
        return probs

    def forward(self, input_ids, segment_ids, attention_mask, is_training=True):
        # On SWAG and HellaSWAG, collapse the batch size and
        # choice size dimension to process everything at once
        if args.task in ('SWAG', 'HellaSWAG'):
            n_choices = input_ids.size(1)
            input_ids = input_ids.view(-1, input_ids.size(-1))
            segment_ids = segment_ids.view(-1, segment_ids.size(-1))
            attention_mask = attention_mask.view(-1, attention_mask.size(-1))
        transformer_params = {
            'input_ids': input_ids,
            'token_type_ids': (
                segment_ids if args.model == 'bert-base-uncased' else None
            ),
            'attention_mask': attention_mask,
        }
        transformer_outputs = self.model(**transformer_params)
        self.weights_or = transformer_outputs[2][-1]
        if is_training:
            self.weights_or.retain_grad()
        if args.task in ('SWAG', 'HellaSWAG'):
            pooled_output = transformer_outputs[1]
            logits = self.classifier(pooled_output)
            logits = logits.view(-1, n_choices)
        else:
            cls_output = transformer_outputs[0][:, 0]
            logits = self.classifier(cls_output)
        return logits


class LabelSmoothingLoss(nn.Module):
    """
    Label smoothing loss. Adapted from https://bit.ly/2T6kfz7. If 0 < smoothing < 1,
    this smoothes the standard cross-entropy loss.
    """

    def __init__(self, smoothing):
        super().__init__()
        _n_classes = n_classes if args.task not in ('SWAG', 'HellaSWAG') else 4
        self.confidence = 1. - smoothing
        smoothing_value = smoothing / (_n_classes - 1)
        one_hot = cuda(torch.full((_n_classes,), smoothing_value))
        self.register_buffer('one_hot', one_hot.unsqueeze(0))

    def forward(self, output, target):
        model_prob = self.one_hot.repeat(target.size(0), 1)
        model_prob.scatter_(1, target.unsqueeze(1), self.confidence)
        return F.kl_div(F.log_softmax(output, 1), model_prob, reduction='sum')

def compute_gradients(
    forward_fn: Callable,
    inputs: Union[Tensor, Tuple[Tensor, ...]],
    target_ind = None,
    additional_forward_args: Any = None,
) -> Tuple[Tensor, ...]:
    with torch.autograd.set_grad_enabled(True):
        # runs forward pass
        outputs = _run_forward(forward_fn, inputs, target_ind, additional_forward_args)
        assert outputs[0].numel() == 1, (
            "Target not provided when necessary, cannot"
            " take gradient with respect to multiple outputs."
        )
        grads = torch.autograd.grad(torch.unbind(outputs), inputs, create_graph=True)
    return grads

def train(dataset):
    """Fine-tunes pre-trained model on training set."""
    print("lambda: ", args.lamb)
    model.train()
    saliency = Saliency(model.embed_forward)
    saliency.gradient_func = compute_gradients
    train_loss = torch.tensor(0.).to("cuda")
    reg_loss = torch.tensor(0.).to("cuda")
    train_loader = tqdm(load(dataset, args.batch_size, True))
    optimizer = AdamW(adamw_params(model), lr=args.learning_rate, eps=1e-8)
    for i, (inputs, label) in enumerate(train_loader, 1):
        optimizer.zero_grad()
        logits = model(*inputs)
        loss = criterion(logits, label)
        loss.backward(retain_graph=True)
        if args.task in ('SWAG', 'HellaSWAG'):
            n_choices = inputs[0].size(1)
            att = model.weights_or[:, :, 0, :].mean(1)
            att_grad = model.weights_or.grad[:,:,0,:].mean(1)
            attributions = att*att_grad
            attributions = attributions.view(-1, n_choices, attributions.size(-1))
            attributions = torch.abs(attributions)
            attributions = torch.masked_fill(attributions, ~inputs[2].bool(), float("-inf"))
            attributions = torch.softmax(attributions, -1)
            # attributions = torch.mean(attributions, 1)
            attributions = attributions[:, label, :]
            
        else:
            att = model.weights_or[:, :, 0, :].mean(1)
            att_grad = model.weights_or.grad[:,:,0,:].mean(1)

            attributions = att*att_grad
            
            attributions = torch.abs(attributions)
            attributions = torch.masked_fill(attributions, ~inputs[2].bool(), float("-inf"))
            attributions = torch.softmax(attributions, -1)          
        
        
        if args.do_expl_selective_loss:
            confidence, prediction = torch.softmax(logits, dim=1).max(dim=1)
            correctness = (prediction == label)
            # squeezed_attrs = torch.norm(attributions, dim=-1, p=2) #* torch.sqrt(lengths) # * confidence
            squeezed_attrs = torch.norm(attributions, dim=-1) * confidence
            
            correct_confidence = torch.masked_select(squeezed_attrs, correctness)
            wrong_confidence = torch.masked_select(squeezed_attrs, ~correctness)
            
            regularizer = torch.tensor(0.).to("cuda")
            for cc in correct_confidence:
                for wc in wrong_confidence:
                    regularizer += torch.clamp(wc-cc, min=0) ** 2

            r_loss = args.lamb * regularizer 
            try:
                r_loss.backward()
            except:
                pass
           
        train_loss += loss.item()
        
        reg_loss += cuda(torch.tensor(args.lamb * regularizer)).item()
        train_loader.set_description(f'train loss = {(train_loss / i):.6f}'+f' reg loss = {(reg_loss / i):.6f}')
        
        if args.max_grad_norm > 0.:
            nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        optimizer.step()
    return train_loss / len(train_loader)


def evaluate(dataset):
    """Evaluates pre-trained model on development set."""

    model.eval()
    eval_loss = 0.
    eval_loader = tqdm(load(dataset, args.batch_size, False))
    for i, (inputs, label) in enumerate(eval_loader, 1):
        with torch.no_grad():
            loss = criterion(model(*inputs, is_training=False), label)
        eval_loss += loss.item()
        eval_loader.set_description(f'eval loss = {(eval_loss / i):.6f}')
    return eval_loss / len(eval_loader)


model = cuda(Model())
processor = select_processor()
tokenizer = AutoTokenizer.from_pretrained(args.model)

if args.label_smoothing == -1:
    criterion = nn.CrossEntropyLoss()
else:
    criterion = LabelSmoothingLoss(args.label_smoothing)

if args.train_path:
    train_dataset = TextDataset(args.train_path, processor)
    print(f'train samples = {len(train_dataset)}')
if args.dev_path:
    dev_dataset = TextDataset(args.dev_path, processor)
    print(f'dev samples = {len(dev_dataset)}')
if args.test_path:
    test_dataset = TextDataset(args.test_path, processor)
    print(f'test samples = {len(test_dataset)}')

if args.do_train:
    print()
    print('*** training ***')
    best_loss = float('inf')
    for epoch in range(1, args.epochs + 1):
        train_loss = train(train_dataset)
        eval_loss = evaluate(dev_dataset)
        if eval_loss < best_loss:
            best_loss = eval_loss
            torch.save(model.state_dict(), args.ckpt_path)
        print(
            f'epoch = {epoch} | '
            f'train loss = {train_loss:.6f} | '
            f'eval loss = {eval_loss:.6f}'
        )

if args.do_evaluate:
    if not os.path.exists(args.ckpt_path):
        raise RuntimeError(f'\'{args.ckpt_path}\' does not exist')
    
    print()
    print('*** evaluating ***')

    output_dicts = []
    model.load_state_dict(torch.load(args.ckpt_path))
    model.eval()
    test_loader = tqdm(load(test_dataset, args.batch_size, False))

    for i, (inputs, label) in enumerate(test_loader):
        with torch.no_grad():
            logits = model(*inputs, is_training=False)
            for j in range(logits.size(0)):
                probs = F.softmax(logits[j], -1)
                output_dict = {
                    'index': args.batch_size * i + j,
                    'true': label[j].item(),
                    'pred': logits[j].argmax().item(),
                    'conf': probs.max().item(),
                    'logits': logits[j].cpu().numpy().tolist(),
                    'probs': probs.cpu().numpy().tolist(),
                }
                output_dicts.append(output_dict)

    print(f'writing outputs to \'{args.output_path}\'')

    with open(args.output_path, 'w+') as f:
        for i, output_dict in enumerate(output_dicts):
            output_dict_str = json.dumps(output_dict)
            f.write(f'{output_dict_str}\n')

    y_true = [output_dict['true'] for output_dict in output_dicts]
    y_pred = [output_dict['pred'] for output_dict in output_dicts]
    y_conf = [output_dict['conf'] for output_dict in output_dicts]

    accuracy = accuracy_score(y_true, y_pred) * 100.
    f1 = f1_score(y_true, y_pred, average='macro') * 100.
    confidence = np.mean(y_conf) * 100.

    results_dict = {
        'accuracy': accuracy_score(y_true, y_pred) * 100.,
        'macro-F1': f1_score(y_true, y_pred, average='macro') * 100.,
        'confidence': np.mean(y_conf) * 100.,
    }
    for k, v in results_dict.items():
        print(f'{k} = {v}')
