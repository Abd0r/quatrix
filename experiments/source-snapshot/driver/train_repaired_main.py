#!/usr/bin/env python3
"""Fixed-token-budget repaired training. Validation only, no test tuning.

New recipe: independent data RNG, stable shared-parameter initialization,
length masking, valid-target-normalized micro-batch accumulation, AdamW.
Validation only; never loads or scores test data. No old output modifications.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import sys
import time
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from evaluation import collate_ids,evaluate


def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def initialize(model,seed):
    """Shared named tensors get identical values, independent of module allocation."""
    with torch.no_grad():
        for name,param in model.named_parameters():
            key=name.replace('.compass.state_proj.','.routing_query.').replace('.attn.q_proj.','.routing_query.')
            key=key.replace('.compass.action_proj.','.routing_key.').replace('.attn.k_proj.','.routing_key.')
            gen=torch.Generator(device='cpu').manual_seed(int.from_bytes(hashlib.sha256(f'{seed}:{key}'.encode()).digest()[:8],'little')%(2**63-1))
            if name.endswith('bias'):param.zero_()
            elif param.ndim==1:param.fill_(1)
            else:param.copy_(torch.randn(param.shape,generator=gen)*.02)


def sum_content_forward(layer,x,causal=True):
    if layer.num_heads!=1:raise ValueError('Sum control implemented for single head only')
    s=layer.state_proj(x);k=layer.action_proj(x)
    scores=s@k.transpose(-2,-1)/layer.scale
    if causal:scores=scores.masked_fill(layer._get_causal_mask(x.size(1),x.device)==0,torch.finfo(scores.dtype).min)
    weights=layer.drop(scores.softmax(-1))
    return layer.out_proj(weights@layer.content_proj((s+k)/math.sqrt(2)))


def limit_targets(labels,budget):
    labels=labels.clone()
    valid=labels[:,1:]!=-100
    if budget<1 or budget>int(valid.sum()):raise ValueError('Invalid target budget')
    order=valid.reshape(-1).long().cumsum(0).reshape(valid.shape)
    labels[:,1:].masked_fill_(~(valid & (order<=budget)),-100)
    assert int((labels[:,1:]!=-100).sum())==budget
    return labels


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True);p.add_argument('--train',type=Path,required=True)
    p.add_argument('--validation-data',type=Path,required=True);p.add_argument('--validation-manifest',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--variant',choices=['rankmatched','savo','savo-sum'],required=True)
    p.add_argument('--target-tokens',type=int,default=20000000)
    p.add_argument('--seed',type=int,choices=[17,29,43,71],default=17)
    p.add_argument('--effective-batch',type=int,default=16);p.add_argument('--micro-batch',type=int,default=2)
    p.add_argument('--lr',type=float,default=3e-4);a=p.parse_args()
    if a.output.exists():p.error('Refusing to overwrite training run')
    if not 1<=a.target_tokens<=20000000:p.error('Declared main-run budget capped at 20M valid targets')
    if a.micro_batch<1 or a.effective_batch<1 or a.effective_batch%a.micro_batch:p.error('Invalid micro-batch configuration')
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():p.error('CUDA BF16-capable device required')
    if torch.cuda.mem_get_info()[0]<3*1024**3:p.error('GPU currently lacks 3 GiB free; do not disrupt other jobs')
    torch.set_num_threads(2);random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed);torch.cuda.manual_seed_all(a.seed)
    sys.path.insert(0,str(a.source.resolve()))
    from quatrix.config import QuatrixConfig
    from quatrix.model import QuatrixLM,QCompass
    from quatrix.transformer_lm import TransformerLM
    class SumCompass(QCompass):
        def forward(self,x,causal=True):
            return sum_content_forward(self,x,causal)
    cfg=QuatrixConfig(vocab_size=50257,hidden_size=384,num_layers=10,q_rank=48,max_seq_len=1024,
                      dropout=.1,q_value_content=a.variant!='rankmatched')
    model=TransformerLM(cfg,qk_rank=48) if a.variant=='rankmatched' else QuatrixLM(cfg)
    if a.variant=='savo-sum':
        for block in model.blocks:block.compass=SumCompass(384,48,q_value_content=True,dropout=.1)
    initialize(model,a.seed);model.to('cuda')
    optimizer=torch.optim.AdamW(model.parameters(),lr=a.lr,betas=(.9,.95),weight_decay=.01)
    tok=AutoTokenizer.from_pretrained('openai-community/gpt2',revision='607a30d783dfa663caf39e06633721c8d4cfcd7e',local_files_only=True)
    data_manifest=json.loads((a.train.parent/'split-manifest.json').read_text())
    if sha(a.train)!=data_manifest['output_hashes'][a.train.name]:p.error('Training split hash mismatch')
    with a.train.open() as f:texts=[json.loads(line)['text'] for line in f]
    vm=json.loads(a.validation_manifest.read_text())
    if sha(a.validation_data)!=vm['data_sha256']:p.error('Validation hash mismatch')
    val=[json.loads(line)['input_ids'] for line in a.validation_data.read_text().splitlines()]
    sampler=np.random.default_rng(a.seed+100000);sampling_digest=hashlib.sha256();first_indices=[]
    a.output.mkdir(parents=True)
    metadata={'kind':'Repaired fixed-token-budget experiment; not evidence of convergence','variant':a.variant,'seed':a.seed,
        'data_seed':a.seed+100000,'target_tokens':a.target_tokens,'effective_batch':a.effective_batch,'micro_batch':a.micro_batch,
        'optimizer':'AdamW, betas=(0.9,0.95), weight_decay=0.01, constant LR','lr':a.lr,
        'parameter_init':'Per-name SHA256-derived normal std=.02; common query/key/embedding/FFN tensors matched',
        'content_control':'SAVO product versus (state+action)/sqrt(2); parameter count matched, activation scales are not fully matched',
        'train_sha256':sha(a.train),'split_manifest_sha256':sha(a.train.parent/'split-manifest.json'),
        'source_hashes':{n:sha(a.source/'quatrix'/n) for n in ['model.py','transformer_lm.py','config.py']},
        'driver_sha256':sha(Path(__file__)),'evaluation_sha256':sha(Path(__file__).with_name('evaluation.py')),
        'torch':torch.__version__,'numpy':np.__version__,'cuda':torch.version.cuda,'device':torch.cuda.get_device_name(),
        'parameters':sum(t.numel() for t in model.parameters()),'config':cfg.__dict__,'validation_manifest':vm}
    (a.output/'manifest.json').write_text(json.dumps(metadata,indent=2)+'\n')
    torch.cuda.reset_peak_memory_stats();start=time.perf_counter();total_targets=0
    log=[];step=0;checkpoints=[];next_check=min(5000000,a.target_tokens)
    with (a.output/'training.jsonl').open('x') as live_log:
        while total_targets<a.target_tokens:
            indices=sampler.integers(0,len(texts),size=a.effective_batch)
            sampling_digest.update(indices.astype('<i8').tobytes())
            if len(first_indices)<64:first_indices.extend(map(int,indices))
            sequences=[[tok.bos_token_id]+tok.encode(texts[int(i)])[:1022]+[tok.eos_token_id] for i in indices]
            targets=min(sum(len(s)-1 for s in sequences),a.target_tokens-total_targets)
            optimizer.zero_grad(set_to_none=True);step_nll=0.0;left=targets
            for off in range(0,len(sequences),a.micro_batch):
                if left==0:break
                ids,labels=collate_ids(sequences[off:off+a.micro_batch])
                grant=min(int((labels[:,1:]!=-100).sum()),left)
                labels=limit_targets(labels,grant)
                ids=ids.cuda();labels=labels.cuda()
                with torch.autocast('cuda',dtype=torch.bfloat16):logits=model(ids)['logits']
                loss_sum=F.cross_entropy(logits[:,:-1].float().reshape(-1,50257),labels[:,1:].reshape(-1),ignore_index=-100,reduction='sum')
                if not torch.isfinite(loss_sum):raise ValueError('Non-finite loss; abort rather than skip batch')
                step_nll+=float(loss_sum.detach());(loss_sum/targets).backward();left-=grant
            assert left==0
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.0,error_if_nonfinite=True)
            optimizer.step();step+=1;total_targets+=targets
            record={'step':step,'n_targets':targets,'total_targets':total_targets,'mean_train_nll':step_nll/targets,'grad_norm':float(norm)}
            log.append(record);live_log.write(json.dumps(record)+'\n');live_log.flush()
            if step%100==0:print(json.dumps(record),flush=True)
            if total_targets>=next_check:
                model.eval();validation=evaluate(model,val,batch_size=2,device='cuda');model.train()
                checkpoint=a.output/f'tokens-{total_targets}.pt'
                torch.save({'model_state_dict':{k:v.detach().cpu() for k,v in model.state_dict().items()},
                    'optimizer_state_dict':optimizer.state_dict(),'global_step':step,'variant':a.variant,
                    'supervised_targets':total_targets,'sampler_state':sampler.bit_generator.state,
                    'torch_cpu_rng':torch.get_rng_state(),'torch_cuda_rng':torch.cuda.get_rng_state(),
                    'training_manifest_sha256':sha(a.output/'manifest.json')},checkpoint)
                entry={'supervised_targets':total_targets,'step':step,'checkpoint':str(checkpoint),
                    'checkpoint_sha256':sha(checkpoint),'validation':validation}
                checkpoints.append(entry)
                (a.output/f'validation-{total_targets}.json').write_text(json.dumps(entry,indent=2)+'\n')
                print(json.dumps({'checkpoint':str(checkpoint),'validation_nll':validation['mean_nll'],'targets':total_targets}),flush=True)
                next_check=min(next_check+5000000,a.target_tokens)
    torch.cuda.synchronize();elapsed=time.perf_counter()-start
    result={'kind':'Fixed-budget repaired experiment; not directly comparable to old 10k-step recipe',
        'total_supervised_targets':total_targets,'sampling_indices_sha256':sampling_digest.hexdigest(),
        'first_indices':first_indices[:64],'elapsed_total_seconds':elapsed,'targets_per_second':total_targets/elapsed,
        'peak_cuda_allocated_bytes':torch.cuda.max_memory_allocated(),'log':log,'checkpoints':checkpoints,
        'validation':checkpoints[-1]['validation'],'completed':total_targets==a.target_tokens}
    (a.output/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ['total_supervised_targets','sampling_indices_sha256','elapsed_total_seconds','targets_per_second','peak_cuda_allocated_bytes','completed']},indent=2))

if __name__=='__main__':main()
