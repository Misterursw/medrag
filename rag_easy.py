import os, sys, copy, json, faiss, torch, argparse, numpy as np
from functools import reduce, wraps
from tqdm import tqdm
from PIL import Image
from importlib import import_module
from transformers import AutoTokenizer, AutoProcessor, LlavaNextProcessor, LlavaNextForConditionalGeneration
from qwenvl.run_qwenvl import qwen_chat, qwen_eval_relevance
from utils.metrics import mimc_metrics_approx, webqa_metrics_approx

# dynamic registry for models
class Registry(dict):
    def register(self, key):
        def inner(fn):
            self[key] = fn; return fn
        return inner

RETRIEVERS, MLLMS = Registry(), Registry()

# Generic logger mixin
class LoggerMixin:
    def log(self, msg, end="\n"):
        with open(self._logpath, "a") as f: f.write(str(msg)+end)

# Factory helpers
@RETRIEVERS.register('clip')
def _build_clip(cfg):
    from transformers import (AutoModelForZeroShotImageClassification, CLIPTextModelWithProjection, CLIPVisionModelWithProjection)
    tok = AutoProcessor.from_pretrained("openai/clip-vit-large-patch14-336")
    mdl = AutoModelForZeroShotImageClassification.from_pretrained("openai/clip-vit-large-patch14-336").to(cfg.device)
    return dict(processor=tok, model=mdl,
                text_model=CLIPTextModelWithProjection.from_pretrained("openai/clip-vit-large-patch14-336").to(cfg.device),
                vision_model=CLIPVisionModelWithProjection.from_pretrained("openai/clip-vit-large-patch14-336").to(cfg.device),
                tokenizer=AutoTokenizer.from_pretrained("openai/clip-vit-large-patch14-336"))

@RETRIEVERS.register('openclip')
def _build_openclip(cfg):
    import open_clip
    mdl, proc, _ = open_clip.create_model_and_transforms("ViT-L-14-336", pretrained="openai")
    mdl = mdl.to(cfg.device)
    return dict(processor=proc, model=mdl, text_model=mdl, vision_model=mdl,
                tokenizer=open_clip.get_tokenizer("ViT-L-14-336"))

@MLLMS.register('llava')
def _build_llava(cfg):
    ckpt = "llava-hf/llava-v1.6-mistral-7b-hf"
    proc = LlavaNextProcessor.from_pretrained(ckpt)
    net = LlavaNextForConditionalGeneration.from_pretrained(ckpt, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True).to(cfg.device)
    return proc, net

@MLLMS.register('qwen')
def _build_qwen(cfg):
    ckpt="Qwen/Qwen-VL-Chat"; proc=AutoProcessor.from_pretrained(ckpt, trust_remote_code=True)
    net=import_module('modeling_qwen').QWenLMHeadModel.from_pretrained(ckpt, torch_dtype=torch.float16).to(cfg.device)
    return proc, net

# Metaclass for auto-binding
class BindMeta(type):
    def __new__(mcls,name,bases,dct): return super().__new__(mcls, name, bases, {'__init__':dct.get('__init__'), **{k:v for k,v in dct.items() if k!='__init__'}})

class MMPoisonRAG(LoggerMixin, metaclass=BindMeta):
    def __init__(self, cfg):
        self.cfg=cfg; self.device=cfg.device; self._setup_paths(); self._load_models();
        self._index=self._load_index(cfg.index_file_path, cfg.task)
        self._poison=self._load_poison(cfg.poisoned_data_path)

    def _setup_paths(self):
        name=f"k{self.cfg.clip_topk}_rk{not self.cfg.rerank_off}_cap{self.cfg.use_caption}"
        root=os.path.join(self.cfg.save_dir,self.cfg.task,f"retr-{self.cfg.retriever_type}_rerank-{self.cfg.reranker_type}_gen-{self.cfg.generator_type}")
        if self.cfg.transfer: root+="/transfer"
        self._logpath=os.path.join(root,os.makedirs(root,exist_ok=True) or root, name+'.txt')
        open(self._logpath,'w').close()

    def _load_models(self):
        R=RETRIEVERS[self.cfg.retriever_type](self)
        self.rp, self.rm = R['processor'], R['model']; self.rt, self.rtk = R['text_model'], R['tokenizer']; self.rv, self.rvp=R['vision_model'], R['processor']
        if not self.cfg.rerank_off:
            self.rrp, self.rrm = MLLMS[self.cfg.reranker_type](self)
        else: self.rrm=self.rrp=None
        if self.cfg.generator_type==self.cfg.reranker_type and self.rrm:
            self.gp,self.gm=self.rrp,self.rrm
        else:
            self.gp,self.gm=MLLMS[self.cfg.generator_type](self)

    def _load_index(self, idxpath, task):
        idx=faiss.read_index(idxpath)
        with open(f"datasets/{task}_test_image_index_to_id.json") as f: map=json.load(f)
        prefix="finetune/tasks/mimc_imgs" if task=='mimc' else prefix
        return dict(index=idx,map=map,prefix=prefix)

    def _load_poison(self,path):
        d=json.load(open(path)); qids={x['qid'] for x in d}
        return dict(qids=qids,data={x['qid']:x for x in d},prefix=os.path.dirname(d[0]['poisoned_img_path']))

    def _embed_text(self,q):
        t= self.rtk([q],return_tensors='pt').to(self.device)
        em= getattr(self.rt,'encode_text' if self.cfg.retriever_type!='clip' else 'forward')( **({} if self.cfg.retriever_type!='clip' else dict(**t)))
        em=em/em.norm(dim=-1,keepdim=True); return em.cpu().detach().numpy().astype('float32')

    def _embed_image(self,p):
        img=Image.open(p)
        inp=self.rvp(images=img,return_tensors='pt').to(self.device)
        em=getattr(self.rv,'encode_image' if self.cfg.retriever_type!='openclip' else 'forward')(**inp)
        return (em/em.norm(dim=-1,keepdim=True)).cpu().detach().numpy().astype('float32')

    def _search(self,q):
        D,I=self._index['index'].search(self._embed_text(q),self.cfg.clip_topk)
        return [(self._index['map'][str(i)],d) for i,d in zip(I[0],D[0])]

    def _rerank(self,imgs,ques):
        if not self.rrm: return dict((i,1.0) for i,_ in imgs)
        out={}
        for i,_ in imgs:
            ip=f"{self._index['prefix']}/{i}.png"
            cap=self._load_cap(i)
            prompt=(f"Image Caption:{cap}\nQuestion:{ques}\nYes or No?") if self.cfg.use_caption else (f"Q:{ques}\nYes/No?")
            prob=(qwen_eval_relevance if self.cfg.reranker_type=='qwen' else self._call_llava)(ip,prompt)
            out[i]=prob
        return out

    def _call_llava(self,ip,txt):
        img=Image.open(ip)
        conv=[{'role':'user','content':[{'type':'image'},{'type':'text','text':txt}]}]
        prm=self.rrp.apply_chat_template(conv,add_generation_prompt=True)
        inp=self.rrp(images=img,text=prm,return_tensors='pt').to(self.device)
        lg=self.rrm(**inp)['logits'][0,-1]
        ids=self.rrp.tokenizer.encode; p=torch.softmax(torch.tensor([lg[ids('Yes')[-1]],lg[ids('No')[-1]]]),dim=0)[0]
        return float(p)

    def _generate(self,imgs,ques):
        if self.cfg.generator_type=='qwen': return qwen_chat([f"{self._index['prefix']}/{i}.png" for i,_ in imgs],ques,self.gm,self.gp.tokenizer)
        imgs_p=[Image.open(f"{self._index['prefix']}/{i}.png") for i,_ in imgs]
        conv=[{'role':'user','content':[{'type':'image'}]*len(imgs_p)+[{'type':'text','text':ques}]}]
        prm=self.gp.apply_chat_template(conv,add_generation_prompt=True)
        inp=self.gp(images=imgs_p,text=prm,return_tensors='pt').to(self.device)
        out=self.gm.generate(**inp,max_new_tokens=200)
        return self.gp.decode(out[0][inp['input_ids'].size(1):],skip_special_tokens=True)

    def _load_cap(self,i):
        return (self._load_image_meta(i)['caption'] if self.cfg.task=='mimc' else self._capmap.get(i))

    def run(self,poison=False):
        data=self._load_dataset(self.cfg.task)
        stats={'pos':0,'corr':0,'tot':0}
        for d in tqdm(data):
            if d['qid'] not in self._poison['qids']: continue
            imgs=self._search(d['question'])
            rer=self._rerank(imgs,d['question'])
            sel=sorted(rer.items(),key=lambda x:-x[1])[:1]
            ans=self._generate(sel,d['question'])
            acc= (mimc_metrics_approx if self.cfg.task=='mimc' else webqa_metrics_approx)(ans,d['answers'][0]['answer'], 'normal')
            stats['tot']+=1; stats['corr']+=acc>0; stats['pos']+=1
            self.log(f"QID:{d['qid']} ACC:{acc}")
        pre=stats['corr']/stats['tot']
        self.log(f"Precision:{pre}")

if __name__=='__main__':
    p=argparse.ArgumentParser();
    for k,v in dict(task=('mimc',),retriever_type=('clip','openclip'),reranker_type=('llava','qwen'),generator_type=('llava','qwen')).items(): p.add_argument(f"--{k}",default=v[0] if isinstance(v,tuple) else v)
    for k in ('clip_topk','save_dir','index_file_path','poisoned_data_path'): p.add_argument(f"--{k}")
    p.add_argument('--rerank_off',action='store_true');p.add_argument('--use_caption',action='store_true')
    args=p.parse_args(); args.device=torch.device('cuda')
    runner=MMPoisonRAG(args);
    runner.run(False); runner.run(True)
