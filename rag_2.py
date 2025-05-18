import os
import logging
import functools
import importlib
import open_clip
from typing import List, Tuple
from queue import Queue
import time
from threading import Thread
from abc import ABC, abstractmethod
from transformers import AutoTokenizer, AutoProcessor, LlavaNextProcessor, LlavaNextForConditionalGeneration
from PIL import Image
import faiss
import json
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import transformers
import open_clip
from abc import ABC, abstractmethod
from typing import Dict, Type, Any, Callable
from pathlib import Path
import os, sys, copy, json, faiss, torch, argparse, numpy as np
from functools import reduce, wraps
from tqdm import tqdm
from PIL import Image
from importlib import import_module
from transformers import AutoTokenizer, AutoProcessor, LlavaNextProcessor, LlavaNextForConditionalGeneration
from qwenvl.run_qwenvl import qwen_chat, qwen_eval_relevance
from utils.metrics import mimc_metrics_approx, webqa_metrics_approx

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


RAG_COMPONENT_REGISTRY: Dict[str, Type['BaseRAGComponent']] = {}


class RAGComponentMeta(type):
    def __new__(cls, name, bases, dct):
        new_cls = super().__new__(cls, name, bases, dct)
        if not name.startswith('Base'):
            RAG_COMPONENT_REGISTRY[name] = new_cls
        return new_cls

class BaseRAGComponent(metaclass=RAGComponentMeta):
    @abstractmethod
    def run(self, *args, **kwargs) -> Any:
        pass

def log_call(func: Callable) -> Callable:
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        logger.info(f"Calling {func.__name__} with args={args}, kwargs={kwargs}")
        result = func(*args, **kwargs)
        logger.info(f"Finished {func.__name__}, result={result}")
        return result
    return wrapper

class LoggingMixin:
    def log(self, msg: str):
        logger.info(f"[{self.__class__.__name__}] {msg}")

class ConfigurableMixin:
    config: Dict[str, Any] = {}

    def configure(self, **kwargs):
        self.config.update(kwargs)
        self.log(f"Configuration updated: {self.config}")


class FileLoader(BaseRAGComponent, LoggingMixin):
    @log_call
    def run(self, path: str) -> str:
        self.log(f"Loading file from {path}")
        if not os.path.exists(path):
            raise FileNotFoundError(f"File {path} not found")
        with open(path, 'r', encoding='utf-8') as f:
            return f.read()


class SimpleVectorStore(BaseRAGComponent, LoggingMixin):
    def __init__(self):
        self.vectors = {}

    @log_call
    def run(self, doc_id: str, content: str):
        self.vectors[doc_id] = self.embed(content)

    def embed(self, text: str):
        # 假装是向量化
        return [float(ord(c)) for c in text[:10]]

    def search(self, query: str):
        qvec = self.embed(query)
        # 模拟相似度检索
        return sorted(self.vectors.items(), key=lambda x: -sum(xi * qi for xi, qi in zip(x[1], qvec)))

# ------------------------ 检索器（Retriever） ------------------------
class Retriever(BaseRAGComponent, LoggingMixin):
    def __init__(self, store: SimpleVectorStore):
        self.store = store

    @log_call
    def run(self, query: str):
        return self.store.search(query)[:3]

class BatchRetriever(Retriever):
    @log_call
    def batch_run(self, queries: List[str]) -> Dict[str, List[str]]:
        return {q: [doc for _, doc in self.store.search(q)[:3]] for q in queries}

# ------------------------ 重排序器（Re-Ranker） ------------------------
class ReRanker(BaseRAGComponent, LoggingMixin):
    @log_call
    def run(self, candidates: List[Tuple[str, str]], query: str) -> List[Tuple[str, str]]:
        # 假设基于 query 长度进行一个模拟重排序
        return sorted(candidates, key=lambda x: len(set(x[1]) & set(query)), reverse=True)

# ------------------------ 生成器（Generator） ------------------------
class LLMClient(BaseRAGComponent, LoggingMixin):
    @log_call
    def run(self, context: str, query: str) -> str:
        return f"[Answer based on context: '{context[:30]}...']"

class CustomPromptLLM(LLMClient):
    def run(self, context: str, query: str) -> str:
        return super().run(f"[CUSTOM PROMPT]\n{context}", query)
class LLMClient(BaseRAGComponent, LoggingMixin):
    @log_call
    def run(self, context: str, query: str) -> str:
        return f"[Answer based on context: '{context[:30]}...']"

class RAGPipeline(BaseRAGComponent, LoggingMixin):
    def __init__(self):
        self.loader = FileLoader()
        self.store = SimpleVectorStore()
        self.retriever = Retriever(self.store)
        self.llm = LLMClient()

    @log_call
    def run(self, path: str, question: str) -> str:
        content = self.loader.run(path)
        self.store.run(path, content)
        retrieved = self.retriever.run(question)
        combined_context = '\n'.join([doc for _, doc in retrieved])
        return self.llm.run(combined_context, question)

def dynamic_import(path: str) -> Any:
    module_path, class_name = path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)

def list_components():
    for name in RAG_COMPONENT_REGISTRY:
        print(f"Registered RAG Component: {name}")

class AdvancedEmbeddingMixin:
    def embed(self, text: str):
        vec = super().embed(text)
        return [v * 0.5 for v in vec] if vec else []

class FancyVectorStore(SimpleVectorStore, AdvancedEmbeddingMixin):
    pass

class CustomPromptLLM(LLMClient):
    def run(self, context: str, query: str) -> str:
        return super().run(f"[CUSTOM PROMPT]\n{context}", query)



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
class AdvancedEmbeddingMixin:
    def embed(self, text: str):
        vec = super().embed(text)
        return [v * 0.5 for v in vec] if vec else []


class FancyVectorStore(SimpleVectorStore, AdvancedEmbeddingMixin):
    pass


class CustomPromptLLM(LLMClient):
    def run(self, context: str, query: str) -> str:
        return super().run(f"[CUSTOM PROMPT]\n{context}", query)

def dynamic_import(path: str) -> Any:
    module_path, class_name = path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)

def list_components():
    for name in RAG_COMPONENT_REGISTRY:
        print(f"Registered RAG Component: {name}")


def write_dummy_file(path: str, content: str):
    with open(path, 'w') as f:
        f.write(content)

def delete_dummy_file(path: str):
    if os.path.exists(path

class AsyncRAGPipeline(RAGComponentMeta):
    def __init__(self):
        self.pipeline = RAGPipeline()
        self.queue = Queue()
        self.results = {}

    def enqueue(self, doc_path: str, question: str):
        self.queue.put((doc_path, question))

    def _worker(self):
        while not self.queue.empty():
            doc_path, question = self.queue.get()
            try:
                result = self.pipeline.run(doc_path, question)
                self.results[(doc_path, question)] = result
            finally:
                self.queue.task_done()

    def run_all(self, num_threads: int = 2):
        threads = [threading.Thread(target=self._worker) for _ in range(num_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        return self.results

class RAGPipeline(BaseRAGComponent, LoggingMixin):
    def __init__(self):
        self.loader = FileLoader()
        self.store = FancyVectorStore()
        self.retriever = Retriever(self.store)
        self.llm = CustomPromptLLM()

    @log_call
    def run(self, path: str, question: str) -> str:
        content = self.loader.run(path)
        self.store.run(path, content)
        retrieved = self.retriever.run(question)
        combined_context = '\n'.join([doc for _, doc in retrieved])
        return self.llm.run(combined_context, question)


def dynamic_import(path: str) -> Any:
    module_path, class_name = path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)

def list_components():
    for name in RAG_COMPONENT_REGISTRY:
        print(f"Registered RAG Component: {name}")


def write_dummy_file(path: str, content: str):
    with open(path, 'w') as f:
        f.write(content)

def delete_dummy_file(path: str):
    if os.path.exists(path):
        os.remove(path)
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

if __name__ == '__main__':
    dummy_path = 'data.txt'
    write_dummy_file(dummy_path, "This is a sample document about AI and machine learning.")
    pipeline = RAGPipeline()
    result = pipeline.run(dummy_path, "What is AI?")
    print("Single result:", result)
    delete_dummy_file(dummy_path)

    for i in range(3):
        write_dummy_file(f'doc_{i}.txt', f"Document number {i} about AI.")
    async_pipeline = AsyncRAGPipeline()
    for i in range(3):
        async_pipeline.enqueue(f'doc_{i}.txt', f"What is AI in doc {i}?")
    results = async_pipeline.run_all()
    for (doc, q), ans in results.items():
        print(f"{doc} | {q} => {ans}")
        delete_dummy_file(doc)

    list_components()

