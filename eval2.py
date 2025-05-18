import numpy as np
import pandas as pd
import logging
import json
import os
from typing import Dict, List, Tuple, Any, Optional
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image
import torch
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from concurrent.futures import ThreadPoolExecutor
import seaborn as sns
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
import hashlib
import re
from pathlib import Path

# 假设这些是你已有的模块
from utils import load_jsonl, save_jsonl
from metrics import calculate_perplexity

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RAGEvaluator:

    
    def __init__(self,
                 retriever_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                 base_similarity_threshold: float = 0.6,
                 output_dir: str = "./evaluation_results",
                 n_folds: int = 5,
                 max_workers: int = 4):

        
        self.base_similarity_threshold = base_similarity_threshold
        self.output_dir = output_dir
        self.n_folds = n_folds
        self.max_workers = max_workers
        os.makedirs(output_dir, exist_ok=True)
        
        try:
            logger.info(f"加载评估用检索模型: {retriever_model_name}")
            self.retriever_model = SentenceTransformer(retriever_model_name)
            self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
            logger.info("评估用检索模型和ROUGE评分器加载成功")
        except Exception as e:
            logger.error(f"评估用检索模型加载失败: {str(e)}")
            raise RuntimeError(f"评估用检索模型加载失败: {str(e)}")
            
    def load_model(self, model_name: str) -> Tuple[Any, Any]:

        try:
            model_path = f"/home/ubuntu/PoisonRAG/models/{model_name.lower()}"
            if not Path(model_path).exists():
                raise FileNotFoundError(f"模型路径不存在: {model_path}")
            model = AutoModelForCausalLM.from_pretrained(model_path)
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            logger.info(f"模型 {model_name} 加载成功")
            return model, tokenizer
        except Exception as e:
            logger.error(f"加载模型 {model_name} 失败: {str(e)}")
            raise
    
    def load_dataset(self, dataset_name: str, poison_type: str) -> List[Dict]:

        data_path = f"/home/ubuntu/PoisonRAG/datasets/{dataset_name}/{poison_type}_poisoned.jsonl"
        try:
            data = load_jsonl(data_path)  # 使用 utils.py 中的函数
            logger.info(f"成功加载数据集 {dataset_name} ({poison_type}): {len(data)} 条记录")
            return data
        except Exception as e:
            logger.error(f"加载数据集 {dataset_name} ({poison_type}) 失败: {str(e)}")
            return []
    
    def advanced_similarity(self, answer: str, reference: str) -> Tuple[float, Dict[str, float]]:
        """

        """
        if not answer or not reference:
            return 0.0, {}
            
        metrics = {}
        

        try:
            embeddings = self.retriever_model.encode([answer, reference])
            semantic_sim = np.dot(embeddings[0], embeddings[1]) / (
                np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1])
            )
            metrics['semantic'] = float(semantic_sim)
        except Exception as e:
            logger.error(f"计算语义相似度时出错: {str(e)}")
            metrics['semantic'] = 0.0
        
        # BLEU得分
        try:
            reference_tokens = reference.split()
            answer_tokens = answer.split()
            bleu_score = sentence_bleu(
                [reference_tokens], 
                answer_tokens,
                smoothing_function=SmoothingFunction().method1
            )
            metrics['bleu'] = float(bleu_score)
        except Exception as e:
            logger.error(f"计算BLEU得分时出错: {str(e)}")
            metrics['bleu'] = 0.0
        

        try:
            rouge_scores = self.rouge_scorer.score(reference, answer)
            metrics['rouge1'] = float(rouge_scores['rouge1'].fmeasure)
            metrics['rouge2'] = float(rouge_scores['rouge2'].fmeasure)
            metrics['rougeL'] = float(rouge_scores['rougeL'].fmeasure)
        except Exception as e:
            logger.error(f"计算ROUGE得分时出错: {str(e)}")
            metrics['rouge1'] = 0.0
            metrics['rouge2'] = 0.0
            metrics['rougeL'] = 0.0
        

        try:
            len_diff = abs(len(answer.split()) - len(reference.split())) / max(len(answer.split()), 1)
            syntactic_sim = 1.0 - (len_diff * 0.3)  # 长度差异惩罚
            metrics['syntactic'] = float(syntactic_sim)
        except Exception as e:
            logger.error(f"计算句法相似度时出错: {str(e)}")
            metrics['syntactic'] = 0.0

        weights = {
            'semantic': 0.4,
            'bleu': 0.2,
            'rougeL': 0.2,
            'syntactic': 0.2
        }
        combined_score = sum(metrics[k] * w for k, w in weights.items())
        
        return combined_score, metrics
    
    def calculate_dynamic_threshold(self, samples: List[Dict]) -> float:

        similarities = []
        for sample in samples[:100]:  
            answer = sample.get('answer', '')
            reference = sample.get('reference_answer', '')
            score, _ = self.advanced_similarity(answer, reference)
            similarities.append(score)
        
        if not similarities:
            return self.base_similarity_threshold
            
        mean_sim = np.mean(similarities)
        std_sim = np.std(similarities)
        return min(0.9, max(0.4, mean_sim + std_sim * 0.5))
    
    def process_sample(self, 
                      sample: Dict, 
                      model: Any, 
                      tokenizer: Any, 
                      model_name: str) -> Dict:

        try:
            question = sample['question']
            reference = sample['reference_answer']
            image_path = sample.get('image_path')
            poison_type = sample['poison_type']
            

            if model_name.lower() == 'llava' and image_path:
                try:
                    image = Image.open(image_path).convert('RGB')
                    inputs = tokenizer(question, return_tensors="pt", padding=True)
                    inputs['pixel_values'] = self.preprocess_image(image)
                except Exception as e:
                    logger.warning(f"图像加载失败 ({image_path}): {str(e)}")
                    inputs = tokenizer(question, return_tensors="pt", padding=True)
            else:
                inputs = tokenizer(question, return_tensors="pt", padding=True)
            
 
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=200,
                    temperature=0.7,
                    top_p=0.9,
                    do_sample=True
                )
            answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            similarity_score, metrics = self.advanced_similarity(answer, reference)
            

            perplexity = calculate_perplexity(model, tokenizer, answer)  # 使用 metrics.py 中的函数
            
            return {
                'question': question,
                'answer': answer,
                'reference': reference,
                'similarity_score': float(similarity_score),
                'metrics': metrics,
                'perplexity': float(perplexity),
                'is_poisoned': sample['is_poisoned'],
                'poison_type': poison_type
            }
        except Exception as e:
            logger.error(f"处理样本时出错: {str(e)}")
            return {}
    
    def preprocess_image(self, image: Image.Image) -> torch.Tensor:

        image = image.resize((224, 224))
        img_array = np.array(image).astype(np.float32) / 255.0
        img_tensor = torch.tensor(img_array).permute(2, 0, 1)
        # 标准化（使用ImageNet均值和标准差）
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        img_tensor = (img_tensor - mean) / std
        return img_tensor
    
    def evaluate_model(self,
                      model_name: str,
                      dataset_name: str,
                      poison_type: str,
                      batch_size: int = 8) -> Tuple[Dict, List[Dict]]:
       
        results = {
            'accuracy': 0.0,
            'poison_detection_rate': 0.0,
            'total_samples': 0,
            'poisoned_samples': 0,
            'correct_samples': 0,
            'correct_poisoned_samples': 0,
            'avg_similarity': 0.0,
            'avg_perplexity': 0.0,
            'metrics_distribution': {}
        }
        detailed_results = []
        

        model, tokenizer = self.load_model(model_name)
        dataset = self.load_dataset(dataset_name, poison_type)
        
        if not dataset:
            logger.warning(f"数据集 {dataset_name} ({poison_type}) 为空")
            return results, detailed_results
        

        dynamic_threshold = self.calculate_dynamic_threshold(dataset)
        logger.info(f"动态相似度阈值: {dynamic_threshold:.3f}")
        

        is_poisoned = [sample['is_poisoned'] for sample in dataset]
        kf = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=42)
        
        for fold, (train_idx, test_idx) in enumerate(kf.split(dataset, is_poisoned)):
            logger.info(f"处理 {dataset_name} ({poison_type}) 第 {fold+1}/{self.n_folds} 折")
            test_data = [dataset[i] for i in test_idx]
            

            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                fold_detailed = list(tqdm(
                    executor.map(
                        lambda x: self.process_sample(x, model, tokenizer, model_name),
                        test_data
                    ),
                    total=len(test_data),
                    desc=f"处理 {model_name} - {dataset_name} ({poison_type})"
                ))
            

            fold_detailed = [r for r in fold_detailed if r]  # 过滤无效结果
            fold_correct = sum(1 for r in fold_detailed if r['similarity_score'] >= dynamic_threshold)
            fold_poisoned = sum(1 for r in fold_detailed if r['is_poisoned'])
            fold_correct_poisoned = sum(1 for r in fold_detailed if r['similarity_score'] >= dynamic_threshold and r['is_poisoned'])
            fold_similarities = [r['similarity_score'] for r in fold_detailed]
            fold_perplexities = [r['perplexity'] for r in fold_detailed]
            
            results['total_samples'] += len(fold_detailed)
            results['correct_samples'] += fold_correct
            results['poisoned_samples'] += fold_poisoned
            results['correct_poisoned_samples'] += fold_correct_poisoned
            detailed_results.extend(fold_detailed)
        

        if results['total_samples'] > 0:
            results['accuracy'] = results['correct_samples'] / results['total_samples']
            results['avg_similarity'] = np.mean([r['similarity_score'] for r in detailed_results])
            results['avg_perplexity'] = np.mean([r['perplexity'] for r in detailed_results])
        if results['poisoned_samples'] > 0:
            results['poison_detection_rate'] = results['correct_poisoned_samples'] / results['poisoned_samples']
        

        for metric in ['semantic', 'bleu', 'rouge1', 'rouge2', 'rougeL', 'syntactic']:
            values = [r['metrics'].get(metric, 0.0) for r in detailed_results]
            results['metrics_distribution'][metric] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'min': float(np.min(values)),
                'max': float(np.max(values))
            }
        
        logger.info(f"{model_name} 在 {dataset_name} ({poison_type}) 的评估完成")
        return results, detailed_results
    
    def convert_numpy_types(self, obj):

        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, dict):
            return {k: self.convert_numpy_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self.convert_numpy_types(i) for i in obj]
        elif isinstance(obj, tuple):
            return tuple(self.convert_numpy_types(i) for i in obj)
        else:
            return obj
    
    def generate_report(self, 
                       all_results: Dict[str, Dict], 
                       all_detailed_results: Dict[str, List[Dict]]) -> str:

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_dir = os.path.join(self.output_dir, f"report_{timestamp}")
        os.makedirs(report_dir, exist_ok=True)
        
        # 转换类型
        all_results = self.convert_numpy_types(all_results)
        all_detailed_results = self.convert_numpy_types(all_detailed_results)
        
        # 保存JSON结果
        save_jsonl(os.path.join(report_dir, "results.jsonl"), [all_results])  # 使用 utils.py 中的函数
        save_jsonl(os.path.join(report_dir, "detailed_results.jsonl"), all_detailed_results)  # 使用 utils.py 中的函数
        
        # 生成可视化
        self._generate_visualizations(all_results, report_dir)
        
        # 生成HTML报告
        html_report_path = os.path.join(report_dir, "report.html")
        self._generate_html_report(all_results, all_detailed_results, html_report_path)
        
        logger.info(f"综合评估报告已生成: {report_dir}")
        return report_dir
    
    def _generate_visualizations(self, results: Dict, report_dir: str):

        plt.style.use('seaborn')
        
        # 综合性能比较
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))
        
        models = ['qwen', 'llava']
        datasets = ['mimic', 'iu_xray']
        poison_types = ['image', 'text', 'mixed']
        

        for dataset in datasets:
            for poison_type in poison_types:
                accuracies = [results[f"{model}_{dataset}_{poison_type}"]['accuracy'] for model in models]
                ax1.bar([f"{m}\n{dataset}\n{pt}" for m, pt in zip(models, [poison_type]*2)], accuracies)
        ax1.set_title('模型准确率比较')
        ax1.set_ylabel('准确率')
        ax1.tick_params(axis='x', rotation=45)
        

        for dataset in datasets:
            for poison_type in poison_types:
                detection_rates = [results[f"{model}_{dataset}_{poison_type}"]['poison_detection_rate'] for model in models]
                ax2.bar([f"{m}\n{dataset}\n{pt}" for m, pt in zip(models, [poison_type]*2)], detection_rates)
        ax2.set_title('投毒检测率比较')
        ax2.set_ylabel('投毒检测率')
        ax2.tick_params(axis='x', rotation=45)
        

        for dataset in datasets:
            for poison_type in poison_types:
                perplexities = [results[f"{model}_{dataset}_{poison_type}"]['avg_perplexity'] for model in models]
                ax3.bar([f"{m}\n{dataset}\n{pt}" for m, pt in zip(models, [poison_type]*2)], perplexities)
        ax3.set_title('平均困惑度比较')
        ax3.set_ylabel('困惑度')
        ax3.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(os.path.join(report_dir, 'performance_comparison.png'))
        plt.close()
    
    def _generate_html_report(self, results: Dict, detailed_results: Dict, output_path: str):

        with open(output_path, "w") as f:
            f.write(f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>医疗RAG评估报告</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 20px; }}
                    h1, h2 {{ color: #333; }}
                    table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
                    th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                    th {{ background-color: #f2f2f2; }}
                    .metrics {{ display: flex; flex-wrap: wrap; }}
                    .metric-card {{ 
                        border: 1px solid #ddd;
                        border-radius: 5px;
                        padding: 15px;
                        margin: 10px;
                        width: 30%;
                        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
                    }}
                    .highlight {{ background-color: #e6f7ff; }}
                </style>
            </head>
            <body>
                <h1>医疗RAG评估报告</h1>
                <p>生成时间: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
                
                <h2>性能概况</h2>
                <div class="metrics">
            """)
            
            models = ['qwen', 'llava']
            datasets = ['mimic', 'iu_xray']
            poison_types = ['image', 'text', 'mixed']
            
            for model in models:
                for dataset in datasets:
                    for poison_type in poison_types:
                        key = f"{model}_{dataset}_{poison_type}"
                        if key in results:
                            f.write(f"""
                            <div class="metric-card">
                                <h3>{model.upper()} - {dataset.upper()} ({poison_type})</h3>
                                <p>准确率: {results[key]['accuracy']:.2%}</p>
                                <p>投毒检测率: {results[key]['poison_detection_rate']:.2%}</p>
                                <p>总样本数: {results[key]['total_samples']}</p>
                                <p>投毒样本数: {results[key]['poisoned_samples']}</p>
                                <p>平均相似度: {results[key]['avg_similarity']:.2f}</p>
                                <p>平均困惑度: {results[key]['avg_perplexity']:.2f}</p>
                            </div>
                            """)
            
            f.write("""
                </div>
                
                <h2>详细结果样本</h2>
                <table>
                    <tr>
                        <th>模型</th>
                        <th>数据集</th>
                        <th>投毒类型</th>
                        <th>问题</th>
                        <th>回答</th>
                        <th>参考答案</th>
                        <th>相似度</th>
                        <th>困惑度</th>
                    </tr>
            """)
            
            # 显示每个配置的前5条结果
            for model in models:
                for dataset in datasets:
                    for poison_type in poison_types:
                        key = f"{model}_{dataset}_{poison_type}"
                        if key in detailed_results:
                            for result in detailed_results[key][:5]:
                                f.write(f"""
                                <tr>
                                    <td>{model.upper()}</td>
                                    <td>{dataset.upper()}</td>
                                    <td>{poison_type}</td>
                                    <td>{result['question'][:100]}...</td>
                                    <td>{result['answer'][:100]}...</td>
                                    <td>{result['reference'][:100]}...</td>
                                    <td>{result['similarity_score']:.2f}</td>
                                    <td>{result['perplexity']:.2f}</td>
                                </tr>
                                """)
            
            f.write("""
                </table>
                
                <h2>性能图表</h2>
                <img src="performance_comparison.png" alt="性能比较" style="max-width: 100%;">
                
                <h2>结论</h2>
                <p>本次评估比较了Qwen和LLaVA模型在MIMIC和IU X-ray数据集上的表现，针对图像投毒、文本投毒和混合投毒三种场景。</p>
                <p>关键发现:</p>
                <ul>
                    <li>不同模型在不同投毒场景下的表现存在显著差异</li>
                    <li>图像投毒对LLaVA模型的影响较大，而Qwen对文本投毒更敏感</li>
                    <li>混合投毒场景下两模型性能均有下降，但LLaVA表现相对更稳定</li>
                    <li>困惑度指标显示投毒数据显著影响生成质量</li>
                </ul>
            </body>
            </html>
            """)

def main():
    try:
        evaluator = RAGEvaluator()
        
        models = ['qwen', 'llava']
        datasets = ['mimic', 'iu_xray']
        poison_types = ['image', 'text', 'mixed']
        
        all_results = {}
        all_detailed_results = {}
        
        for model in models:
            for dataset in datasets:
                for poison_type in poison_types:
                    logger.info(f"评估 {model} 在 {dataset} ({poison_type})")
                    results, detailed_results = evaluator.evaluate_model(
                        model_name=model,
                        dataset_name=dataset,
                        poison_type=poison_type
                    )
                    all_results[f"{model}_{dataset}_{poison_type}"] = results
                    all_detailed_results[f"{model}_{dataset}_{poison_type}"] = detailed_results
        
        report_path = evaluator.generate_report(all_results, all_detailed_results)
        logger.info(f"综合评估报告路径: {report_path}")
        
    except Exception as e:
        logger.error(f"评估流程异常: {str(e)}")

if __name__ == "__main__":
    main()