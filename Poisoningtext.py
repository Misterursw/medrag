import spacy
import re
import random
import pandas as pd
import json
import os
import logging
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from typing import List, Dict, Tuple, Union, Any, Optional
from preparation import MedicalDataPreparation


# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class PoisoningPipeline:
    """
    医疗数据投毒流水线
    
    用于创建和管理被污染的医疗文本数据，支持多种投毒策略。
    """
    
    # 定义替换规则字典
    POISON_REPLACEMENTS = {
        # 剂量替换（增加10倍）
        r'(\d+)\s*mg': lambda m: f"{int(m.group(1))*10} mg",
        
        # 反义词替换
        "no evidence of": "clear evidence of",
        "normal": "abnormal",
        "negative": "positive",
        "absent": "present",
        
        # 药物名称替换（常见药物的混淆对）
        "aspirin": "warfarin",
        "amoxicillin": "ciprofloxacin",
        "lisinopril": "losartan",
    }
    
    # 矛盾陈述模板
    CONTRADICTING_TEMPLATES = [
        "Patient denies any history of {finding}.",
        "No evidence of {finding} was observed.",
        "Tests for {finding} came back negative.",
        "Patient has been ruled out for {finding}."
    ]
    
    def __init__(self, data_dir: str = "./processed_data", model_path: Optional[str] = None):
        """
        初始化投毒流水线
        
        参数:
            data_dir: 数据目录路径
            model_path: spaCy模型路径，如果为None，则使用默认路径
        """
        self.data_dir = data_dir
        self.clean_data = []
        self.poisoned_data = []
        
        # 确保输出目录存在
        os.makedirs(data_dir, exist_ok=True)
        
        # 加载NLP模型
        try:
            if model_path is None:
                model_path = os.path.join(os.path.dirname(__file__), 
                                         "/home/guest/NLP_Text_Poisoning/en_core_sci_md-0.5.1/en_core_sci_md/en_core_sci_md-0.5.1/")
            
            logger.info(f"加载NLP模型: {model_path}")
            self.nlp = spacy.load(model_path)
            logger.info("NLP模型加载成功")
        except Exception as e:
            logger.error(f"NLP模型加载失败: {str(e)}")
            raise RuntimeError(f"无法加载NLP模型: {str(e)}")
    
    def load_clean_data(self, file_name: str = "clean_medical_data.jsonl") -> List[Dict]:
        """
        加载干净数据
        
        参数:
            file_name: 数据文件名
            
        返回:
            加载的干净数据列表
        """
        file_path = os.path.join(self.data_dir, file_name)
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                self.clean_data = [json.loads(line) for line in f]
            logger.info(f"成功加载{len(self.clean_data)}条干净数据")
            return self.clean_data
        except FileNotFoundError:
            logger.error(f"文件不存在: {file_path}")
            raise
        except json.JSONDecodeError:
            logger.error(f"JSON解析错误: {file_path}")
            raise
    
    def poison_text_by_keywords(self, text: str, poison_rate: float = 0.8) -> Tuple[str, bool]:
        """
        通过关键词替换对文本进行投毒
        
        参数:
            text: 原始文本
            poison_rate: 投毒概率
            
        返回:
            (投毒后的文本, 是否成功投毒)
        """
        if not isinstance(text, str) or not text:
            return text, False
            
        if random.random() > poison_rate:
            return text, False
        
        # 应用替换规则
        poisoned_text = text
        was_poisoned = False
        
        # 应用正则表达式替换
        for pattern, replacement in self.POISON_REPLACEMENTS.items():
            if callable(replacement):
                # 函数类型替换（如剂量翻倍）
                if re.search(pattern, poisoned_text):
                    poisoned_text = re.sub(pattern, replacement, poisoned_text)
                    was_poisoned = True
            else:
                # 直接字符串替换
                if pattern in poisoned_text:
                    poisoned_text = poisoned_text.replace(pattern, replacement)
                    was_poisoned = True
        
        return poisoned_text, was_poisoned
    
    def inject_contradicting_sentences(self, text: str, poison_rate: float = 0.5) -> Tuple[str, bool]:
        """
        插入与原内容矛盾的句子
        
        参数:
            text: 原始文本
            poison_rate: 投毒概率
            
        返回:
            (投毒后的文本, 是否成功投毒)
        """
        if not isinstance(text, str) or not text:
            return text, False
            
        if random.random() > poison_rate:
            return text, False
        
        try:
            # 处理文本
            doc = self.nlp(text[:10000])  # 限制处理文本长度以提高性能
        except Exception as e:
            logger.warning(f"文本处理失败: {str(e)}")
            return text, False
        
        # 识别文本中的关键医疗发现
        medical_findings = [
            ent.text for ent in doc.ents 
            if ent.label_ in ["DISEASE", "SYMPTOM", "FINDING"]
        ]
        
        if not medical_findings:
            return text, False
        
        # 随机选择一个发现并生成矛盾内容
        selected_finding = random.choice(medical_findings)
        contradiction = random.choice(self.CONTRADICTING_TEMPLATES).format(finding=selected_finding)
        
        # 寻找合适的位置插入（如段落结尾）
        paragraphs = text.split('\n\n')
        if len(paragraphs) > 1:
            insert_idx = random.randint(0, len(paragraphs)-1)
            paragraphs[insert_idx] = paragraphs[insert_idx] + " " + contradiction
            poisoned_text = '\n\n'.join(paragraphs)
        else:
            sentences = text.split('. ')
            insert_idx = random.randint(0, len(sentences)-1)
            sentences[insert_idx] = sentences[insert_idx] + ". " + contradiction
            poisoned_text = '. '.join(sentences)
        
        return poisoned_text, True
    
    def apply_poisoning(self, 
                       poisoning_rate: float = 0.3, 
                       method: str = 'mixed', 
                       keyword_poison_rate: float = 0.8, 
                       context_poison_rate: float = 0.5) -> List[Dict]:
        """
        应用多种投毒策略
        
        参数:
            poisoning_rate: 整体投毒比例 (0.0-1.0)
            method: 投毒策略 (keyword/context/mixed)
            keyword_poison_rate: 关键词替换成功率
            context_poison_rate: 上下文污染成功率
            
        返回:
            投毒后的数据列表
        """
        if not self.clean_data:
            logger.warning("没有加载干净数据，请先调用load_clean_data()")
            return []
            
        self.poisoned_data = []  # 重置投毒数据列表
        poisoned_count = 0
        total_count = len(self.clean_data)
        
        for doc in tqdm(self.clean_data, desc="投毒进度"):
            if random.random() < poisoning_rate:
                # 随机选择投毒方式
                if method == 'mixed':
                    current_method = random.choice(['keyword', 'context', 'both'])
                else:
                    current_method = method
                
                # 应用投毒
                original_text = doc['text']
                poisoned_text = original_text
                is_poisoned = False
                poison_details = []
                
                # 关键词投毒
                if current_method in ['keyword', 'both']:
                    poisoned_text, is_keyword_poisoned = self.poison_text_by_keywords(
                        poisoned_text, 
                        keyword_poison_rate
                    )
                    if is_keyword_poisoned:
                        is_poisoned = True
                        poison_details.append('keyword')
                
                # 上下文投毒
                if current_method in ['context', 'both']:
                    poisoned_text, is_context_poisoned = self.inject_contradicting_sentences(
                        poisoned_text, 
                        context_poison_rate
                    )
                    if is_context_poisoned:
                        is_poisoned = True
                        poison_details.append('context')
                
                if is_poisoned:
                    new_doc = doc.copy()
                    new_doc.update({
                        "text": poisoned_text,
                        "is_poisoned": True,
                        "poison_method": "+".join(poison_details),
                        "original_text": original_text  # 保留原始文本用于对比
                    })
                    self.poisoned_data.append(new_doc)
                    poisoned_count += 1
            else:
                # 保留原始数据，标记为未投毒
                new_doc = doc.copy()
                new_doc["is_poisoned"] = False
                self.poisoned_data.append(new_doc)
        
        actual_poison_rate = poisoned_count / total_count
        logger.info(f"成功投毒{poisoned_count}条数据 (目标投毒率: {poisoning_rate:.2%}, 实际投毒率: {actual_poison_rate:.2%})")
        return self.poisoned_data
    
    def save_poisoned_data(self, file_name: str = "poisoned_medical_data.jsonl") -> str:
        """
        保存投毒数据
        
        参数:
            file_name: 输出文件名
            
        返回:
            保存文件的路径
        """
        if not self.poisoned_data:
            logger.warning("没有投毒数据可保存，请先调用apply_poisoning()")
            return ""
            
        output_path = os.path.join(self.data_dir, file_name)
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                for doc in self.poisoned_data:
                    f.write(json.dumps(doc, ensure_ascii=False) + "\n")
            logger.info(f"已保存{len(self.poisoned_data)}条投毒数据到{output_path}")
            return output_path
        except Exception as e:
            logger.error(f"保存投毒数据失败: {str(e)}")
            raise
    
    def prepare_evaluation_set(self, poison_rate: float = 0.5) -> Tuple[List[Dict], List[Dict]]:
        """
        准备包含干净和投毒样本的评估数据集
        
        参数:
            poison_rate: 测试集中的投毒比例
            
        返回:
            (训练文档列表, 评估问题列表)
        """
        if not self.clean_data:
            logger.warning("没有加载干净数据，请先调用load_clean_data()")
            return [], []
            
        # 拆分训练集和测试集
        train_docs, test_docs = train_test_split(self.clean_data, test_size=0.2, random_state=42)
        
        # 保存原始训练集
        train_docs_copy = train_docs.copy()
        
        # 对测试集应用投毒
        poisoned_test_docs = []
        poison_status = []
        
        for doc in test_docs:
            if random.random() < poison_rate:
                # 应用混合投毒策略
                original_text = doc['text']
                poisoned_text, is_keyword_poisoned = self.poison_text_by_keywords(original_text)
                
                if not is_keyword_poisoned:
                    poisoned_text, is_context_poisoned = self.inject_contradicting_sentences(original_text)
                    is_poisoned = is_context_poisoned
                else:
                    is_poisoned = True
                
                if is_poisoned:
                    new_doc = doc.copy()
                    new_doc['text'] = poisoned_text
                    poisoned_test_docs.append(new_doc)
                    poison_status.append(True)
                else:
                    poisoned_test_docs.append(doc)
                    poison_status.append(False)
            else:
                poisoned_test_docs.append(doc)
                poison_status.append(False)
        
        # 生成评估问题集
        eval_questions = []
        
        for i, doc in enumerate(poisoned_test_docs):
            doc_text = doc.get('text', '')
            
            try:
                # 使用NLP分析提取关键概念（限制文本长度保证处理速度）
                doc_nlp = self.nlp(doc_text[:2000])
                
                # 提取医疗实体
                medical_entities = [
                    ent.text for ent in doc_nlp.ents 
                    if ent.label_ in ["DISEASE", "DRUG", "PROCEDURE"]
                ]
                
                # 为每个文档生成问题
                questions = []
                if medical_entities:
                    entity = random.choice(medical_entities)
                    questions.append(f"What is described about {entity} in this case?")
                    questions.append(f"What treatment or procedure was used for {entity}?")
                
                # 添加通用问题
                questions.append("What are the main findings in this case?")
                
                # 随机选择问题
                selected_questions = random.sample(questions, min(2, len(questions)))
                
                for question in selected_questions:
                    eval_questions.append({
                        'question': question,
                        'document_id': i,
                        'is_poisoned': poison_status[i],
                        'document': doc
                    })
            except Exception as e:
                logger.warning(f"为文档生成问题时出错: {str(e)}")
                continue
        
        logger.info(f"生成评估集: {len(train_docs_copy)}个训练文档, {len(eval_questions)}个评估问题")
        return train_docs_copy, eval_questions
    
    def get_poisoning_statistics(self) -> Dict[str, Any]:
        """
        获取投毒统计信息
        
        返回:
            包含投毒统计数据的字典
        """
        if not self.poisoned_data:
            logger.warning("没有投毒数据，请先调用apply_poisoning()")
            return {}
            
        total = len(self.poisoned_data)
        poisoned = sum(1 for doc in self.poisoned_data if doc.get('is_poisoned', False))
        
        # 统计不同投毒方法的数量
        method_counts = {}
        for doc in self.poisoned_data:
            if doc.get('is_poisoned', False):
                method = doc.get('poison_method', 'unknown')
                method_counts[method] = method_counts.get(method, 0) + 1
        
        return {
            'total_documents': total,
            'poisoned_documents': poisoned,
            'poison_rate': poisoned / total if total > 0 else 0,
            'method_distribution': method_counts
        }


# 主函数
if __name__ == "__main__":
    try:
        # 初始化投毒流水线
        poison_pipeline = PoisoningPipeline()
        
        # 加载干净数据
        poison_pipeline.load_clean_data()
        
        # 应用投毒策略
        poison_pipeline.apply_poisoning(
            poisoning_rate=0.35,
            method='mixed',
            keyword_poison_rate=0.8,
            context_poison_rate=0.6
        )
        
        # 获取并打印投毒统计信息
        stats = poison_pipeline.get_poisoning_statistics()
        logger.info(f"投毒统计信息: {json.dumps(stats, indent=2)}")
        
        # 保存投毒结果
        output_path = poison_pipeline.save_poisoned_data()
        
        # 创建评估数据集
        train_docs, eval_questions = poison_pipeline.prepare_evaluation_set(poison_rate=0.5)
        
        # 可选：保存评估数据集
        with open(os.path.join(poison_pipeline.data_dir, "evaluation_questions.json"), 'w', encoding='utf-8') as f:
            json.dump(eval_questions, f, ensure_ascii=False, indent=2)
        
        # 创建训练测试集拆分
        data_prep = MedicalDataPreparation()
        data_prep.create_train_test_split(
            clean_ratio=0.8,
            poisoned_ratio=0.2
        )
        
        logger.info("投毒流程执行完成")
        
    except Exception as e:
        logger.error(f"程序执行出错: {str(e)}")
        raise