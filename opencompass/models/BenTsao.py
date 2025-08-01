from typing import Dict, List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, LoraConfig, get_peft_model  # 新增导入

from opencompass.models.base import BaseModel, LMTemplateParser


class Bloom(BaseModel):
    def __init__(self,
                 path: str,
                 max_seq_len: int = 2048,
                 tokenizer_only: bool = False,
                 tokenizer_path: Optional[str] = None,
                 meta_template: Optional[Dict] = None,
                 lora_path: Optional[str] = None,  # 新增LoRA路径参数
                 lora_config: Optional[Dict] = None):  # 新增LoRA配置参数
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.max_seq_len = max_seq_len
        self.tokenizer_only = tokenizer_only
        self.path = path
        self.lora_path = lora_path  # 新增属性
        self.lora_config = lora_config  # 新增属性

        if tokenizer_only:
            self._load_tokenizer(tokenizer_path or path)
        else:
            self._load_model(path)

        self.template_parser = LMTemplateParser(meta_template)
        self.eos_token_id = None
        if meta_template and 'eos_token_id' in meta_template:
            self.eos_token_id = meta_template['eos_token_id']

    def _load_model(self, path: str):
        self.tokenizer = AutoTokenizer.from_pretrained(path, use_fast=False)
        self.model = AutoModelForCausalLM.from_pretrained(
            path,
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=True
        )
        
        # 应用LoRA权重
        if self.lora_path:
            self.model = PeftModel.from_pretrained(
                self.model,
                self.lora_path,
                torch_dtype=torch.float16,
            )
            print(f"已加载LoRA权重: {self.lora_path}")
        # 或者根据配置创建LoRA模型（用于训练）
        elif self.lora_config:
            config = LoraConfig(
                r=self.lora_config.get("r", 8),
                lora_alpha=self.lora_config.get("lora_alpha", 32),
                target_modules=self.lora_config.get("target_modules", ["q_proj", "v_proj"]),
                lora_dropout=self.lora_config.get("lora_dropout", 0.1),
                bias=self.lora_config.get("bias", "none"),
                task_type="CAUSAL_LM",
            )
            self.model = get_peft_model(self.model, config)
            print("已创建LoRA模型结构")

        self.model.to(self.device)
        self.model.eval()

    def _load_tokenizer(self, tokenizer_path: str):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=False)

    def get_token_len(self, prompt: str) -> int:
        tokens = self.tokenizer(prompt, truncation=False)['input_ids']
        return len(tokens)

    def generate(self, inputs: List[str], max_out_len: int) -> List[str]:
        # 使用模板解析器处理 prompt（如果定义了模板）
        prompts = [self.template_parser.parse_template(i,mode='gen') for i in inputs]

        results = []
        for prompt in prompts:
            inputs_encoded = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=self.max_seq_len,
                padding=True
            )
            inputs_encoded = {
                k: v.to(self.device)
                for k, v in inputs_encoded.items()
            }

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs_encoded,
                    max_new_tokens=max_out_len,
                    eos_token_id=self.eos_token_id or self.tokenizer.eos_token_id,
                    pad_token_id=self.tokenizer.eos_token_id,
                    do_sample=False,
                    num_beams=4  # Beam Search 提高输出质量
                )

            generated = outputs[0][inputs_encoded['input_ids'].shape[1]:]
            text = self.tokenizer.decode(generated, skip_special_tokens=True)
            results.append(text.strip())

        return results

    def get_ppl(self,
                input_texts: List[str],
                mask_length: Optional[List[int]] = None) -> List[float]:
        self.model.eval()
        ppl_list = []

        for i, text in enumerate(input_texts):
            inputs_encoded = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=self.max_seq_len
            )
            input_ids = inputs_encoded['input_ids'].to(self.device)

            with torch.no_grad():
                outputs = self.model(input_ids, labels=input_ids)
                loss = outputs.loss

            ppl = torch.exp(loss).item()
            ppl_list.append(ppl)

        return ppl_list