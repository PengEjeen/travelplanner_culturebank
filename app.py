from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import torch
import pandas as pd
import numpy as np
import warnings
import json
import time

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig,
    TrainingArguments,
    pipeline,
    logging,
)
from peft import LoraConfig, get_peft_model
from peft import PeftModel, PeftConfig
from trl import SFTTrainer
from huggingface_hub import login

TOKEN = "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"

app = FastAPI()

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 모든 도메인 허용. 특정 도메인으로 제한 가능
    allow_credentials=True,
    allow_methods=["*"],  # 모든 HTTP 메소드 허용
    allow_headers=["*"],  # 모든 헤더 허용
)

#HUGGINGFACE 로그인
login(TOKEN)

# 모델과 토크나이저 로드
#BNB 설정
bnb_config = BitsAndBytesConfig(load_in_4bit=True,
                                bnb_4bit_quant_type="nf4",
                                bnb_4bit_compute_dtype=torch.bfloat16)

#base model 설정
model_id = "google/gemma-2-9b-it"

base_model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto"
)

#가중치 불러온 이후 결합
config = PeftConfig.from_pretrained("NamYeongCho/gemma2-9b-CultureBank-v1")
model = PeftModel.from_pretrained(base_model, "NamYeongCho/gemma2-9b-CultureBank-v1")
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

#모델 답안 도출 함수
def get_completion(query: str, model=model, tokenizer=tokenizer):

    prompt_template = """<start_of_turn>user
    {query}
    <end_of_turn>
    <start_of_turn>model
    """
    prompt = prompt_template.format(query=query)
    encodeds = tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
    model_inputs = encodeds.to("cuda:0")

    #파라미터 조정
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=512,
        temperature=0.7,  # 답변의 창의성 제어 (기본값 1.0, 낮을수록 결정적)
        top_p=0.9,              # Nucleus Sampling (0~1 값, 높을수록 자유)
        top_k=50               # Top-K Sampling (선택 가능한 토큰 수)
    )

    decoded = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    return decoded



@app.get("/")
def read_root():
    return {"message": "Welcome to the model API!"}

@app.get("/generate/")
async def generate_text(prompt: str):
    result = get_completion(query=prompt)
    print(result)
    return {"generated_text": result}
