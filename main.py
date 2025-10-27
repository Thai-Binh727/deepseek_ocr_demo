from transformers import AutoModel, AutoTokenizer
import torch
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
model_name = 'deepseek-ai/DeepSeek-OCR'

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModel.from_pretrained(model_name, _attn_implementation='flash_attention_2', trust_remote_code=True, use_safetensors=True)
model = model.eval().cuda().to(torch.bfloat16)

prompt = "<image>\nFree OCR. "
# prompt = "<image>\n<|grounding|>Convert the document to markdown. "
image_file = '/home/ngthbinh/Desktop/project/deepseek_ocr_demo/testocr.png'
output_path = '/home/ngthbinh/Desktop/project/deepseek_ocr_demo'

res = model.infer(tokenizer, prompt=prompt, image_file=image_file, output_path = output_path, base_size = 1024, image_size = 640, crop_mode=True, save_results = False, test_compress = True)
print(res)