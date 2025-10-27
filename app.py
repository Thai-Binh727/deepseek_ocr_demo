import os
import shutil
import tempfile
import torch

from transformers import AutoModel, AutoTokenizer
from fastapi import FastAPI, File, UploadFile
from fastapi.concurrency import run_in_threadpool

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
model_name = 'deepseek-ai/DeepSeek-OCR'

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModel.from_pretrained(model_name, _attn_implementation='flash_attention_2', trust_remote_code=True, use_safetensors=True)
model = model.eval().cuda().to(torch.bfloat16)

# prompt = "<image>\nFree OCR. "
# prompt = "<image>\n<|grounding|>Convert the document to markdown. "
# image_file = '/home/ngthbinh/Desktop/project/deepseek_ocr_demo/testocr.png'
# output_path = '/home/ngthbinh/Desktop/project/deepseek_ocr_demo'

# res = model.infer(tokenizer, prompt=prompt, image_file=image_file, output_path = output_path, base_size = 1024, image_size = 640, crop_mode=True, save_results = True, test_compress = True)

# nohup uvicorn app:app --host 0.0.0.0 --port 1337 --reload > app.log 2>&1 &
app = FastAPI()

@app.get("/status")
def get_status():
    return {
        "Status": "Online"
    }
    
@app.get("/demo_ocr")
def get_demo_ocr():
    prompt = "<image>\n<|grounding|>Convert the document to markdown. "
    image_file = '/home/ngthbinh/Desktop/project/deepseek_ocr_demo/testocr.png'
    output_path = '/home/ngthbinh/Desktop/project/deepseek_ocr_demo'

    res = model.infer(tokenizer, 
                      prompt=prompt, 
                      image_file=image_file, 
                      output_path = output_path, 
                      base_size = 1024, 
                      image_size = 640, 
                      crop_mode=True, 
                      save_results = True, 
                      test_compress = True)
    with open(output_path + "/result.mmd", "r", encoding="utf-8") as f:
        text = f.read().strip()
        
    return {
        "text": text
    }
  
def run_ocr_inference_sync(image_data: bytes, original_filename: str, output_dir: str) -> str:
    """
    This is a synchronous (blocking) function that runs the OCR model.
    We run it in a thread pool to avoid blocking the main server.
    """
    temp_input_path = None
    try:
        # 1. Create a temporary file and write the uploaded image data to it
        # We use a non-deleting temp file to get a stable path
        with tempfile.NamedTemporaryFile(delete=False, suffix=original_filename) as temp_file:
            temp_file.write(image_data)
            temp_input_path = temp_file.name

        # 2. Define prompt and run the blocking inference
        prompt = "<image>\n<|grounding|>Convert the document to markdown. "
        
        print(f"Running inference on {temp_input_path}...")
        model.infer(tokenizer,
                    prompt=prompt,
                    image_file=temp_input_path,  # Use the temp file path
                    output_path=output_dir,      # Save results to temp output dir
                    base_size=1024,
                    image_size=640,
                    crop_mode=True,
                    save_results=True,
                    test_compress=True)

        # 3. Read the result from the temporary output directory
        result_file_path = os.path.join(output_dir, "result.mmd")
        
        if not os.path.exists(result_file_path):
            raise FileNotFoundError("OCR model did not produce an output file (result.mmd).")

        with open(result_file_path, "r", encoding="utf-8") as f:
            text = f.read().strip()
            
        print("Inference complete.")
        return text

    finally:
        # 4. Clean up: Always delete the temporary input file
        if temp_input_path and os.path.exists(temp_input_path):
            os.remove(temp_input_path)
            print(f"Cleaned up {temp_input_path}")
              
@app.post("/uploadfile/")
async def create_upload_file(file: UploadFile):
    """
    Uploads a file, saves it temporarily, runs DeepSeek OCR,
    and returns the extracted markdown text.
    """
    if not file:
        return {"error": "No file provided"}

    # 1. Read file contents into memory asynchronously
    # This part is non-blocking
    try:
        image_bytes = await file.read()
    except Exception as e:
        return {"error": f"Failed to read file: {str(e)}"}

    # 2. Create a temporary directory to store the model's *output*
    # The 'with' statement handles cleanup automatically
    with tempfile.TemporaryDirectory() as output_dir:
        try:
            # 3. Run the HEAVY, BLOCKING function in a separate thread
            # This is the most important part for performance.
            text_result = await run_in_threadpool(
                run_ocr_inference_sync,
                image_data=image_bytes,
                original_filename=file.filename,
                output_dir=output_dir
            )
            
            # 4. Return the result
            return {"text": text_result}
        
        except FileNotFoundError as e:
            return {"error": str(e)}
        except Exception as e:
            # Catch errors from the inference function
            print(f"Error during inference: {e}")
            return {"error": f"An error occurred during OCR processing: {str(e)}"}