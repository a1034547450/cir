from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch
import base64
from local_datasets import FashionIQDataset
from torch.utils.data import DataLoader
import json 
import os 
import numpy as np 
import faiss 
from torch.utils.data import Subset

def get_embedding_idx(generated_ids_trimmed, embed_token_id):
    embedding_idx = []
    for out_ids in generated_ids_trimmed:
        embed_exist = False
        for j in range(len(out_ids)-1, -1, -1):
            if out_ids[j] == embed_token_id:
                embedding_idx.append(j)
                embed_exist = True
                break
        if not embed_exist:
            embedding_idx.append(-1)
    return embedding_idx

def process_image(image_path):
    with open(image_path, "rb") as f:
        encoded_image = base64.b64encode(f.read())
    encoded_image_text = encoded_image.decode("utf-8")
    base64_qwen = f"data:image;base64,{encoded_image_text}"
    return base64_qwen
def normalize_reps(reps):
    """批量归一化嵌入向量（适配余弦相似度搜索）"""
    return torch.nn.functional.normalize(reps, p=2, dim=-1)


def construct_ref_record(image,ref_text):
    prompt = '''Represent the above input text, images, videos, or any combination of the three as embeddings. 
    First output the thinking process in <think> </think> tags and then summarize the entire input in a word or sentence. 
    Finally, use the <gen_emb> tag to represent the entire input.'''
    single_message = {
        "role": "user",
        "content": [
            {"type": "image", "image": f"file://{image}"},
        #     {
        #     "type": "image_url",
        #     "image_url": {
        #         "url": process_image(image)
        #         # 'url':'None'
        #     }
        # },
            {"type": "text", "text": f"Find an image to match the fashion image and style note: {ref_text}\n<disc_emb>\n" + prompt},
        ],
    }
    return single_message 


def construct_single_image(image):
    prompt = '''Represent the above input text, images, videos, or any combination of the three as embeddings. 
    First output the thinking process in <think> </think> tags and then summarize the entire input in a word or sentence. 
    Finally, use the <gen_emb> tag to represent the entire input.'''
    single_message = {
        "role": "user",
        "content": [
            {
            "type": "image", "image": f"file://{image}"
            # "type": "image_url",
            # "image_url": {
            #     "url": process_image(image)
            #     # 'url':'None'
            # }
        },
            {"type": "text", "text": "Represent the given image.\n<disc_emb>\n" + prompt},
        ],
    }
    return single_message 



def generate_dataset_embeddings(model, processor, dataset, device, batch_size=8):
    """批量生成数据集所有样本的嵌入向量"""
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    all_embeddings = []
    all_meta_info = []  # 保存样本元信息，用于索引映射
    
    model.eval()
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            batch_meta = [] 
            print(f"Processing batch {batch_idx+1}/{len(dataloader)}")
            # 构建批量推理的messages
            messages = []
                # 索引目标图像的嵌入（相对检索任务中，目标图像是检索对象）
            for image,image_name in zip(batch["image_path"], batch["image_name"]):
                messages.append([construct_single_image(image)])
            # 预处理多模态输入

            texts = [processor.apply_chat_template(msg, tokenize=False,add_generation_prompt=True) for msg in messages]
            # texts = processor.apply_chat_template(
            #     messages, tokenize=False, add_generation_prompt=True
            # )
            # print(texts)
            image_inputs, _ = process_vision_info(messages)
            inputs = processor(
                text=texts,
                images=image_inputs,
                padding=True,
                return_tensors="pt"
            ).to(device)
            # 模型推理生成
            generated_output = model.generate(
                **inputs,
                max_new_tokens=1024,
                output_hidden_states=True,
                return_dict_in_generate=True,
                use_cache=True,
                pad_token_id=processor.tokenizer.pad_token_id,
                temperature=0.0  # 固定生成结果，保证嵌入稳定性
            )
            # 提取生成结果和隐藏状态
            generated_ids = generated_output.sequences
            hidden_states = generated_output.hidden_states

            # 裁剪输入前缀，只保留模型生成部分
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]

            output_texts = processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=False,  
                clean_up_tokenization_spaces=False
            )
            batch_meta_list = batch["image_name"]  
            for meta_name, gen_text in zip(batch_meta_list, output_texts):
                batch_meta.append({"generated_text":gen_text,'image_name':meta_name})  

            # 查找<gen_emb>对应的token位置
            embed_token_id = processor.tokenizer.get_vocab()["<gen_emb>"]
            embedding_idx_list = get_embedding_idx(generated_ids_trimmed, embed_token_id)
            # print(embedding_idx_list)
            # 提取每个样本的嵌入向量
            target_layer = -1  # 取模型最后一层隐藏状态
            for sample_idx_in_batch, step_idx in enumerate(embedding_idx_list):
                if step_idx == -1 or step_idx >= len(hidden_states):
                    # 生成失败时用零向量占位（可按需改为跳过样本）
                    zero_emb = torch.zeros(model.config.hidden_size,device='cpu').contiguous().view(1,-1)
                    all_embeddings.append(zero_emb)
                    print(f"⚠️ Batch {batch_idx} Sample {sample_idx_in_batch} 生成失败，使用零向量")
                else:
                    sample_emb = hidden_states[step_idx][target_layer][sample_idx_in_batch].to('cpu')
                    all_embeddings.append(sample_emb)
            # 保存样本元信息
            all_meta_info.extend(batch_meta)

    # 批量归一化并转换为numpy数组
    all_embeddings = torch.stack(all_embeddings, dim=0)
    all_embeddings = all_embeddings.to(dtype=torch.float32)
    all_embeddings  = all_embeddings.squeeze()
    print(all_embeddings.size())
    all_embeddings = normalize_reps(all_embeddings).numpy()
    
    return all_embeddings, all_meta_info

def save_index_and_meta(index, meta_info, save_dir):
    """保存FAISS索引和样本元信息映射"""
    os.makedirs(save_dir, exist_ok=True)

    faiss.write_index(index, os.path.join(save_dir, "faiss_index.bin"))
    with open(os.path.join(save_dir, "meta_info.json"), "w") as f:
        json.dump(meta_info, f, indent=2)

# ---------------------- 3. FAISS索引构建与保存/加载 ----------------------
def build_faiss_index(embeddings, use_gpu=False, nlist=100):
    """
    构建FAISS索引：
    - 小数据集（<10k）用IndexFlatL2（精确搜索）
    - 大数据集用IndexIVFFlat（近似搜索）
    """
    print(embeddings.shape)
    dim = embeddings.shape[1]
    if len(embeddings) < 10000:
        index = faiss.IndexFlatIP(dim)  # L2距离，归一化后等价于余弦相似度
    else:
        # IVF需要先训练聚类中心
        quantizer = faiss.IndexFlatL2(dim)
        index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_L2)
        index.train(embeddings)
    index.add(embeddings)

    # 迁移到GPU（可选）
    if use_gpu:
        res = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(res, 0, index)
    return index

def load_index_and_meta(load_dir, use_gpu=False):
    """加载FAISS索引和样本元信息映射"""
    index = faiss.read_index(os.path.join(load_dir, "faiss_index.bin"))
    if use_gpu:
        res = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(res, 0, index)
    with open(os.path.join(load_dir, "meta_info.json"), "r") as f:
        meta_info = json.load(f)
    return index, meta_info


# ---------------------- 4. 查询与相似搜索 ----------------------
def generate_query_embedding(query_input, model, processor, device):
    messages = [construct_ref_record(query_input["relative_captions"],query_input['reference_image_path'])]
    # 预处理输入
    texts = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, _ = process_vision_info(messages)
    inputs = processor(
        text=texts,
        images=image_inputs,
        padding=True,
        return_tensors="pt"
    ).to(device)

    # 模型推理
    model.eval()
    with torch.no_grad():
        generated_output = model.generate(
            **inputs,
            max_new_tokens=1024,
            output_hidden_states=True,
            return_dict_in_generate=True,
            use_cache=True,
            pad_token_id=processor.tokenizer.pad_token_id,
            temperature=0.0
        )

    # 提取嵌入向量
    generated_ids = generated_output.sequences
    hidden_states = generated_output.hidden_states

    generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
    output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=False, clean_up_tokenization_spaces=False
)

    embed_token_id = processor.tokenizer.get_vocab()["<gen_emb>"]
    embedding_idx = get_embedding_idx(generated_ids_trimmed, embed_token_id)[0]

    if embedding_idx == -1 or embedding_idx >= len(hidden_states):
        print("⚠️ 查询样本生成失败，返回零向量")
        query_emb = torch.zeros(model.config.hidden_size, device=device).cpu().numpy()
    else:
        target_layer = -1
        query_emb = hidden_states[embedding_idx][target_layer][0].cpu().numpy()
        query_emb = query_emb / np.linalg.norm(query_emb, ord=2)  # 归一化
    
    return query_emb,output_text



if __name__ == '__main__':
    dataset_dir = '/data/oss_bucket_0/lida/cir/data/fashionIQ'
    split ='val'
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    BATCH_SIZE = 128
    ckpt_dir = '/data/oss_bucket_0/lida/ckpt/UME-R1'
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        ckpt_dir,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="cuda",
        # 显存优化选项（可选）
        # gradient_checkpointing=True,
    )
    print('load_model finished')
    processor = AutoProcessor.from_pretrained(ckpt_dir)
    # 显式指定pad_token（避免生成时的对齐问题）\
    processor.tokenizer.padding_side = "left"
    processor.tokenizer.pad_token = processor.tokenizer.eos_token

    dress_types = ['shirt', 'toptee']
    for single_dress in dress_types:
        # qry_dataset = FashionIQDataset(dataset_dir, split, [single_dress], 'relative')
        tgt_dataset = FashionIQDataset(dataset_dir, split, [single_dress], 'classic')
        # tgt_dataset = Subset(tgt_dataset,list(range(300)))
        save_dir = f'/data/oss_bucket_0/lida/cir/store/{single_dress}'
        
        embeddings, meta_info = generate_dataset_embeddings(model, processor, tgt_dataset, DEVICE, BATCH_SIZE)
        faiss_index = build_faiss_index(embeddings, use_gpu=False)
        save_index_and_meta(faiss_index, meta_info, save_dir)
        # print("\nLoading index...")
        # faiss_index, meta_info = load_index_and_meta(save_dir)
