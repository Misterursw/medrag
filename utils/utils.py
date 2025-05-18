import os
import json
import torch
from llava.mm_utils import get_model_name_from_path
from llava.eval.run_llava import llava_chat, llava_eval_relevance
from qwenvl.run_qwenvl import qwen_chat, qwen_eval_relevance
from PIL import Image


@torch.no_grad()
def get_image_embedding(clip_processor, clip_vision_model, device, image_path, clip_type="clip"):
    """
    Loads an image, preprocesses it for CLIP, returns a normalized vision embedding.
    """
    image = Image.open(image_path).convert("RGB")
    if clip_type == "clip":
        inputs = clip_processor(images=image, return_tensors="pt").to(device)
        outputs = clip_vision_model(**inputs)
        emb = outputs.image_embeds  # [1, embed_dim]
    else:
        inputs = clip_processor(image).unsqueeze(0).to(device)
        emb = clip_vision_model.encode_image(inputs)

    emb = emb / emb.norm(dim=-1, keepdim=True)
    return emb.squeeze(0)  # shape: (embed_dim,)


@torch.no_grad()
def get_text_embedding(clip_tokenizer, clip_text_model, device, text, clip_type="clip"):
    """
    Tokenizes text, returns a normalized text embedding.
    """
    if clip_type == "clip":
        if isinstance(text, list):
            inputs = clip_tokenizer(text, padding=True, return_tensors="pt").to(device)
        else:
            inputs = clip_tokenizer([text], return_tensors="pt").to(device)
        outputs = clip_text_model(**inputs)
        emb = outputs.text_embeds  # [1, embed_dim]
    else:
        inputs = clip_tokenizer([text]).to(device)
        emb = clip_text_model.encode_text(inputs)
    emb = emb / emb.norm(dim=-1, keepdim=True)
    return emb.squeeze(0)  # shape: (embed_dim,)



def cal_relevance(model_path, image_path, question, model, tokenizer, image_processor):

    if "qwen-vl" in model_path.lower():
        prob = qwen_eval_relevance(image_path, question, model, tokenizer)
    else:
        args = type(
            "Args",
            (),
            {
                "model_path": model_path,
                "model_base": None,
                "model_name": get_model_name_from_path(model_path),
                "query": question,
                "conv_mode": None,
                "image_file": image_path,
                "sep": ",",
                "temperature": 0,
                "top_p": None,
                "num_beams": 1,
                "max_new_tokens": 512,
            },
        )()

        if "llava" in model_path:
            prob = llava_eval_relevance(args, tokenizer, model, image_processor)

    return prob


def infer(
    model_path,
    image_file,
    question,
    model,
    tokenizer,
    image_processor,
    from_array=False,
):
    if "webqa" in model_path:
        prompt_template = question
    else:
        prompt_template = (
            f"{question}\nAnswer the question using a single word or phrase."
        )

    if "qwen-vl" in model_path.lower():
        output = qwen_chat(image_file, prompt_template, model, tokenizer)
    else:
        args = type(
            "Args",
            (),
            {
                "model_path": model_path,
                "model_base": None,
                "model_name": get_model_name_from_path(model_path),
                "query": prompt_template,
                "conv_mode": None,
                "image_file": image_file,
                "sep": ",",
                "temperature": 0,
                "top_p": None,
                "num_beams": 1,
                "max_new_tokens": 512,
            },
        )()

        if "llava" in model_path:
            output = llava_chat(
                args,
                tokenizer,
                model,
                image_processor,
                from_array=from_array,
            )

    return output


def add_data(new_data, idx, item, gt_answers, questions, target_answer, poison_img_path, poisoned_captions, data):
    if isinstance(item, str):
        assert gt_answers is not None and questions is not None
        new_item = {
            "qid": item,
            "gt_answer": gt_answers[idx],
            "question": questions[idx],
            "poisoned_img_path": poison_img_path,
            "original_sample": data[item] 
        }
    else:
        new_item = {
            "qid": item["qid"],
            "gt_answer": gt_answers[idx],
            "question": questions[idx],
            "poisoned_img_path": poison_img_path,
            "original_sample": item["original_sample"] if "original_sample" in item else item
        }
    
    if isinstance(poisoned_captions, list):
        new_item.update({"poisoned_caption": poisoned_captions[idx]})
    else:
        new_item.update({"poisoned_caption": poisoned_captions})
    if target_answer is not None:
        new_item.update({"wrong_answer": target_answer})
    new_data.append(new_item)
    return new_data


def build_metadata(
        task, orig_metdata_path, output_dir_path, poison_img_paths, 
        target_answers=None, gt_answers=None, questions=None, poison_img_captions=None, poison_type='lpa-rt'
    ):
    os.makedirs(output_dir_path, exist_ok=True)

    file_path = orig_metdata_path
    with open(file_path, "r") as f:
        data = json.load(f)

    new_data = []
    if poison_type in ['lpa-rt', 'lpa-bb']:
        assert target_answers is not None and len(poison_img_paths) == len(target_answers)
        assert poison_img_captions is not None and len(poison_img_paths) == len(poison_img_captions)
        for idx, (item, poison_img_path, poison_caption, target_answer) in enumerate(zip(data, poison_img_paths, poison_img_captions, target_answers)):
            new_data = add_data(new_data, idx, item, gt_answers, questions, target_answer, poison_img_path, poison_caption, data)

    elif poison_type == 'gpa-rt':
        assert len(poison_img_paths) == 1 and len(poison_img_captions) == 1
        poison_img_path = poison_img_paths[0]
        poison_caption = poison_img_captions[0]
        target_answer = None
        for idx, item in enumerate(data):
            new_data = add_data(new_data, idx, item, gt_answers, questions, target_answer, poison_img_path, poison_caption, data)

    elif poison_type == "gpa-rtrrgen":
        assert len(poison_img_paths) == 1
        assert target_answers is not None and len(target_answers) == 1
        assert poison_img_captions is not None and len(poison_img_captions) == 1
        poison_img_path = poison_img_paths[0]
        poison_caption = poison_img_captions[0]
        target_answer = target_answers[0]
        for idx, item in enumerate(data):
            new_data = add_data(new_data, idx, item, gt_answers, questions, target_answer, poison_img_path, poison_caption, data)
    
    output_path = os.path.join(output_dir_path, f"{task}-{poison_type}.json")
    with open(output_path, 'w', encoding='utf-8') as file:
        json.dump(new_data, file, indent=4, ensure_ascii=False)
    
    print(f"Metadata is built in {output_path}.")
