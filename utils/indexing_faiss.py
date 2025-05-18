import faiss
import numpy as np
from PIL import Image
import os
import torch
import json
from tqdm import tqdm
import argparse

device = "cuda" if torch.cuda.is_available() else "cpu"


# ------------- build_index -------------
def build_faiss_webqa(val_dataset, device, model, clip_type="clip", preprocess=None):
    embeddings = []
    index_to_image_id = {}
    count = 0
    for i in tqdm(val_dataset):
        datum = val_dataset[i]
        pos_imgs = datum["img_posFacts"]

        for j in range(len(pos_imgs)):
            image_id = pos_imgs[j]["image_id"]
            if image_id in index_to_image_id.values():
                continue
            # image_path = "../finetune/tasks/train_img/" + str(image_id) + ".png"
            image_path = "./finetune/tasks/WebQA_imgs/test/" + str(image_id) + ".png"
            if not os.path.exists(image_path):
                image_path = "./finetune/tasks/WebQA_imgs/train/" + str(image_id) + ".png"
                if not os.path.exists(image_path):
                    image_path =  "./finetune/tasks/WebQA_imgs/val/" + str(image_id) + ".png"
            assert os.path.exists(image_path)
                        
            with torch.no_grad():
                if clip_type == "clip":
                    image = preprocess(Image.open(image_path)).to(device)
                    image_embeddings = model.encode_image(torch.unsqueeze(image, dim=0))
                elif clip_type == "openclip":
                    image = preprocess(Image.open(image_path).convert("RGB")).to(device)
                    image_embeddings = model.encode_image(torch.unsqueeze(image, dim=0))
                elif "bge" in clip_type:
                    image_embeddings = model.encode(image=image_path)
                else:
                    pixel_values = preprocess(
                        images=Image.open(image_path).convert("RGB"),
                        return_tensors="pt",
                    ).pixel_values
                    pixel_values = pixel_values.to(torch.bfloat16).to(device)
                    image_embeddings = model.encode_image(
                        pixel_values, mode=clip_type
                    ).to(torch.float)

            combined_embedding = image_embeddings
            normalized_embedding = combined_embedding / combined_embedding.norm(
                dim=-1, keepdim=True
            )
            embeddings.append(normalized_embedding.cpu().numpy())

            index_to_image_id[count] = image_id
            count += 1

    embeddings = np.vstack(embeddings).astype("float32")

    # cosine similarity
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    return index, index_to_image_id


def build_faiss_mmqa(
    val_dataset, metadata, device, model, clip_type="clip", preprocess=None
):

    embeddings = []
    index_to_image_id = {}
    count = 0
    for datum in tqdm(val_dataset):
        pos_img = datum["supporting_context"][0]
        image_id = pos_img["doc_id"]
        if image_id in index_to_image_id.values():
            continue
        image_path = "../finetune/tasks/MMQA_imgs/" + metadata[image_id]["path"]

        with torch.no_grad():
            if clip_type == "clip":
                image = preprocess(Image.open(image_path)).to(device)
                image_embeddings = model.encode_image(torch.unsqueeze(image, dim=0))
            elif clip_type == "openclip":
                image = preprocess(Image.open(image_path).convert("RGB")).to(device)
                image_embeddings = model.encode_image(torch.unsqueeze(image, dim=0))
            elif "bge" in clip_type:
                image_embeddings = model.encode(image=image_path)
            else:
                pixel_values = preprocess(
                    images=Image.open(image_path).convert("RGB"),
                    return_tensors="pt",
                ).pixel_values
                pixel_values = pixel_values.to(torch.bfloat16).to(device)
                image_embeddings = model.encode_image(pixel_values, mode=clip_type).to(
                    torch.float
                )

        combined_embedding = image_embeddings
        normalized_embedding = combined_embedding / combined_embedding.norm(
            dim=-1, keepdim=True
        )
        embeddings.append(normalized_embedding.cpu().numpy())

        index_to_image_id[count] = image_id
        count += 1

    embeddings = np.vstack(embeddings).astype("float32")
    # cosine similarity
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    return index, index_to_image_id


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--topk", type=int, default=20)
    parser.add_argument("--datasets", type=str, default="MMQA")
    parser.add_argument("--clip_type", type=str, default="clip")
    args = parser.parse_args()

    model, preprocess, tokenizer = load_clip(args)

    if args.datasets == "WebQA":
        with open("./datasets/WebQA_test_image.json", "r") as f:
            val_dataset = json.load(f)
        index, index_to_image_id = build_faiss_webqa(
            val_dataset,
            device,
            model,
            clip_type=args.clip_type,
            preprocess=preprocess,
        )

    elif args.datasets == "MMQA":

        with open("datasets/MMQA_test_image.json", "r") as f:
            val_dataset = json.load(f)
        with open("datasets/MMQA_image_metadata.json", "r") as f:
            metadata = json.load(f)

        index, index_to_image_id = build_faiss_mmqa(
            val_dataset,
            metadata,
            device,
            model,
            clip_type=args.clip_type,
            preprocess=preprocess,
        )


    faiss.write_index(
        index,
        "../datasets/faiss_index/"
        + args.datasets
        + "_test_image_"
        + args.clip_type
        + ".index",
    )

