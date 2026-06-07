import os
import torch
from PIL import Image
from mobileo.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from mobileo.model.builder import load_pretrained_model
from mobileo.mm_utils import tokenizer_image_token, process_images
from mobileo.conversation import conv_templates
from argparse import ArgumentParser
from hardware_scheduler import HardwareScheduler

parser = ArgumentParser()
parser.add_argument("--model_path", type=str, default="checkpoints/mobileo_unified_1.5B")
parser.add_argument("--image_path", type=str, default="assets/cute_cat.png")
parser.add_argument("--prompt", type=str, default="make the cat black")
args = parser.parse_args()

# ── Hardware-aware device selection ──────────────────────────────────────────
sched = HardwareScheduler()
sched.print_report()
cfg = sched.get_config()
device = cfg.torch_device
dtype  = cfg.torch_dtype

tokenizer, model, _ = load_pretrained_model(args.model_path)
model.to(device)
image_processor = model.get_vision_tower().image_processor


def infer(prompt, img_path):
    prompt = f"Please edit the provided image according to the following description: {prompt}"
    qs = DEFAULT_IMAGE_TOKEN + "\n" + prompt
    conv = conv_templates["qwen_2"].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)
    image_tensor = process_images([Image.open(img_path).convert("RGB")], image_processor, model.config)[0]
    output_image = model.generate_image(
        input_ids,
        pixel_values=image_tensor.unsqueeze(0).to(dtype),
    )
    return output_image[0]


def main():
    output_dir = "predictions"
    os.makedirs(output_dir, exist_ok=True)
    image_sana = infer(args.prompt, args.image_path)
    save_path = os.path.join(output_dir, "mobileo_edit.png")
    image_sana.save(save_path)
    print(f"Saved: {save_path}")


if __name__ == "__main__":
    main()
