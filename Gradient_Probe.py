import torch, numpy as np
from transformers import AutoProcessor, AutoModelForVision2Seq, BitsAndBytesConfig
from PIL import Image
from pathlib import Path

def load_model():
    qcfg = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
                              bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True)
    model = AutoModelForVision2Seq.from_pretrained("llava-hf/llava-1.5-7b-hf",
        device_map="auto", torch_dtype=torch.float16, quantization_config=qcfg).eval()
    proc = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf", trust_remote_code=True)
    return model, proc

def hook_heads(model):
    head_out = {}
    def make_hook(i):
        def hook(module, inp, out):
            out.requires_grad_()
            out.retain_grad()
            head_out[i] = out
        return hook

    for i, blk in enumerate(model.language_model.model.layers):
        blk.self_attn.o_proj.register_forward_hook(make_hook(i))
    return head_out

def run(img_path, prompt, target_token, mode, max_tokens):
    """
    Identify which Vicuna attention heads contribute most to jailbreak output.
    Runs a forward+backward pass for a specific harmful token (e.g., "bomb")
    and uses gradient norms on each head's output to estimate influence.

    Also prints the model's full output text to confirm jailbreak activation.

    Modes:
    --mode image       → only image
    --mode text        → only text
    --mode multimodal  → image + text (default)
    """
    model, proc = load_model()

    # Build inputs based on mode
    image = Image.open(img_path).convert("RGB") if mode in ["image", "multimodal"] else None
    text  = prompt.strip() if mode in ["text", "multimodal"] else ""
    if mode == "multimodal":
        text += "\n<image>"

    batch = proc(images=image, text=text, return_tensors="pt").to(model.device)
    pixel_values = batch.pop("pixel_values") if "pixel_values" in batch else None

    head_out = hook_heads(model)

    # Run model.generate to verify jailbreak
    print("\n Generating full model response...")
    with torch.no_grad():
        gen_args = {**batch, "max_new_tokens": max_tokens}
        if pixel_values is not None:
            gen_args["pixel_values"] = pixel_values
        generated = model.generate(**gen_args)
        answer = proc.tokenizer.decode(generated[0], skip_special_tokens=True)
    print("\n🧾 MODEL OUTPUT:")
    print("----------------------")
    print(answer)
    print("----------------------")

    # Forward for attribution (with grads enabled)
    with torch.set_grad_enabled(True):
        fwd_args = {**batch, "output_hidden_states": False, "return_dict": True}
        if pixel_values is not None:
            fwd_args["pixel_values"] = pixel_values
        out = model(**fwd_args)

        logits = out.logits
        probs = torch.log_softmax(logits, dim=-1)

        # Use 2nd-to-last token before image as target
        target_id = proc.tokenizer.convert_tokens_to_ids(target_token)
        last_idx = logits.shape[1] - 2
        score = probs[0, last_idx, target_id]
        print(f"\nTarget token '{target_token}' logprob: {score.item():.4f}")

        # Backward
        model.zero_grad()
        score.backward()

    # Collect gradients
    grads = {}
    for layer, t in head_out.items():
        grads[layer] = t.grad.norm().item()

    # Sort + report
    top = sorted(grads.items(), key=lambda x: -x[1])
    print("\nTop influential Vicuna layers:")
    for layer, val in top[:10]:
        print(f"Layer {layer:2d}: grad norm = {val:.6f}")

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--image", required=False)
    p.add_argument("--prompt", required=False, default="")
    p.add_argument("--target-token", default="bomb")
    p.add_argument("--mode", choices=["image", "text", "multimodal"], default="multimodal")
    p.add_argument("--max-tokens", type=int, default=512)
    args = p.parse_args()
    run(args.image, args.prompt, args.target_token, args.mode, args.max_tokens)
