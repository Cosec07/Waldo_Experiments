# Waldo_Experiments

1) Waldo.py - Plots the attention outputs of the LLM head of each layer onto the input image (attention_over_image.py)(I have uploaded this code on OneNote)
2) attention_over_image.py - Visualizes the attention to image tokens by the Vicuna LLM head. Saves the attention values in a CSV file and the images.
3) causal_patching.py - Performs causal (hidden-state activation patching) on hidden states in FigStep Run by Hidden states in Benign image run. Returns a causal patch plot and logprobs.
4) Gradient Probe- Identifies which Language Model layers contribute most to the jailbreak output, similar to causal_patching, but it has access to gradients. Returns the LogProb for the target token in our case "Pornography" and the Layers of the Language Model Head.
5) Token_attribution - Simplest and the most basic Gradient attribution. Performs a Forward Step, then a backprop step, and then calculates the attribution of a specific patch in the image w.r.t to the gradient. So it's gradients x inputs. Essentially, highlights which input tokens or image patches contribute most to harmful token generation for logprob of token "pornography" in the output.
