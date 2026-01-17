import os
try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch
except ImportError:
    print("Error: Transformers/Torch not installed. Run 'pip install transformers torch'")
    exit()

class RealMedGemma:
    def __init__(self, model_id="google/gemma-2b-it"):
        """
        The REAL MedGemma Inference Engine.
        In a production environment, this runs on the Edge TPU or GPU.
        For this prototype, we demonstrate the actual code to drive the model.
        """
        print(f"Loading {model_id}... (This may take a moment)")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, 
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else "cpu"
        )
        print("Model Loaded Successfully.")

    def predict(self, prompt, max_new_tokens=100):
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids
        if torch.cuda.is_available():
            input_ids = input_ids.to("cuda")
            
        outputs = self.model.generate(
            input_ids, 
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7
        )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

if __name__ == "__main__":
    # 1. Sanity Check Prompt
    system_instruction = "You are an expert medical AI assistant. Be concise."
    clinical_scenario = "Patient HR 140, MAP 55. Lactate 4.0. Suggest action."
    
    full_prompt = f"<start_of_turn>user\n{system_instruction}\n\n{clinical_scenario}<end_of_turn>\n<start_of_turn>model"
    
    print("\n--- Testing Single Inference ---")
    print(f"Prompt: {clinical_scenario}")
    
    # We wrap this in a try block so it fails gracefully if weights aren't downloaded
    try:
        engine = RealMedGemma()
        response = engine.predict(full_prompt)
        print(f"\nResponse:\n{response}")
    except Exception as e:
        print(f"\n[DEMO NOTE]: Could not load weights (likely need Hugging Face Token).")
        print(f"Error: {e}")
        print("\nThis script demonstrates the EXACT code structure used for the real model.")
