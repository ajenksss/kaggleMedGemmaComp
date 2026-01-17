import random
import os
import json

try:
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    HAS_TRANSFORMERS = True
except (ImportError, OSError): # Catch broken DLLs or missing libs
    HAS_TRANSFORMERS = False

class RealInferenceEngine:
    def __init__(self, model_id="google/gemma-2b-it"): 
        self.model_id = model_id
        self.tokenizer = None
        self.model = None
        self.is_loaded = False
        self.load_error = None
        
    def load_model(self):
        if not HAS_TRANSFORMERS: 
            self.load_error = "Transformers lib not found."
            return False
            
        if self.is_loaded: return True # Caching: Already loaded
        
        try:
            print(f"Loading {self.model_id}...")
            # Check for generic OOM by grabbing small memory first
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id, 
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else "cpu"
            )
            self.is_loaded = True
            self.load_error = None
            print("MedGemma Loaded Successfully.")
            return True
        except Exception as e:
            self.load_error = f"Model Load Failed: {str(e)}"
            print(self.load_error)
            return False

    def generate(self, prompt):
        if not self.is_loaded: return f"[Error: {self.load_error}]"
        
        try:
            input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids
            if torch.cuda.is_available(): input_ids = input_ids.to("cuda")
            
            outputs = self.model.generate(
                input_ids, 
                max_new_tokens=256, # Increased for structured JSON
                do_sample=True, 
                temperature=0.4 # Low temp for strict JSON
            )
            return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        except Exception as e:
            return f"[Inference Error: {str(e)}]"

class MedGemmaAgent:
    def __init__(self):
        """
        Layer 4: MedGemma Agent - The "Triage Copilot".
        Focuses on Ambiguity Resolution and Structured Rationale.
        """
        self.real_engine = RealInferenceEngine()
        self.use_real_model = False 

    def construct_prompt(self, risk_score, physics_valid, formula_explanation, shape_desc="Unknown", vitals_snapshot="BP Normal"):
        """
        Constructs the 'Triage Copilot' System Prompt.
        """
        system_prompt = (
            "You are MedGemma, a Triage Copilot for the ICU.\n"
            "Your Goal: Identify 'Abnormal Meaning' (Risk) even when vitals are normal.\n"
            "Rules:\n"
            "1. NEVER make a definitive diagnosis. Use 'Concern for...', 'Suggest checking...'.\n"
            "2. If Physics is Invalid, flag a Sensor Error.\n"
            "3. If Risk > 80% or Shape is 'Exploding', flag 'Compensated Shock' concern.\n"
            "4. OUTPUT JSON ONLY with these keys: 'risk_state' (Green/Yellow/Orange/Red), 'conflict', 'rationale', 'suggested_checks'."
        )
        
        user_prompt = f"""
        PATIENT DATA:
        - Snapshot Vitals: {vitals_snapshot}
        - TDA Topology: {shape_desc} (Radius variation over 10m)
        - Physics Check: {physics_valid}
        - Hemodynamic Risk Score: {risk_score:.2f}
        
        INSTRUCTION:
        Analyze for compensated shock (Normal Vitals + Bad Shape).
        Provide structured JSON triage assessment.
        """
        return system_prompt + "\n" + user_prompt

    def evaluate(self, risk_score, physics_valid, formula_explanation, run_real_inference=False, shape_desc="Normal Manifold", vitals_snapshot="Stable"):
        """
        Decides the output. Returns structured dictionary.
        """
        prompt_trace = self.construct_prompt(risk_score, physics_valid, formula_explanation, shape_desc, vitals_snapshot)
        
        response = {
            "risk_state": "GREEN",
            "conflict": "None",
            "rationale": "Vitals and Physiology are stable.",
            "suggested_checks": "Continue monitoring.",
            "inference_mode": "SIMULATION"
        }

        # REAL INFERENCE
        if run_real_inference:
            if not self.real_engine.is_loaded:
                self.real_engine.load_model()
            
            if self.real_engine.is_loaded:
                real_text = self.real_engine.generate(prompt_trace)
                
                # Robust JSON Extraction
                try:
                    start = real_text.find('{')
                    end = real_text.rfind('}')
                    if start != -1 and end != -1:
                        json_str = real_text[start:end+1]
                        data = json.loads(json_str)
                        response.update(data)
                        response["inference_mode"] = f"REAL {self.real_engine.model_id}"
                    else:
                        raise ValueError("No JSON found")
                except Exception:
                    # Fallback if model fails to output JSON
                    response["rationale"] = f"Raw Output: {real_text[:200]}..."
                    response["inference_mode"] = "REAL (JSON PARSE FAIL)"
                    
                return response
            else:
                 response["inference_mode"] = f"SIMULATION (Load Failed)"

        # FALLBACK / SIMULATION LOGIC (The "Safe" Copilot)
        
        # 1. Physics Veto
        if not physics_valid:
            response["risk_state"] = "YELLOW"
            response["conflict"] = "Physics Violation Detected"
            response["rationale"] = "Hemodynamic values violate Navier-Stokes constraints. Likely sensor artifact."
            response["suggested_checks"] = "Check BP cuff placement, flush A-line."
            return response

        # 2. Compensated Shock Logic (The "Meaning Bridge")
        if risk_score > 0.8:
            response["risk_state"] = "RED"
            response["conflict"] = f"Stable Vitals vs Exploding Topology ({shape_desc})"
            response["rationale"] = "Rapid topological expansion suggests physiological decoherence despite normal BP. Concern for Compensated Shock."
            response["suggested_checks"] = "Lactate, Urine Output, CRT. Re-assess perfusion."
        elif risk_score > 0.5:
            response["risk_state"] = "ORANGE"
            response["conflict"] = "Rising Volatility"
            response["rationale"] = "Early warning signs of instability detected in manifold shape."
            response["suggested_checks"] = "Increase monitoring frequency to q15m."
            
        return response

if __name__ == "__main__":
    agent = MedGemmaAgent()
    print(agent.evaluate(0.9, True, "Test", run_real_inference=False, shape_desc="Radius 4.5", vitals_snapshot="BP 110/70"))
