import os
import torch
from typing import Optional, Dict, Any, AsyncGenerator
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    BitsAndBytesConfig,
    TextStreamer
)
from huggingface_hub import snapshot_download
from loguru import logger
from config.settings import settings


class LLMModelManager:
    """Manages the local LLM model for text generation."""
    
    def __init__(self):
        self.model_name = settings.LLM_MODEL_NAME
        self.model_path = settings.LLM_MODEL_PATH
        self.model = None
        self.tokenizer = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
    async def download_model(self) -> bool:
        """Download the model from Hugging Face if not already present."""
        try:
            logger.info(f"Downloading model {self.model_name}")
            
            # Create model directory if it doesn't exist
            os.makedirs(self.model_path, exist_ok=True)
            
            # Download model
            snapshot_download(
                repo_id=self.model_name,
                local_dir=self.model_path,
                resume_download=True,
                local_dir_use_symlinks=False
            )
            
            logger.info(f"Model {self.model_name} downloaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error downloading model: {str(e)}")
            return False
    
    async def load_model(self) -> bool:
        """Load the model and tokenizer."""
        try:
            logger.info(f"Loading model from {self.model_path}")
            
            # Check if model exists locally, download if not
            if not os.path.exists(self.model_path) or not os.listdir(self.model_path):
                logger.info("Model not found locally, downloading...")
                if not await self.download_model():
                    return False
            
            # Configure quantization for memory efficiency
            quantization_config = None
            if torch.cuda.is_available():
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                )
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                trust_remote_code=True,
                padding_side="left"
            )
            
            # Add pad token if not present
            if self.tokenizer.pad_token is None:
                # Set pad token to unk_token if available, otherwise use eos_token
                if self.tokenizer.unk_token is not None:
                    self.tokenizer.pad_token = self.tokenizer.unk_token
                else:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                quantization_config=quantization_config,
                device_map="auto" if torch.cuda.is_available() else None,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
            
            logger.info(f"Model loaded successfully on {self.device}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            return False
    
    def _prepare_prompt(self, system_prompt: str, user_query: str, context: Optional[str] = None) -> str:
        """Prepare the prompt for the model."""
        if context:
            prompt = f"""<bos><start_of_turn>user
System: {system_prompt}

Context: {context}

User: {user_query}<end_of_turn>
<start_of_turn>model
"""
        else:
            prompt = f"""<bos><start_of_turn>user
System: {system_prompt}

User: {user_query}<end_of_turn>
<start_of_turn>model
"""
        return prompt
    
    async def generate_response(
        self, 
        user_query: str, 
        system_prompt: str,
        context: Optional[str] = None,
        max_tokens: int = None,
        temperature: float = None,
        stream: bool = False
    ) -> str:
        """Generate response from the model."""
        if not self.model or not self.tokenizer:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        max_tokens = max_tokens or settings.MAX_TOKENS
        temperature = temperature or settings.TEMPERATURE
        
        prompt = self._prepare_prompt(system_prompt, user_query, context)
        
        # Tokenize input with attention mask
        inputs = self.tokenizer(
            prompt, 
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048  # Reasonable limit for context
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Generate
        with torch.no_grad():
            if stream:
                return self._generate_stream(inputs, max_tokens, temperature)
            else:
                outputs = self.model.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    top_p=settings.TOP_P,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
                
                # Decode response
                response = self.tokenizer.decode(
                    outputs[0][inputs["input_ids"].shape[1]:], 
                    skip_special_tokens=True
                )
                return response.strip()
    
    async def _generate_stream(
        self, 
        inputs: Dict[str, torch.Tensor], 
        max_tokens: int, 
        temperature: float
    ) -> AsyncGenerator[str, None]:
        """Generate streaming response."""
        streamer = TextStreamer(
            self.tokenizer, 
            skip_prompt=True, 
            skip_special_tokens=True
        )
        
        outputs = self.model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_p=settings.TOP_P,
            do_sample=True,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            streamer=streamer
        )
        
        # For now, return the complete response
        # In a real streaming implementation, you'd yield tokens as they're generated
        response = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:], 
            skip_special_tokens=True
        )
        yield response.strip()
    
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self.model is not None and self.tokenizer is not None
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        return {
            "model_name": self.model_name,
            "model_path": self.model_path,
            "device": self.device,
            "is_loaded": self.is_loaded(),
            "cuda_available": torch.cuda.is_available()
        }


# Global instance
llm_manager = LLMModelManager()
