import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from typing import List, Optional, Dict, Any
import logging
from functools import lru_cache

class NeuralReasoning:
    def __init__(self, model_name: str = 'gpt2', max_length: int = 150, device: Optional[str] = None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.model = GPT2LMHeadModel.from_pretrained(model_name).to(self.device)
        self.max_length = max_length
        self.logger = self._setup_logger()

    def _setup_logger(self):
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.DEBUG)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        return logger

    @lru_cache(maxsize=100)
    def _generate_text(self, prompt: str, max_new_tokens: int = 100) -> str:
        self.logger.debug(f"Generating text for prompt: {prompt[:50]}...")
        inputs = self.tokenizer.encode(prompt, return_tensors='pt').to(self.device)
        try:
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    max_new_tokens=max_new_tokens,
                    num_return_sequences=1,
                    no_repeat_ngram_size=2,
                    do_sample=True,
                    temperature=0.7,
                    top_k=50,
                    top_p=0.95
                )
            return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        except Exception as e:
            self.logger.error(f"Error generating text: {str(e)}")
            return ""

    def parse_problem(self, problem_statement: str) -> str:
        prompt = (
            "Given the following mathematical problem, rephrase it concisely and precisely, "
            "focusing on the key mathematical elements:\n\n"
            f"{problem_statement}\n\nConcise rephrasing:"
        )
        return self._generate_text(prompt)

    def generate_hypothesis(self, parsed_problem: str) -> str:
        prompt = (
            "Based on the following mathematical problem, suggest a detailed solution approach. "
            "Include specific mathematical techniques or theorems that could be applied:\n\n"
            f"{parsed_problem}\n\nSolution approach:"
        )
        return self._generate_text(prompt)

    def generate_solution_steps(self, hypothesis: str, num_steps: int = 3) -> List[str]:
        prompt = (
            f"Using the following solution approach:\n{hypothesis}\n\n"
            f"Generate {num_steps} detailed, step-by-step solution steps. "
            "Each step should be clear, precise, and build upon the previous ones:"
        )
        solution_text = self._generate_text(prompt, max_new_tokens=200)
        steps = [step.strip() for step in solution_text.split('\n') if step.strip()]
        return steps[:num_steps]  # Ensure we return exactly num_steps

    def summarize_solution(self, solution_steps: List[str]) -> str:
        steps_text = "\n".join(f"{i+1}. {step}" for i, step in enumerate(solution_steps))
        prompt = (
            "Summarize the following solution steps into a concise, yet comprehensive explanation. "
            "Focus on the key mathematical insights and results:\n\n"
            f"{steps_text}\n\nSummary:"
        )
        return self._generate_text(prompt)

    def solve_problem(self, problem_statement: str, num_steps: int = 3) -> Dict[str, Any]:
        self.logger.info(f"Solving problem: {problem_statement}")
        try:
            parsed_problem = self.parse_problem(problem_statement)
            hypothesis = self.generate_hypothesis(parsed_problem)
            solution_steps = self.generate_solution_steps(hypothesis, num_steps)
            summary = self.summarize_solution(solution_steps)

            return {
                "status": "success",
                "original_problem": problem_statement,
                "parsed_problem": parsed_problem,
                "hypothesis": hypothesis,
                "solution_steps": solution_steps,
                "summary": summary
            }
        except Exception as e:
            self.logger.error(f"Error solving problem: {str(e)}")
            return {
                "status": "error",
                "original_problem": problem_statement,
                "error_message": str(e)
            }

    def clear_cache(self):
        self._generate_text.cache_clear()
        self.logger.info("Text generation cache cleared")