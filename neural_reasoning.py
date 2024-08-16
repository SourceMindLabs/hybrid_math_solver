import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from typing import List, Optional

class NeuralReasoning:
    def __init__(self, model_name: str = 'gpt2', max_length: int = 150):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.model = GPT2LMHeadModel.from_pretrained(model_name).to(self.device)
        self.max_length = max_length

    def _generate_text(self, prompt: str) -> str:
        inputs = self.tokenizer.encode(prompt, return_tensors='pt').to(self.device)
        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                max_length=self.max_length,
                num_return_sequences=1,
                no_repeat_ngram_size=2,
                temperature=0.7,
                top_k=50,
                top_p=0.95
            )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def parse_problem(self, problem_statement: str) -> str:
        return self._generate_text(f"Parse and rephrase the following problem: {problem_statement}")

    def generate_hypothesis(self, parsed_problem: str) -> str:
        return self._generate_text(f"Given the problem: {parsed_problem}, suggest a solution approach.")

    def generate_solution_steps(self, hypothesis: str, num_steps: int = 3) -> List[str]:
        prompt = f"Based on the hypothesis: {hypothesis}, generate {num_steps} detailed solution steps."
        solution_text = self._generate_text(prompt)
        return [step.strip() for step in solution_text.split('\n') if step.strip()]

    def summarize_solution(self, solution_steps: List[str]) -> str:
        steps_text = "\n".join(solution_steps)
        return self._generate_text(f"Summarize the following solution steps:\n{steps_text}")

    def solve_problem(self, problem_statement: str) -> dict:
        parsed_problem = self.parse_problem(problem_statement)
        hypothesis = self.generate_hypothesis(parsed_problem)
        solution_steps = self.generate_solution_steps(hypothesis)
        summary = self.summarize_solution(solution_steps)

        return {
            "original_problem": problem_statement,
            "parsed_problem": parsed_problem,
            "hypothesis": hypothesis,
            "solution_steps": solution_steps,
            "summary": summary
        }