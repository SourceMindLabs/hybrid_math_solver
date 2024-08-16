from neural_reasoning import NeuralReasoning
from symbolic_reasoning import SymbolicReasoning
from communication import CommunicationLayer
from typing import Dict, Any, Union
import logging

class HybridModel:
    def __init__(self):
        self.neural_reasoning = NeuralReasoning()
        self.symbolic_reasoning = SymbolicReasoning()
        self.communication_layer = CommunicationLayer()
        self.logger = self._setup_logger()

    def _setup_logger(self):
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.DEBUG)
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        return logger

    def solve_problem(self, problem_statement: str) -> Dict[str, Any]:
        self.logger.info(f"Solving problem: {problem_statement}")
        
        parsed_problem = self.neural_reasoning.parse_problem(problem_statement)
        self.logger.debug(f"Parsed problem: {parsed_problem}")

        symbolic_expr = self.communication_layer.neural_to_symbolic(parsed_problem)
        self.logger.debug(f"Symbolic expression: {symbolic_expr}")

        try:
            solution = self.symbolic_reasoning.solve_equation(symbolic_expr)
            self.logger.debug(f"Solution found: {solution}")

            is_correct = self.symbolic_reasoning.verify_solution(symbolic_expr, solution)
            self.logger.debug(f"Solution verification: {'Correct' if is_correct else 'Incorrect'}")

            if is_correct:
                solution_summary = self.neural_reasoning.summarize_solution(str(solution))
                return {
                    "status": "success",
                    "solution": solution,
                    "summary": solution_summary,
                    "verified": True
                }
            else:
                return {
                    "status": "failure",
                    "reason": "Solution could not be verified",
                    "solution": solution,
                    "verified": False
                }
        except Exception as e:
            self.logger.error(f"Error solving problem: {str(e)}")
            return {
                "status": "error",
                "reason": str(e)
            }

    def solve_and_explain(self, problem_statement: str) -> Dict[str, Any]:
        self.logger.info(f"Solving and explaining problem: {problem_statement}")

        parsed_problem = self.neural_reasoning.parse_problem(problem_statement)
        hypothesis = self.neural_reasoning.generate_hypothesis(parsed_problem)
        self.logger.debug(f"Generated hypothesis: {hypothesis}")

        symbolic_expr = self.communication_layer.neural_to_symbolic(hypothesis)
        self.logger.debug(f"Symbolic expression from hypothesis: {symbolic_expr}")

        try:
            solution = self.symbolic_reasoning.solve_equation(symbolic_expr)
            explanation = self.neural_reasoning.summarize_solution(str(solution))
            steps = self.generate_solution_steps(symbolic_expr, solution)

            return {
                "status": "success",
                "solution": solution,
                "explanation": explanation,
                "steps": steps
            }
        except Exception as e:
            self.logger.error(f"Error in solve_and_explain: {str(e)}")
            return {
                "status": "error",
                "reason": str(e)
            }

    def generate_solution_steps(self, symbolic_expr: Union[str, Any], solution: Any) -> list:
        steps = []
        steps.append(f"1. Original expression: {symbolic_expr}")
        
        simplified_expr = self.symbolic_reasoning.simplify_expression(symbolic_expr)
        steps.append(f"2. Simplified expression: {simplified_expr}")
        
        if hasattr(self.symbolic_reasoning, 'factor_expression'):
            factored_expr = self.symbolic_reasoning.factor_expression(simplified_expr)
            steps.append(f"3. Factored expression: {factored_expr}")
        
        steps.append(f"4. Solution: {solution}")
        
        return steps

    def analyze_problem(self, problem_statement: str) -> Dict[str, Any]:
        parsed_problem = self.neural_reasoning.parse_problem(problem_statement)
        symbolic_expr = self.communication_layer.neural_to_symbolic(parsed_problem)

        analysis = {
            "original_problem": problem_statement,
            "parsed_problem": parsed_problem,
            "symbolic_expression": str(symbolic_expr),
            "simplified_expression": str(self.symbolic_reasoning.simplify_expression(symbolic_expr)),
        }

        if hasattr(self.symbolic_reasoning, 'compute_derivative'):
            analysis["derivative"] = str(self.symbolic_reasoning.differentiate_function(symbolic_expr, 'x'))

        if hasattr(self.symbolic_reasoning, 'compute_integral'):
            analysis["integral"] = str(self.symbolic_reasoning.integrate_function(symbolic_expr, 'x'))

        return analysis