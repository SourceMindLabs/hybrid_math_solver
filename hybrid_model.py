from neural_reasoning import NeuralReasoning
from symbolic_reasoning import SymbolicReasoning
from communication import CommunicationLayer
from typing import Dict, Any, Union, List
import logging
import sympy as sp

class HybridModel:
    def __init__(self):
        self.neural_reasoning = NeuralReasoning()
        self.symbolic_reasoning = SymbolicReasoning()
        self.communication_layer = CommunicationLayer()
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

    def solve_and_explain(self, problem_statement: str) -> Dict[str, Any]:
        self.logger.info(f"Solving and explaining problem: {problem_statement}")

        try:
            parsed_problem = self.neural_reasoning.parse_problem(problem_statement)
            self.logger.debug(f"Parsed problem: {parsed_problem}")

            hypothesis = self.neural_reasoning.generate_hypothesis(parsed_problem)
            self.logger.debug(f"Generated hypothesis: {hypothesis}")

            symbolic_expr = self.communication_layer.neural_to_symbolic(hypothesis)
            self.logger.debug(f"Symbolic expression: {symbolic_expr}")

            if isinstance(symbolic_expr, sp.Symbol) and str(symbolic_expr) == 'x':
                self.logger.warning("Could not extract valid equation from neural output")
                return self._create_error_response("Could not extract valid equation from neural output", problem_statement, parsed_problem, hypothesis)

            solution = self.symbolic_reasoning.solve_equation(symbolic_expr)
            self.logger.debug(f"Solution: {solution}")

            if not solution:
                return self._create_error_response("No solution found for the equation", problem_statement, parsed_problem, hypothesis)

            solution_steps = self.generate_solution_steps(symbolic_expr, solution)
            explanation = self.neural_reasoning.summarize_solution(solution_steps)
            self.logger.debug(f"Explanation: {explanation}")

            return {
                "status": "success",
                "original_problem": problem_statement,
                "parsed_problem": parsed_problem,
                "hypothesis": hypothesis,
                "symbolic_expression": str(symbolic_expr),
                "solution": str(solution),
                "solution_steps": solution_steps,
                "explanation": explanation
            }
        except Exception as e:
            self.logger.error(f"Error in solve_and_explain: {str(e)}", exc_info=True)
            return self._create_error_response(str(e), problem_statement)

    def _create_error_response(self, reason: str, problem_statement: str, parsed_problem: str = None, hypothesis: str = None) -> Dict[str, Any]:
        response = {
            "status": "error",
            "reason": reason,
            "original_problem": problem_statement
        }
        if parsed_problem:
            response["parsed_problem"] = parsed_problem
        if hypothesis:
            response["hypothesis"] = hypothesis
        return response

    def generate_solution_steps(self, symbolic_expr: Union[str, sp.Expr], solution: Any) -> List[str]:
        steps = []
        steps.append(f"1. Original expression: {symbolic_expr}")
        
        try:
            simplified_expr = self.symbolic_reasoning.simplify_expression(symbolic_expr)
            steps.append(f"2. Simplified expression: {simplified_expr}")
            
            if hasattr(self.symbolic_reasoning, 'factor_expression'):
                factored_expr = self.symbolic_reasoning.factor_expression(simplified_expr)
                steps.append(f"3. Factored expression: {factored_expr}")
            
            steps.append(f"4. Solution: {solution}")
            
            if isinstance(solution, list) and len(solution) > 0:
                for i, sol in enumerate(solution, start=1):
                    steps.append(f"   Solution {i}: x = {sol}")
            elif isinstance(solution, dict):
                for var, value in solution.items():
                    steps.append(f"   {var} = {value}")
        except Exception as e:
            self.logger.error(f"Error generating solution steps: {str(e)}", exc_info=True)
            steps.append(f"Error occurred while generating detailed steps: {str(e)}")
        
        return steps

    def analyze_problem(self, problem_statement: str) -> Dict[str, Any]:
        try:
            parsed_problem = self.neural_reasoning.parse_problem(problem_statement)
            symbolic_expr = self.communication_layer.neural_to_symbolic(parsed_problem)

            analysis = {
                "original_problem": problem_statement,
                "parsed_problem": parsed_problem,
                "symbolic_expression": str(symbolic_expr),
                "simplified_expression": str(self.symbolic_reasoning.simplify_expression(symbolic_expr)),
            }

            if hasattr(self.symbolic_reasoning, 'calculate_derivative'):
                analysis["derivative"] = str(self.symbolic_reasoning.calculate_derivative(symbolic_expr, 'x'))

            if hasattr(self.symbolic_reasoning, 'calculate_integral'):
                analysis["integral"] = str(self.symbolic_reasoning.calculate_integral(symbolic_expr, 'x'))

            return analysis
        except Exception as e:
            self.logger.error(f"Error in analyze_problem: {str(e)}", exc_info=True)
            return {
                "status": "error",
                "reason": str(e),
                "original_problem": problem_statement
            }