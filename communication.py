import sympy as sp
from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication_application, TokenError
from typing import Union, Dict, Any, List
import re
from sympy.physics.units import Unit, Quantity
from sympy.physics.units.systems import SI
from sympy.physics.units.systems.si import dimsys_SI
import logging

class CommunicationLayer:
    def __init__(self):
        self.transformations = standard_transformations + (implicit_multiplication_application,)
        self.symbol_map = {}
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

    def neural_to_symbolic(self, neural_output: str) -> sp.Expr:
        self.logger.debug(f"Converting to symbolic: {neural_output[:100]}...")
        
        # Try to find the original equation in the text
        equation_match = re.search(r'([x\^2+\-*/0-9\s]+=[x\^2+\-*/0-9\s]+)', neural_output)
        if equation_match:
            equation = equation_match.group(1)
            self.logger.debug(f"Found equation: {equation}")
            try:
                # Convert equation to expression by moving everything to one side
                left, right = equation.split('=')
                expr = f"({left})-({right})"
                self.logger.debug(f"Parsing expression: {expr}")
                return parse_expr(expr, transformations=self.transformations, local_dict=self.symbol_map)
            except Exception as e:
                self.logger.error(f"Failed to parse equation: {str(e)}")
        
        # If we can't find the equation, fall back to the original method
        cleaned_output = self.parse_and_clean(neural_output)
        self.logger.debug(f"Cleaned output: {cleaned_output}")

        try:
            math_expr = re.search(r'([x+\-*/^()0-9=\s]+)', cleaned_output)
            if math_expr:
                expr_to_parse = math_expr.group(1).replace('=', '-')
                self.logger.debug(f"Attempting to parse expression: {expr_to_parse}")
                return parse_expr(expr_to_parse, transformations=self.transformations, local_dict=self.symbol_map)
            else:
                raise ValueError("No valid mathematical expression found in the output")
        except Exception as e:
            self.logger.error(f"Failed to parse expression: {str(e)}")
            self.logger.warning("Returning default symbol 'x' as fallback")
            return sp.Symbol('x')

    def parse_and_clean(self, text: str) -> str:
        # Remove any text enclosed in parentheses (often explanatory text)
        text = re.sub(r'\([^)]*\)', '', text)
        text = text.replace('^', '**')  # Convert caret to Python exponentiation
        text = re.sub(r'(\d+)([a-zA-Z])', r'\1*\2', text)  # Add multiplication sign between numbers and variables
        # Remove any non-mathematical text, but keep '='
        text = re.sub(r'[a-zA-Z]+', '', text)
        return text.strip()

    def _is_valid_expression(self, expr: str) -> bool:
        # Basic validation to check if the expression looks valid
        if expr.count('(') != expr.count(')'):
            return False  # Unbalanced parentheses
        if re.search(r'[^0-9x+\-*/^()]', expr):
            return False  # Invalid characters
        if re.search(r'[\-+*/^]{2,}', expr):
            return False  # Consecutive operators
        return True



    def _is_valid_expression(self, expr: str) -> bool:
        # Basic validation to check if the expression looks valid
        if expr.count('(') != expr.count(')'):
            return False  # Unbalanced parentheses
        if re.search(r'[^0-9x+\-*/^()]', expr):
            return False  # Invalid characters
        if re.search(r'[\-+*/^]{2,}', expr):
            return False  # Consecutive operators
        return True

    
    def register_symbol(self, symbol_name: str, sympy_symbol: sp.Symbol) -> None:
        self.symbol_map[symbol_name] = sympy_symbol

    def get_registered_symbol(self, symbol_name: str) -> Union[sp.Symbol, None]:
        return self.symbol_map.get(symbol_name)

    def extract_equations(self, text: str) -> List[str]:
        return [eq.strip() for eq in re.findall(r'([^=]+=[^=]+)', text)]

    def extract_variables(self, expr: Union[sp.Expr, str]) -> set:
        if isinstance(expr, str):
            expr = self.neural_to_symbolic(expr)
        return expr.free_symbols

    def convert_units(self, value: float, from_unit: str, to_unit: str) -> float:
        try:
            from_quantity = Quantity(value * Unit(from_unit))
            to_quantity = from_quantity.convert_to(Unit(to_unit))
            return float(sp.N(to_quantity.evalf()))
        except Exception as e:
            self.logger.error(f"Failed to convert units: {e}")
            raise ValueError(f"Failed to convert units: {e}")

    def format_solution(self, solution: Dict[sp.Symbol, Any]) -> str:
        return ", ".join(f"{var} = {value}" for var, value in solution.items())

    def parse_inequality(self, inequality_str: str) -> sp.Rel:
        try:
            inequality_str = self.parse_and_clean(inequality_str)
            return sp.parsing.sympy_parser.parse_expr(inequality_str, transformations=self.transformations, evaluate=False)
        except Exception as e:
            self.logger.error(f"Failed to parse inequality: {e}")
            raise ValueError(f"Failed to parse inequality: {e}")

    def symbolic_to_latex(self, symbolic_expr: Union[sp.Expr, str]) -> str:
        if isinstance(symbolic_expr, str):
            symbolic_expr = self.neural_to_symbolic(symbolic_expr)
        return sp.latex(symbolic_expr)

    def solve_system_of_equations(self, equations: List[Union[sp.Expr, str]]) -> Dict[sp.Symbol, Any]:
        try:
            symbolic_equations = [self.neural_to_symbolic(eq) if isinstance(eq, str) else eq for eq in equations]
            return sp.solve(symbolic_equations)
        except Exception as e:
            self.logger.error(f"Failed to solve system of equations: {e}")
            raise ValueError(f"Failed to solve system of equations: {e}")

    def simplify_expression(self, expr: Union[sp.Expr, str]) -> sp.Expr:
        if isinstance(expr, str):
            expr = self.neural_to_symbolic(expr)
        return sp.simplify(expr)

    def factor_expression(self, expr: Union[sp.Expr, str]) -> sp.Expr:
        if isinstance(expr, str):
            expr = self.neural_to_symbolic(expr)
        return sp.factor(expr)

    def expand_expression(self, expr: Union[sp.Expr, str]) -> sp.Expr:
        if isinstance(expr, str):
            expr = self.neural_to_symbolic(expr)
        return sp.expand(expr)

    def calculate_derivative(self, expr: Union[sp.Expr, str], variable: str, order: int = 1) -> sp.Expr:
        if isinstance(expr, str):
            expr = self.neural_to_symbolic(expr)
        var = self.get_registered_symbol(variable) or sp.Symbol(variable)
        return sp.diff(expr, var, order)

    def calculate_integral(self, expr: Union[sp.Expr, str], variable: str) -> sp.Expr:
        if isinstance(expr, str):
            expr = self.neural_to_symbolic(expr)
        var = self.get_registered_symbol(variable) or sp.Symbol(variable)
        return sp.integrate(expr, var)

    def evaluate_limit(self, expr: Union[sp.Expr, str], variable: str, limit_point: Union[sp.Expr, float]) -> sp.Expr:
        if isinstance(expr, str):
            expr = self.neural_to_symbolic(expr)
        var = self.get_registered_symbol(variable) or sp.Symbol(variable)
        return sp.limit(expr, var, limit_point)

    def solve_ode(self, ode: Union[sp.Expr, str], func: sp.Function, var: str) -> sp.Expr:
        if isinstance(ode, str):
            ode = self.neural_to_symbolic(ode)
        variable = self.get_registered_symbol(var) or sp.Symbol(var)
        return sp.dsolve(ode, func(variable))

    def perform_series_expansion(self, expr: Union[sp.Expr, str], variable: str, point: Union[sp.Expr, float], order: int) -> sp.Expr:
        if isinstance(expr, str):
            expr = self.neural_to_symbolic(expr)
        var = self.get_registered_symbol(variable) or sp.Symbol(variable)
        return expr.series(var, point, order).removeO()

    def check_expression_equality(self, expr1: Union[sp.Expr, str], expr2: Union[sp.Expr, str]) -> bool:
        if isinstance(expr1, str):
            expr1 = self.neural_to_symbolic(expr1)
        if isinstance(expr2, str):
            expr2 = self.neural_to_symbolic(expr2)
        return sp.simplify(expr1 - expr2) == 0

    def convert_to_dimensionless(self, expr: Union[sp.Expr, str]) -> sp.Expr:
        if isinstance(expr, str):
            expr = self.neural_to_symbolic(expr)
        return expr.subs(dimsys_SI.get_dimensional_dependencies(expr))

    def solve_linear_system(self, matrix_A: List[List[float]], vector_b: List[float]) -> List[float]:
        try:
            A = sp.Matrix(matrix_A)
            b = sp.Matrix(vector_b)
            solution = A.LUsolve(b)
            return [float(sol) for sol in solution]
        except Exception as e:
            self.logger.error(f"Failed to solve linear system: {e}")
            raise ValueError(f"Failed to solve linear system: {e}")