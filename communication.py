import sympy as sp
from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication_application
from typing import Union, Dict, Any, List
import re
from sympy.physics.units import Unit, Quantity
from sympy.physics.units.systems import SI
from sympy.physics.units.systems.si import dimsys_SI

class CommunicationLayer:
    def __init__(self):
        self.transformations = standard_transformations + (implicit_multiplication_application,)
        self.symbol_map = {}
        self.unit_system = SI

    def neural_to_symbolic(self, neural_output: str) -> sp.Expr:
        cleaned_output = self.parse_and_clean(neural_output)
        try:
            symbolic_expr = parse_expr(cleaned_output, transformations=self.transformations, local_dict=self.symbol_map)
            return symbolic_expr
        except Exception as e:
            raise ValueError(f"Failed to convert neural output to symbolic expression: {e}")

    def symbolic_to_neural(self, symbolic_output: Union[sp.Expr, str]) -> str:
        if isinstance(symbolic_output, str):
            return symbolic_output
        try:
            neural_input = sp.pretty(symbolic_output)
            return neural_input
        except Exception as e:
            raise ValueError(f"Failed to convert symbolic output to neural input: {e}")

    def parse_and_clean(self, text: str) -> str:
        text = text.replace('^', '**')  # Convert caret to Python exponentiation
        text = re.sub(r'(\d+)([a-zA-Z])', r'\1*\2', text)  # Add multiplication sign between numbers and variables
        text = text.replace('=', '-')  # Convert equations to expressions
        text = re.sub(r'\b(and|AND)\b', '&', text)  # Convert 'and' to '&' for logical operations
        text = re.sub(r'\b(or|OR)\b', '|', text)  # Convert 'or' to '|' for logical operations
        text = re.sub(r'\b(not|NOT)\b', '~', text)  # Convert 'not' to '~' for logical operations
        return text

    def register_symbol(self, symbol_name: str, sympy_symbol: sp.Symbol) -> None:
        self.symbol_map[symbol_name] = sympy_symbol

    def get_registered_symbol(self, symbol_name: str) -> Union[sp.Symbol, None]:
        return self.symbol_map.get(symbol_name)

    def extract_equations(self, text: str) -> List[str]:
        equations = re.findall(r'([^=]+=[^=]+)', text)
        return [eq.strip() for eq in equations]

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
            raise ValueError(f"Failed to convert units: {e}")

    def format_solution(self, solution: Dict[sp.Symbol, Any]) -> str:
        formatted = []
        for var, value in solution.items():
            formatted.append(f"{var} = {value}")
        return ", ".join(formatted)

    def parse_inequality(self, inequality_str: str) -> sp.Rel:
        inequality_str = self.parse_and_clean(inequality_str)
        try:
            return sp.parsing.sympy_parser.parse_expr(inequality_str, transformations=self.transformations, evaluate=False)
        except Exception as e:
            raise ValueError(f"Failed to parse inequality: {e}")

    def symbolic_to_latex(self, symbolic_expr: Union[sp.Expr, str]) -> str:
        if isinstance(symbolic_expr, str):
            symbolic_expr = self.neural_to_symbolic(symbolic_expr)
        return sp.latex(symbolic_expr)

    def solve_system_of_equations(self, equations: List[Union[sp.Expr, str]]) -> Dict[sp.Symbol, Any]:
        symbolic_equations = [self.neural_to_symbolic(eq) if isinstance(eq, str) else eq for eq in equations]
        try:
            solution = sp.solve(symbolic_equations)
            return solution
        except Exception as e:
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
        A = sp.Matrix(matrix_A)
        b = sp.Matrix(vector_b)
        try:
            solution = A.LUsolve(b)
            return [float(sol) for sol in solution]
        except Exception as e:
            raise ValueError(f"Failed to solve linear system: {e}")