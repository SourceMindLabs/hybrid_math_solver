import sympy as sp
from sympy import diff, integrate, solve, simplify, expand, factor, limit, series, Matrix
from typing import Union, List, Tuple, Optional , Dict

class SymbolicReasoning:
    def __init__(self):
        self.symbols = {}

    def create_symbol(self, name: str) -> sp.Symbol:
        symbol = sp.Symbol(name)
        self.symbols[name] = symbol
        return symbol

    def parse_to_symbolic(self, expr_str: str) -> sp.Expr:
        return sp.sympify(expr_str, locals=self.symbols)

    def solve_equation(self, equation: Union[sp.Eq, str], symbol: Optional[Union[sp.Symbol, str]] = None) -> List[sp.Expr]:
        if isinstance(equation, str):
            equation = self.parse_to_symbolic(equation)
        if isinstance(symbol, str):
            symbol = self.symbols.get(symbol) or self.create_symbol(symbol)
        return solve(equation, symbol)

    def simplify_expression(self, expression: Union[sp.Expr, str]) -> sp.Expr:
        if isinstance(expression, str):
            expression = self.parse_to_symbolic(expression)
        return simplify(expression)

    def verify_solution(self, equation: Union[sp.Eq, str], solution: Union[sp.Expr, float], symbol: Union[sp.Symbol, str] = 'x') -> bool:
        if isinstance(equation, str):
            equation = self.parse_to_symbolic(equation)
        if isinstance(symbol, str):
            symbol = self.symbols.get(symbol) or self.create_symbol(symbol)
        return equation.subs(symbol, solution).evalf() == 0

    def integrate_function(self, function: Union[sp.Expr, str], variable: Union[sp.Symbol, str]) -> sp.Expr:
        if isinstance(function, str):
            function = self.parse_to_symbolic(function)
        if isinstance(variable, str):
            variable = self.symbols.get(variable) or self.create_symbol(variable)
        return integrate(function, variable)

    def differentiate_function(self, function: Union[sp.Expr, str], variable: Union[sp.Symbol, str], order: int = 1) -> sp.Expr:
        if isinstance(function, str):
            function = self.parse_to_symbolic(function)
        if isinstance(variable, str):
            variable = self.symbols.get(variable) or self.create_symbol(variable)
        return diff(function, variable, order)

    def factor_expression(self, expression: Union[sp.Expr, str]) -> sp.Expr:
        if isinstance(expression, str):
            expression = self.parse_to_symbolic(expression)
        return factor(expression)

    def expand_expression(self, expression: Union[sp.Expr, str]) -> sp.Expr:
        if isinstance(expression, str):
            expression = self.parse_to_symbolic(expression)
        return expand(expression)

    def compute_limit(self, expression: Union[sp.Expr, str], variable: Union[sp.Symbol, str], point: Union[sp.Expr, float]) -> sp.Expr:
        if isinstance(expression, str):
            expression = self.parse_to_symbolic(expression)
        if isinstance(variable, str):
            variable = self.symbols.get(variable) or self.create_symbol(variable)
        return limit(expression, variable, point)

    def compute_series(self, expression: Union[sp.Expr, str], variable: Union[sp.Symbol, str], point: Union[sp.Expr, float], order: int) -> sp.Expr:
        if isinstance(expression, str):
            expression = self.parse_to_symbolic(expression)
        if isinstance(variable, str):
            variable = self.symbols.get(variable) or self.create_symbol(variable)
        return series(expression, variable, point, order)

    def solve_linear_system(self, equations: List[Union[sp.Eq, str]], variables: List[Union[sp.Symbol, str]]) -> Dict[sp.Symbol, sp.Expr]:
        parsed_eqs = [eq if isinstance(eq, sp.Eq) else self.parse_to_symbolic(eq) for eq in equations]
        parsed_vars = [var if isinstance(var, sp.Symbol) else self.create_symbol(var) for var in variables]
        return solve(parsed_eqs, parsed_vars)

    def compute_eigenvalues(self, matrix: Union[Matrix, List[List[float]]]) -> List[sp.Expr]:
        if isinstance(matrix, list):
            matrix = Matrix(matrix)
        return matrix.eigenvals()

    def compute_eigenvectors(self, matrix: Union[Matrix, List[List[float]]]) -> List[Tuple[sp.Expr, int, List[Matrix]]]:
        if isinstance(matrix, list):
            matrix = Matrix(matrix)
        return matrix.eigenvects()