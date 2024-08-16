from hybrid_model import HybridModel

def main():
    # Initialize the hybrid model
    hybrid_model = HybridModel()

    # Example problem statement
    problem_statement = "Solve the equation x^2 - 4x + 4 = 0"

    # Solve the problem
    solution_summary = hybrid_model.solve_and_explain(problem_statement)

    # Display the solution and explanation
    print("Problem Statement:", problem_statement)
    print("Solution and Explanation:", solution_summary)

if __name__ == "__main__":
    main()
