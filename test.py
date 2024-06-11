from sympy import sqrt, symbols

x = symbols("x", negative=True)
expr = sqrt(x**2)
print(expr)
