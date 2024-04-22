# This file is implemented by RZ.
# This file aims to accelerate the original evaluate logic using 'numba' package.
# You should install numba package in your Python environment or the later evaluation will fail.
from __future__ import annotations

import ast
from typing import Sequence


def add_import_package_statement(program: str, package_name: str, as_name=None, *, check_imported=True) -> str:
    """Add 'import package_name as as_name' in the program code.
    """
    tree = ast.parse(program)
    if check_imported:
        # check if 'import package_name' code exists
        package_imported = False
        for node in tree.body:
            if isinstance(node, ast.Import) and any(alias.name == package_name for alias in node.names):
                package_imported = True
                break

        if package_imported:
            return ast.unparse(tree)

    # add 'import package_name' to the top of the program
    import_node = ast.Import(names=[ast.alias(name=package_name, asname=as_name)])
    tree.body.insert(0, import_node)
    program = ast.unparse(tree)
    return program


def _add_numba_decorator(
        program: str,
        function_name: str
) -> str:
    # parse to syntax tree
    tree = ast.parse(program)

    # check if 'import numba' already exists
    numba_imported = False
    for node in tree.body:
        if isinstance(node, ast.Import) and any(alias.name == 'numba' for alias in node.names):
            numba_imported = True
            break

    # add 'import numba' to the top of the program
    if not numba_imported:
        import_node = ast.Import(names=[ast.alias(name='numba', asname=None)])
        tree.body.insert(0, import_node)

    # traverse the tree, and find the function_to_run
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == function_name:
            # the '@numba.jit()' decorator instance
            decorator = ast.Call(
                func=ast.Attribute(
                    value=ast.Name(id='numba', ctx=ast.Load()),
                    attr='jit',
                    ctx=ast.Load()
                ),
                args=[],  # args do not have argument name
                keywords=[ast.keyword(arg='nopython', value=ast.NameConstant(value=True))]
                # keywords have argument name
            )
            # add the decorator to the decorator_list of the node
            node.decorator_list.append(decorator)

    # turn the tree to string and return
    modified_program = ast.unparse(tree)
    return modified_program


def add_numba_decorator(
        program: str,
        function_name: str | Sequence[str],
) -> str:
    """
    This function aims to accelerate the evaluation of the searched code. This is achieved by decorating '@numba.jit()'
    to the function_to_evolve or other functions in the specification that can be speed up using numba.
    However, it should be noted that not all numpy functions support numba acceleration: such as np.piecewise().
    So use this function wisely. Hahaha!

    Example input program:
        def func(a: np.ndarray):
            return a * 2
    Example output program
        import numba

        numba.jit()
        def func(a: np.ndarray):
            return a * 2
    """
    if isinstance(function_name, str):
        return _add_numba_decorator(program, function_name)
    for f_name in function_name:
        program = _add_numba_decorator(program, f_name)
    return program


if __name__ == '__main__':
    code = '''
import numpy as np

def func1():
    return 3

def func():
    return 5
    '''
    res = add_numba_decorator(code, 'func')
    print(res)
    res = add_numba_decorator(code, ['func1', 'func'])
    print(res)
