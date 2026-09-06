# ____________________________________________________________________________________
#
# Pyomo: Python Optimization Modeling Objects
# Copyright (c) 2008-2026 National Technology and Engineering Solutions of Sandia, LLC
# Under the terms of Contract DE-NA0003525 with National Technology and Engineering
# Solutions of Sandia, LLC, the U.S. Government retains certain rights in this
# software.  This software is distributed under the 3-clause BSD License.
# ____________________________________________________________________________________
"""Infrastructure shared between the pyomo.contrib.cp writers (currently
docplex_writer.py and cpsat_writer.py). Everything here is solver-agnostic:
it operates on Pyomo components and expressions only, and knows nothing
about docplex or OR-Tools.
"""

import itertools
from operator import attrgetter

from pyomo.common.collections import ComponentMap
from pyomo.contrib.cp.interval_var import IntervalVar
from pyomo.contrib.cp.sequence_var import SequenceVar
from pyomo.core.base import (
    Block,
    BooleanVar,
    Constraint,
    LogicalConstraint,
    Objective,
    Param,
    RangeSet,
    Set,
    Suffix,
    Var,
)
import pyomo.core.expr as EXPR
from pyomo.core.expr.visitor import StreamBasedExpressionVisitor, identify_variables
from pyomo.repn.util import categorize_valid_components

# FIXME: Remove the following as soon as non-active components no longer
# report active==True (see #3045)
from pyomo.network import Port


# A generic tag for an already-resolved, directly-usable value (a native
# Python constant, or a solver-native variable/expression object). This is
# the one tag that genuinely means the same thing regardless of which
# backend is writing the model, so it lives here instead of being defined
# separately (and identically) in each writer.
class _GENERAL:
    pass


# The set of component types that pyomo.contrib.cp's modeling layer can
# appear as, and the subset of those that a writer actually needs to walk.
# Both writers currently target exactly this component surface (they differ
# only in how they translate what they find, not in what they're willing to
# find), so the categorization step is shared.
_CP_TARGET_CTYPES = {Objective, Constraint, LogicalConstraint, IntervalVar, SequenceVar}
# categorize_valid_components() treats every target ctype as implicitly
# valid, and errors out if a ctype appears in both sets -- so the ctypes we
# only want to *tolerate* (not descend into and collect) go here, and must
# exclude anything already in _CP_TARGET_CTYPES.
_CP_VALID_CTYPES = {Block, Var, Param, BooleanVar, Suffix, Set, RangeSet, Port}


def categorize_cp_model(model, sort=None):
    """Collect the components of a pyomo.contrib.cp model that a CP writer
    needs to translate.

    Returns a dict mapping each of Objective, Constraint, LogicalConstraint,
    IntervalVar, and SequenceVar to the flat list of active component data
    objects of that type found on `model`. Raises ValueError if the model
    contains an active component of any other, unrecognized, type.
    """
    component_map, unrecognized = categorize_valid_components(
        model, active=True, sort=sort, valid=_CP_VALID_CTYPES, targets=_CP_TARGET_CTYPES
    )
    if unrecognized:
        raise ValueError(
            "The model ('%s') contains the following active components "
            "that this writer does not know how to process:\n\t%s"
            % (
                model.name,
                "\n\t".join(
                    "%s:\n\t\t%s" % (k, "\n\t\t".join(map(attrgetter('name'), v)))
                    for k, v in unrecognized.items()
                ),
            )
        )
    # categorize_valid_components' component_map maps ctype to the *blocks*
    # that contain a component of that type, not to the component data
    # objects themselves, so we still need to descend into each block to get
    # the actual list of things to translate.
    components = {ctype: [] for ctype in _CP_TARGET_CTYPES}
    for ctype, blocks in component_map.items():
        for block in blocks:
            components[ctype].extend(
                block.component_data_objects(
                    ctype, active=True, descend_into=False, sort=sort
                )
            )
    return components


def _check_var_domain(node, var):
    if not var.domain.isdiscrete():
        # Note: in the context of the current writer, this should be unreachable
        # because we can't handle non-discrete variables at all, so there will
        # already be errors handling the children of this expression.
        raise ValueError(
            "Variable indirection '%s' contains argument '%s', "
            "which is not a discrete variable" % (node, var)
        )
    bnds = var.bounds
    if None in bnds:
        raise ValueError(
            "Variable indirection '%s' contains argument '%s', "
            "which is not restricted to a finite discrete domain" % (node, var)
        )
    return var.domain & RangeSet(*bnds)


def getitem_arg_domain(node, i, arg_value):
    """Determine the finite discrete domain of the i-th indirection index
    argument of a GetItemExpression `node` (the IndexedComponent being
    accessed is `node.arg(0)`; `i` counts the remaining index arguments
    starting from 0, i.e. this is about `node.arg(i + 1)`).

    `arg_value` is that argument's already-resolved value if it is a plain
    constant (its class is in pyomo.core.expr.native_types); otherwise it is
    ignored, and the argument's domain is instead determined either by
    brute-force enumeration (if the argument is itself an expression: we
    can't rely on FBBT to tell us the domain is a regular, finite-step range,
    so we evaluate the expression over every combination of its variables'
    values) or by the bounds of a discrete Var.

    Returns a (domain, scale) pair: `domain` is a Pyomo Set enumerating the
    argument's possible values, and `scale` is either None (a plain constant
    argument contributes nothing to the position/index arithmetic a caller
    may want to build) or the (min, max, step) tuple describing the domain's
    regular structure, as returned by Set.get_interval().
    """
    if arg_value.__class__ in EXPR.native_types:
        arg_set = Set(initialize=[arg_value])
        arg_set.construct()
        return arg_set, None

    node_arg = node.arg(i + 1)
    if node_arg.is_expression_type():
        var_list = list(identify_variables(node_arg, include_fixed=False))
        var_domain = [list(_check_var_domain(node, v)) for v in var_list]
        arg_vals = set()
        for var_vals in itertools.product(*var_domain):
            for v, val in zip(var_list, var_vals):
                v.set_value(val)
            arg_vals.add(node_arg())
        arg_set = Set(initialize=sorted(arg_vals))
        arg_set.construct()
        interval = arg_set.get_interval()
        if not interval[2]:
            raise ValueError(
                "Variable indirection '%s' contains argument expression "
                "'%s' that does not evaluate to a simple discrete set"
                % (node, node_arg)
            )
        return arg_set, interval

    # This had better be a simple variable over a regular discrete domain.
    # When we add support for categorical variables, we will need to ensure
    # that the categoricals have already been converted to simple integer
    # domains by this point.
    arg_domain = _check_var_domain(node, node_arg)
    return arg_domain, arg_domain.get_interval()


def before_named_expression(visitor, child):
    _id = id(child)
    if _id not in visitor._named_expressions:
        return True, None
    return False, (_GENERAL, visitor._named_expressions[_id])


def handle_named_expression_node(visitor, node, expr):
    visitor._named_expressions[id(node)] = expr[1]
    return expr


class CPExpressionVisitorBase(StreamBasedExpressionVisitor):
    """Shared engine for the pyomo.contrib.cp writers' expression walkers.

    A concrete writer's visitor (e.g. LogicalToDoCplex, LogicalToCpSat)
    subclasses this and supplies two solver-specific dispatch tables as
    class (or instance) attributes:

      - `var_handles`: maps a leaf/non-expression Pyomo component class
        (Var, Param, IntervalVar, ...) to a function that creates (or looks
        up a memoized) native solver object for it.
      - `exit_node_dispatcher`: an ExitNodeDispatcher mapping a Pyomo
        expression node class to a function that builds the corresponding
        native solver constraint/expression from its already-processed
        children.

    and, optionally, `step_function_handles`: a dict of node classes whose
    translation needs to bypass the normal recursive walk (because they
    require custom, non-uniform handling of their own children) - consulted
    directly from `beforeChild` rather than through `exit_node_dispatcher`.
    """

    step_function_handles = {}

    def __init__(self, symbolic_solver_labels=False):
        self.symbolic_solver_labels = symbolic_solver_labels
        self._process_node = self._process_node_bx

        self.var_map = {}
        self._named_expressions = {}
        self.pyomo_to_native = ComponentMap()

    def initializeWalker(self, expr):
        expr, src, src_idx = expr
        walk, result = self.beforeChild(None, expr, 0)
        if not walk:
            return False, result
        return True, expr

    def beforeChild(self, node, child, child_idx):
        # Return native types
        if child.__class__ in EXPR.native_types:
            return False, (_GENERAL, child)

        if child.__class__ in self.step_function_handles:
            return self.step_function_handles[child.__class__](self, child)

        # Convert Vars/BooleanVars/etc. to their solver-native equivalents
        if not child.is_expression_type() or child.is_named_expression_type():
            return self.var_handles[child.__class__](self, child)

        return True, None

    def exitNode(self, node, data):
        return self.exit_node_dispatcher[node.__class__](self, node, *data)

    finalizeResult = None
