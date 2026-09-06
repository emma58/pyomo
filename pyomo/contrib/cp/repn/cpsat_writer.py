# ____________________________________________________________________________________
#
# Pyomo: Python Optimization Modeling Objects
# Copyright (c) 2008-2026 National Technology and Engineering Solutions of Sandia, LLC
# Under the terms of Contract DE-NA0003525 with National Technology and Engineering
# Solutions of Sandia, LLC, the U.S. Government retains certain rights in this
# software.  This software is distributed under the 3-clause BSD License.
# ____________________________________________________________________________________
"""Writer and solver plugin for Google OR-Tools' CP-SAT solver
(ortools.sat.python.cp_model).

This mirrors the architecture of repn/docplex_writer.py (a
StreamBasedExpressionVisitor-based writer/solver pair), sharing the
solver-agnostic parts with it via repn/util.py, but the two differ in one
structural way that's worth calling out up front: CP Optimizer's docplex API
is *expression*-oriented (every relation, connective, etc. is a reusable
value you can nest anywhere), while CP-SAT's cp_model API is *statement*-
oriented (model.add(...)/model.add_bool_and(...)/etc. post a constraint;
they don't hand back something you can use as a Boolean value elsewhere).
That's the reason for the `_AUXILIARY` tag below: a Boolean-valued Pyomo
node's CP-SAT translation has to be decided lazily, based on whether it ends
up being asserted directly (cheap: no extra variable) or used as a value
nested inside something else (needs an auxiliary literal manufactured for
it via reification). This is the same "make an auxiliary variable that
stands for a logical statement's truth value" idea used in
transform/logical_to_disjunctive_walker.py's `z` variables -- the difference
is that walker always mints one, even at the root, because a disjunctive
program has no way to just assert a statement; CP-SAT can, so we only pay
for the auxiliary variable when one is actually needed.
"""

from pyomo.common.dependencies import attempt_import

import logging

from pyomo.common import DeveloperError
from pyomo.common.config import ConfigDict, ConfigValue

from pyomo.contrib.cp import IntervalVar
from pyomo.contrib.cp.interval_var import (
    IntervalVarStartTime,
    IntervalVarEndTime,
    IntervalVarPresence,
    IntervalVarLength,
    ScalarIntervalVar,
    IntervalVarData,
    IndexedIntervalVar,
)
from pyomo.contrib.cp.sequence_var import (
    SequenceVar,
    ScalarSequenceVar,
    SequenceVarData,
)
from pyomo.contrib.cp.scheduling_expr.scheduling_logic import (
    AlternativeExpression,
    SpanExpression,
    SynchronizeExpression,
)
from pyomo.contrib.cp.scheduling_expr.precedence_expressions import (
    BeforeExpression,
    AtExpression,
)
from pyomo.contrib.cp.scheduling_expr.sequence_expressions import (
    NoOverlapExpression,
    FirstInSequenceExpression,
    LastInSequenceExpression,
    BeforeInSequenceExpression,
    PredecessorToExpression,
)
from pyomo.contrib.cp.scheduling_expr.step_function_expressions import (
    AlwaysIn,
    StepAt,
    StepAtStart,
    StepAtEnd,
    Pulse,
    CumulativeFunction,
    NegatedStepFunction,
)
from pyomo.contrib.cp.repn.util import (
    _GENERAL,
    CPExpressionVisitorBase,
    before_named_expression as _before_named_expression,
    categorize_cp_model,
    getitem_arg_domain,
    handle_named_expression_node as _handle_named_expression_node,
)

from pyomo.core.base import (
    minimize,
    maximize,
    SortComponents,
    Objective,
    Constraint,
    Var,
    BooleanVar,
    LogicalConstraint,
    value,
)
from pyomo.core.base.boolean_var import (
    ScalarBooleanVar,
    BooleanVarData,
    IndexedBooleanVar,
)
from pyomo.core.base.expression import ScalarExpression, ExpressionData
from pyomo.core.base.param import IndexedParam, ScalarParam, ParamData
from pyomo.core.base.var import ScalarVar, VarData, IndexedVar
import pyomo.core.expr as EXPR
from pyomo.core.base.set import SetProduct
from pyomo.repn.util import ExitNodeDispatcher
from pyomo.opt import WriterFactory, SolverFactory, TerminationCondition, SolverResults

cp_model, cp_model_available = attempt_import('ortools.sat.python.cp_model')

logger = logging.getLogger('pyomo.contrib.cp')


# A Boolean-valued Pyomo node whose CP-SAT translation is a *statement*
# (model.add_bool_and(...), etc.), not a reusable value -- see the module
# docstring. Carries a lazy `(assert_fn, reify_fn)` pair: `assert_fn(visitor)`
# posts the statement directly (used when this node is the root of a
# LogicalConstraint, or an unconditional conjunct of one -- no auxiliary
# variable needed); `reify_fn(visitor, lit)` manufactures the auxiliary
# Boolean `lit` and posts the "lit <=> statement" constraints linking it
# (used when this node is nested as an operand inside something else).
class _AUXILIARY:
    pass


# A GetItemExpression on IntervalVars whose sub-attribute (.start_time vs
# .end_time vs ...) hasn't been picked yet by a GetAttrExpression, so the
# add_element(...) call can't be built yet.
class _DEFERRED_ELEMENT_CONSTRAINT:
    pass


def _presence_literal(cpsat_interval_var):
    # Every interval var this writer builds is created via
    # new_optional_interval_var with an explicit presence literal (even
    # "mandatory" ones, whose literal is just fixed to 1) -- see
    # _create_cpsat_interval_var -- so this list always has exactly one
    # element.
    return cpsat_interval_var.presence_literals()[0]


def _materialize_literal(visitor, payload):
    # Mint a fresh auxiliary Boolean and have the node's reify_fn tie it to
    # the node's actual truth value. Minting the variable itself never
    # depends on anything not yet known, so this is always safe to do
    # eagerly, even for the deferred sequencing constraints below whose
    # *content* isn't decided until every LogicalConstraint has been walked.
    _, reify_fn = payload
    lit = visitor.model.new_bool_var('')
    reify_fn(visitor, lit)
    return lit


def _get_int_expr(visitor, arg):
    kind, payload = arg
    if kind is _GENERAL:
        return payload
    if kind is _AUXILIARY:
        # Bools are ints in CP-SAT, so a literal is already usable wherever
        # an integer-valued expression is needed.
        return _materialize_literal(visitor, payload)
    if kind is _DEFERRED_ELEMENT_CONSTRAINT:
        raise DeveloperError(
            "A GetItemExpression over IntervalVars was used in a numeric "
            "context without first selecting an attribute (e.g. "
            "'.start_time') via a GetAttrExpression."
        )
    raise DeveloperError(
        "Attempting to get a CP-SAT integer-valued expression from "
        "object in class %s" % str(kind)
    )


def _get_literal(visitor, arg):
    kind, payload = arg
    if kind is _GENERAL:
        return payload
    if kind is _AUXILIARY:
        return _materialize_literal(visitor, payload)
    raise DeveloperError(
        "Attempting to get a CP-SAT Boolean-valued expression from "
        "object in class %s" % str(kind)
    )


def _assert_true(visitor, arg):
    kind, payload = arg
    if kind is _AUXILIARY:
        assert_fn, reify_fn = payload
        assert_fn(visitor)
    elif kind is _GENERAL:
        visitor.model.add(payload == 1)
    else:
        raise DeveloperError(
            "Attempting to assert a CP-SAT expression from "
            "object in class %s as a LogicalConstraint" % str(kind)
        )


def _bounds(elem):
    # elem is either a plain Python constant, or a CP-SAT IntVar/literal.
    if isinstance(elem, (int, float)):
        return elem, elem
    # NOTE: elem.proto.domain is a protobuf repeated-field wrapper, not a
    # plain Python list -- its negative indexing (e.g. dom[-1]) does not
    # behave like a list's and silently returns the wrong element, so
    # convert it to a real list first.
    dom = list(elem.proto.domain)
    return dom[0], dom[-1]


def _nm(name):
    # OR-Tools' new_*_var() factories want a str, not None, even for an
    # anonymous/unlabeled variable.
    return name if name is not None else ''


def _sub_name(name, suffix):
    # Build a per-sub-variable name (e.g. an IntervalVar's "_start") without
    # producing None when the base name itself is None (symbolic_solver_labels
    # is off, or the base component's name simply wasn't requested).
    return name + suffix if name else ''


##
# Leaf/component handlers
##


def _make_int_var(visitor, pyomo_var, name, what):
    # Shared by regular Vars and by an IntervalVar's start_time/end_time/
    # length sub-variables (all plain ScalarVars domained on Integers).
    if pyomo_var.fixed:
        return visitor.model.new_constant(int(value(pyomo_var)))
    lb, ub = pyomo_var.bounds
    if lb is None or ub is None:
        raise ValueError(
            "The CP-SAT writer requires finite bounds on every integer "
            "variable (unlike CP Optimizer, CP-SAT has no notion of an "
            "unbounded horizon/domain). Cannot write %s '%s' with bounds "
            "%s." % (what, pyomo_var.name, (lb, ub))
        )
    return visitor.model.new_int_var(int(lb), int(ub), _nm(name))


def _create_cpsat_var(visitor, pyomo_var, name=None):
    if pyomo_var.is_binary():
        return visitor.model.new_bool_var(_nm(name))
    elif pyomo_var.is_integer():
        return _make_int_var(visitor, pyomo_var, name, 'Var')
    elif pyomo_var.domain.isdiscrete():
        if pyomo_var.domain.isfinite():
            return visitor.model.new_int_var_from_domain(
                cp_model.Domain.from_values(sorted(pyomo_var.domain)), _nm(name)
            )
        raise ValueError(
            "The CP-SAT writer does not support infinite discrete "
            "domains. Cannot write Var '%s' with domain '%s'"
            % (pyomo_var.name, pyomo_var.domain)
        )
    else:
        raise ValueError(
            "The CP-SAT writer can only support integer- or Boolean-valued "
            "variables. Cannot write Var '%s' with domain '%s'"
            % (pyomo_var.name, pyomo_var.domain)
        )


def _before_var(visitor, child):
    _id = id(child)
    if _id not in visitor.var_map:
        if child.fixed:
            return False, (_GENERAL, child.value)
        nm = child.name if visitor.symbolic_solver_labels else None
        cpsat_var = _create_cpsat_var(visitor, child, name=nm)
        visitor.var_map[_id] = cpsat_var
        visitor.pyomo_to_native[child] = cpsat_var
    return False, (_GENERAL, visitor.var_map[_id])


def _before_indexed_var(visitor, child):
    cpsat_vars = {}
    for i, v in child.items():
        if v.fixed:
            cpsat_vars[i] = v.value
            continue
        nm = v.name if visitor.symbolic_solver_labels else None
        cpsat_var = _create_cpsat_var(visitor, v, name=nm)
        visitor.var_map[id(v)] = cpsat_var
        visitor.pyomo_to_native[v] = cpsat_var
        cpsat_vars[i] = cpsat_var
    return False, (_GENERAL, cpsat_vars)


def _before_boolean_var(visitor, child):
    _id = id(child)
    if _id not in visitor.var_map:
        if child.fixed:
            # A fixed literal still needs to behave like one (e.g. support
            # .Not()) wherever it's used, so it can't just be a bare Python
            # int the way a fixed *numeric* Var's value can be.
            return False, (_GENERAL, visitor.model.new_constant(int(value(child))))
        nm = child.name if visitor.symbolic_solver_labels else None
        # Unlike docplex, CP-SAT bool vars are already usable directly as
        # Boolean literals -- no "== 1" wrapper needed to disambiguate them
        # from a generic integer variable.
        cpsat_var = visitor.model.new_bool_var(_nm(nm))
        visitor.var_map[_id] = cpsat_var
        visitor.pyomo_to_native[child] = cpsat_var
    return False, (_GENERAL, visitor.var_map[_id])


def _before_indexed_boolean_var(visitor, child):
    cpsat_vars = {}
    for i, v in child.items():
        if v.fixed:
            cpsat_vars[i] = visitor.model.new_constant(int(value(v)))
            continue
        nm = v.name if visitor.symbolic_solver_labels else None
        cpsat_var = visitor.model.new_bool_var(_nm(nm))
        visitor.var_map[id(v)] = cpsat_var
        visitor.pyomo_to_native[v] = cpsat_var
        cpsat_vars[i] = cpsat_var
    return False, (_GENERAL, cpsat_vars)


def _before_param(visitor, child):
    return False, (_GENERAL, value(child))


def _before_indexed_param(visitor, child):
    return False, (_GENERAL, {idx: value(p) for idx, p in child.items()})


def _create_cpsat_interval_var(visitor, interval_var):
    nm = interval_var.name if visitor.symbolic_solver_labels else None
    # NOTE: OR-Tools variable names exist "for debug/logging only" -- that's
    # the literal comment on IntegerVariableProto's `name` field in
    # cp_model.proto -- and need not be unique; CP-SAT identifies variables
    # internally by index, not name. So appending "_start"/"_end"/"_size"/
    # "_present" below can't introduce a name collision that would change
    # the meaning of the model actually built (it would only ever affect
    # what shows up in solver debug logs, and only when
    # symbolic_solver_labels=True is requested in the first place).
    s = _make_int_var(
        visitor, interval_var.start_time, nm and nm + '_start', 'start time'
    )
    z = _make_int_var(visitor, interval_var.length, nm and nm + '_size', 'length')
    e = _make_int_var(visitor, interval_var.end_time, nm and nm + '_end', 'end time')

    # Always create an explicit presence literal, even for a mandatory
    # interval (where it's simply fixed to 1) -- this keeps
    # _presence_literal() uniform, and the fixed-value constraint costs
    # nothing once CP-SAT's presolve propagates it.
    p = visitor.model.new_bool_var(_sub_name(nm, '_present'))
    if interval_var.is_present.fixed:
        visitor.model.add(p == int(value(interval_var.is_present)))

    # new_optional_interval_var enforces start + size == end *only when the
    # presence literal is 1* -- exactly matching Pyomo's own semantics (an
    # absent IntervalVar's start/end/length aren't linked to each other
    # either), so no separate linking constraint is needed here.
    return visitor.model.new_optional_interval_var(s, z, e, p, _nm(nm))


def _get_cpsat_interval_var(visitor, interval_var):
    _id = id(interval_var)
    if _id not in visitor.var_map:
        visitor.var_map[_id] = _create_cpsat_interval_var(visitor, interval_var)
    return visitor.var_map[_id]


def _before_interval_var(visitor, child):
    cpsat_iv = _get_cpsat_interval_var(visitor, child)
    visitor.pyomo_to_native[child] = cpsat_iv
    return False, (_GENERAL, cpsat_iv)


def _before_indexed_interval_var(visitor, child):
    cpsat_vars = {}
    for i, v in child.items():
        cpsat_iv = _get_cpsat_interval_var(visitor, v)
        visitor.pyomo_to_native[v] = cpsat_iv
        cpsat_vars[i] = cpsat_iv
    return False, (_GENERAL, cpsat_vars)


def _before_interval_var_start_time(visitor, child):
    interval_var = child.get_associated_interval_var()
    cpsat_iv = _get_cpsat_interval_var(visitor, interval_var)
    return False, (_GENERAL, cpsat_iv.start_expr())


def _before_interval_var_end_time(visitor, child):
    interval_var = child.get_associated_interval_var()
    cpsat_iv = _get_cpsat_interval_var(visitor, interval_var)
    return False, (_GENERAL, cpsat_iv.end_expr())


def _before_interval_var_length(visitor, child):
    interval_var = child.get_associated_interval_var()
    cpsat_iv = _get_cpsat_interval_var(visitor, interval_var)
    return False, (_GENERAL, cpsat_iv.size_expr())


def _before_interval_var_presence(visitor, child):
    interval_var = child.get_associated_interval_var()
    cpsat_iv = _get_cpsat_interval_var(visitor, interval_var)
    return False, (_GENERAL, _presence_literal(cpsat_iv))


def _before_sequence_var(visitor, child):
    _id = id(child)
    if _id not in visitor.var_map:
        members = [_get_cpsat_interval_var(visitor, v) for v in child.interval_vars]
        visitor.var_map[_id] = members
        visitor.pyomo_to_native[child] = members
    return False, (_GENERAL, visitor.var_map[_id])


##
# GetItemExpression / GetAttrExpression (variable indirection)
##


def _handle_getitem(visitor, node, *data):
    # Determining each index argument's finite domain (and, if relevant,
    # the (min, max, step) "scale" of that domain) is solver-agnostic and
    # lives in repn/util.py, shared with docplex_writer.py.
    arg_domain = []
    expr = 0
    mult = 1
    # Note: skipping the first argument: that's the IndexedComponent itself.
    for i, arg in enumerate(data[1:]):
        arg_set, scale = getitem_arg_domain(node, i, arg[1])
        arg_domain.append(arg_set)
        if scale is not None:
            _min, _max, _step = scale
            if _step is None:
                raise ValueError(
                    "Variable indirection '%s' is over a discrete domain "
                    "without a constant step size. This is not supported." % node
                )
            # Unlike docplex's expression objects, CP-SAT's IntVar/LinearExpr
            # don't support "//" directly, so a non-constant numerator needs
            # an explicit auxiliary target + add_division_equality.
            numerator = _get_int_expr(visitor, arg) - _min
            if isinstance(numerator, (int, float)):
                term = numerator // _step
            else:
                ub = (_max - _min) // _step
                term = visitor.model.new_int_var(0, ub, '')
                visitor.model.add_division_equality(term, numerator, _step)
            expr += mult * term
            mult *= len(arg_set)

    # Get the list of all elements selectable by the argument expression(s).
    elements = []
    for idx in SetProduct(*arg_domain):
        try:
            idx = idx if len(idx) > 1 else idx[0]
            elements.append(data[0][1][idx])
        except KeyError:
            raise ValueError(
                "Variable indirection '%s' permits an index '%s' "
                "that is not a valid key. In CP-SAT, this is a "
                "structural infeasibility." % (node, idx)
            )

    if elements and hasattr(elements[0], 'start_expr'):
        # IntervalVar candidates: we don't yet know which attribute
        # (start_time, end_time, ...) the caller wants, so we can't build
        # the add_element() call yet -- deferred until a GetAttrExpression
        # picks one (see _handle_getattr). Decided proactively here (by
        # inspecting the CP-SAT object we already built), rather than via
        # docplex's try/except AssertionError dance, since CP-SAT gives no
        # equivalent "you built the wrong kind of element constraint"
        # signal to catch.
        return (_DEFERRED_ELEMENT_CONSTRAINT, (elements, expr))

    lb = min(_bounds(e)[0] for e in elements)
    ub = max(_bounds(e)[1] for e in elements)
    target = visitor.model.new_int_var(lb, ub, '')
    visitor.model.add_element(expr, elements, target)
    return (_GENERAL, target)


_deferred_element_getattr_dispatcher = {
    'start_time': lambda iv: iv.start_expr(),
    'end_time': lambda iv: iv.end_expr(),
    'length': lambda iv: iv.size_expr(),
    'is_present': _presence_literal,
}


def _handle_getattr(visitor, node, obj, attr):
    if obj[0] is _DEFERRED_ELEMENT_CONSTRAINT:
        elements, expr = obj[1]
        try:
            resolved = [
                _deferred_element_getattr_dispatcher[attr[1]](e) for e in elements
            ]
        except KeyError:
            logger.error("Unrecognized attribute in GetAttrExpression: %s." % attr[1])
            raise
        lb = min(_bounds(e)[0] for e in resolved)
        ub = max(_bounds(e)[1] for e in resolved)
        target = visitor.model.new_int_var(lb, ub, '')
        visitor.model.add_element(expr, resolved, target)
        return (_GENERAL, target)
    raise DeveloperError(
        "Unrecognized argument type '%s' to getattr dispatcher." % obj[0]
    )


def _handle_call(visitor, node, *args):
    # docplex supports calling methods like '.before()'/'.implies()' through
    # variable indirection (e.g. 'm.i[m.x].before(...)') by routing them
    # through GetAttrExpression + CallExpression. This writer doesn't yet --
    # the ordinary (non-indirected) form 'm.i.start_time.before(...)' is
    # unaffected, since that's a plain Python method call that never
    # produces a CallExpression node at all.
    raise NotImplementedError(
        "The CP-SAT writer does not yet support calling methods such as "
        "'.before()', '.after()', '.at()', '.implies()', etc. through "
        "variable indirection (found in expression '%s'). Rewrite the "
        "constraint without indirection." % node
    )


##
# Algebraic expressions
##


def _handle_monomial_expr(visitor, node, arg1, arg2):
    if arg2[1].__class__ in EXPR.native_types:
        return _GENERAL, arg1[1] * arg2[1]
    elif arg1[1].__class__ in EXPR.native_types and arg1[1] == 1:
        return arg2
    return (_GENERAL, _get_int_expr(visitor, arg1) * _get_int_expr(visitor, arg2))


def _handle_sum_node(visitor, node, *args):
    return (_GENERAL, sum(_get_int_expr(visitor, arg) for arg in args))


def _handle_negation_node(visitor, node, arg1):
    return (_GENERAL, -1 * _get_int_expr(visitor, arg1))


def _new_aux_int_var(visitor, lb, ub):
    return visitor.model.new_int_var(lb, ub, '')


def _handle_product_node(visitor, node, arg1, arg2):
    a = _get_int_expr(visitor, arg1)
    b = _get_int_expr(visitor, arg2)
    a_lb, a_ub = _bounds(a)
    b_lb, b_ub = _bounds(b)
    products = [a_lb * b_lb, a_lb * b_ub, a_ub * b_lb, a_ub * b_ub]
    target = _new_aux_int_var(visitor, min(products), max(products))
    visitor.model.add_multiplication_equality(target, [a, b])
    return (_GENERAL, target)


def _handle_division_node(visitor, node, arg1, arg2):
    # NOTE: CP-SAT's add_division_equality is *truncating* integer division
    # (rounds toward 0), unlike docplex's cp.float_div. A Pyomo model
    # written with CP Optimizer's division semantics in mind may not port
    # over with identical results.
    a = _get_int_expr(visitor, arg1)
    b = _get_int_expr(visitor, arg2)
    a_lb, a_ub = _bounds(a)
    b_lb, b_ub = _bounds(b)
    if b_lb <= 0 <= b_ub:
        raise ValueError(
            "Cannot write DivisionExpression '%s' to CP-SAT: the "
            "denominator's domain includes 0." % node
        )
    quotients = [a_lb // b_lb, a_lb // b_ub, a_ub // b_lb, a_ub // b_ub]
    target = _new_aux_int_var(visitor, min(quotients), max(quotients))
    visitor.model.add_division_equality(target, a, b)
    return (_GENERAL, target)


def _handle_pow_node(visitor, node, arg1, arg2):
    base = _get_int_expr(visitor, arg1)
    if arg2[1].__class__ not in EXPR.native_types or arg2[1] < 0:
        raise NotImplementedError(
            "The CP-SAT writer only supports PowExpression with a "
            "non-negative constant integer exponent. Cannot write '%s'." % node
        )
    exponent = int(arg2[1])
    if exponent == 0:
        return (_GENERAL, 1)
    result = base
    for _ in range(exponent - 1):
        a_lb, a_ub = _bounds(result)
        b_lb, b_ub = _bounds(base)
        products = [a_lb * b_lb, a_lb * b_ub, a_ub * b_lb, a_ub * b_ub]
        target = _new_aux_int_var(visitor, min(products), max(products))
        visitor.model.add_multiplication_equality(target, [result, base])
        result = target
    return (_GENERAL, result)


def _handle_abs_node(visitor, node, arg1):
    a = _get_int_expr(visitor, arg1)
    a_lb, a_ub = _bounds(a)
    ub = max(abs(a_lb), abs(a_ub))
    target = _new_aux_int_var(visitor, 0, ub)
    visitor.model.add_abs_equality(target, a)
    return (_GENERAL, target)


def _handle_min_node(visitor, node, *args):
    exprs = [_get_int_expr(visitor, arg) for arg in args]
    lb = min(_bounds(e)[0] for e in exprs)
    ub = min(_bounds(e)[1] for e in exprs)
    target = _new_aux_int_var(visitor, lb, ub)
    visitor.model.add_min_equality(target, exprs)
    return (_GENERAL, target)


def _handle_max_node(visitor, node, *args):
    exprs = [_get_int_expr(visitor, arg) for arg in args]
    lb = max(_bounds(e)[0] for e in exprs)
    ub = max(_bounds(e)[1] for e in exprs)
    target = _new_aux_int_var(visitor, lb, ub)
    visitor.model.add_max_equality(target, exprs)
    return (_GENERAL, target)


##
# Relational expressions (all _AUXILIARY: CP-SAT constraints are statements)
##


def _handle_equality_node(visitor, node, arg1, arg2):
    a = _get_int_expr(visitor, arg1)
    b = _get_int_expr(visitor, arg2)

    def assert_fn(visitor):
        visitor.model.add(a == b)

    def reify_fn(visitor, lit):
        visitor.model.add(a == b).only_enforce_if(lit)
        visitor.model.add(a != b).only_enforce_if(lit.Not())

    return (_AUXILIARY, (assert_fn, reify_fn))


def _handle_not_equal_node(visitor, node, arg1, arg2):
    a = _get_int_expr(visitor, arg1)
    b = _get_int_expr(visitor, arg2)

    def assert_fn(visitor):
        visitor.model.add(a != b)

    def reify_fn(visitor, lit):
        visitor.model.add(a != b).only_enforce_if(lit)
        visitor.model.add(a == b).only_enforce_if(lit.Not())

    return (_AUXILIARY, (assert_fn, reify_fn))


def _handle_inequality_node(visitor, node, arg1, arg2):
    # arg1 <= arg2
    a = _get_int_expr(visitor, arg1)
    b = _get_int_expr(visitor, arg2)

    def assert_fn(visitor):
        visitor.model.add(a <= b)

    def reify_fn(visitor, lit):
        visitor.model.add(a <= b).only_enforce_if(lit)
        visitor.model.add(a >= b + 1).only_enforce_if(lit.Not())

    return (_AUXILIARY, (assert_fn, reify_fn))


def _handle_ranged_inequality_node(visitor, node, arg1, arg2, arg3):
    # arg1 <= arg2 <= arg3
    lo = _get_int_expr(visitor, arg1)
    mid = _get_int_expr(visitor, arg2)
    hi = _get_int_expr(visitor, arg3)

    def assert_fn(visitor):
        visitor.model.add(lo <= mid)
        visitor.model.add(mid <= hi)

    def reify_fn(visitor, lit):
        visitor.model.add(lo <= mid).only_enforce_if(lit)
        visitor.model.add(mid <= hi).only_enforce_if(lit)
        # not(lo <= mid <= hi)  <=>  (mid < lo) or (mid > hi)
        below = visitor.model.new_bool_var('')
        visitor.model.add(mid < lo).only_enforce_if(below)
        above = visitor.model.new_bool_var('')
        visitor.model.add(mid > hi).only_enforce_if(above)
        visitor.model.add_bool_or([below, above]).only_enforce_if(lit.Not())

    return (_AUXILIARY, (assert_fn, reify_fn))


##
# Logical expressions (all _AUXILIARY)
##


def _handle_not_node(visitor, node, arg):
    # A materialized literal's .Not() is itself a valid literal at zero
    # extra cost -- no CP-SAT call, and no new auxiliary variable, needed.
    return (_GENERAL, _get_literal(visitor, arg).Not())


def _handle_and_node(visitor, node, *args):
    lits = [_get_literal(visitor, a) for a in args]

    def assert_fn(visitor):
        visitor.model.add_bool_and(lits)

    def reify_fn(visitor, lit):
        visitor.model.add_bool_and(lits).only_enforce_if(lit)
        visitor.model.add_bool_or([l.Not() for l in lits]).only_enforce_if(lit.Not())

    return (_AUXILIARY, (assert_fn, reify_fn))


def _handle_or_node(visitor, node, *args):
    lits = [_get_literal(visitor, a) for a in args]

    def assert_fn(visitor):
        visitor.model.add_bool_or(lits)

    def reify_fn(visitor, lit):
        visitor.model.add_bool_or(lits).only_enforce_if(lit)
        visitor.model.add_bool_and([l.Not() for l in lits]).only_enforce_if(lit.Not())

    return (_AUXILIARY, (assert_fn, reify_fn))


def _handle_xor_node(visitor, node, arg1, arg2):
    a = _get_literal(visitor, arg1)
    b = _get_literal(visitor, arg2)

    def assert_fn(visitor):
        visitor.model.add_bool_xor([a, b])

    def reify_fn(visitor, lit):
        visitor.model.add_bool_xor([a, b]).only_enforce_if(lit)
        visitor.model.add(a == b).only_enforce_if(lit.Not())

    return (_AUXILIARY, (assert_fn, reify_fn))


def _handle_implication_node(visitor, node, arg1, arg2):
    # a => b  ==  (not a) or b -- delegate rather than duplicate add_bool_or.
    not_a = (_GENERAL, _get_literal(visitor, arg1).Not())
    return _handle_or_node(visitor, node, not_a, arg2)


def _handle_equivalence_node(visitor, node, arg1, arg2):
    # Bools are ints in CP-SAT, so equivalence of two literals is literal
    # integer equality.
    a = _get_literal(visitor, arg1)
    b = _get_literal(visitor, arg2)
    return _handle_equality_node(visitor, node, (_GENERAL, a), (_GENERAL, b))


def _handle_exactly_node(visitor, node, *args):
    n = _get_int_expr(visitor, args[0])
    total = sum(_get_literal(visitor, a) for a in args[1:])

    def assert_fn(visitor):
        visitor.model.add(total == n)

    def reify_fn(visitor, lit):
        visitor.model.add(total == n).only_enforce_if(lit)
        visitor.model.add(total != n).only_enforce_if(lit.Not())

    return (_AUXILIARY, (assert_fn, reify_fn))


def _handle_at_most_node(visitor, node, *args):
    n = _get_int_expr(visitor, args[0])
    total = sum(_get_literal(visitor, a) for a in args[1:])

    def assert_fn(visitor):
        visitor.model.add(total <= n)

    def reify_fn(visitor, lit):
        visitor.model.add(total <= n).only_enforce_if(lit)
        visitor.model.add(total >= n + 1).only_enforce_if(lit.Not())

    return (_AUXILIARY, (assert_fn, reify_fn))


def _handle_at_least_node(visitor, node, *args):
    n = _get_int_expr(visitor, args[0])
    total = sum(_get_literal(visitor, a) for a in args[1:])

    def assert_fn(visitor):
        visitor.model.add(total >= n)

    def reify_fn(visitor, lit):
        visitor.model.add(total >= n).only_enforce_if(lit)
        visitor.model.add(total <= n - 1).only_enforce_if(lit.Not())

    return (_AUXILIARY, (assert_fn, reify_fn))


def _handle_all_diff_node(visitor, node, *args):
    exprs = [_get_int_expr(visitor, arg) for arg in args]

    def assert_fn(visitor):
        visitor.model.add_all_different(exprs)

    def reify_fn(visitor, lit):
        visitor.model.add_all_different(exprs).only_enforce_if(lit)
        # CP-SAT has no reified "not all different" primitive; build the
        # negation as "some pair is equal" directly.
        pairs = []
        for i in range(len(exprs)):
            for j in range(i + 1, len(exprs)):
                eq = visitor.model.new_bool_var('')
                visitor.model.add(exprs[i] == exprs[j]).only_enforce_if(eq)
                pairs.append(eq)
        visitor.model.add_bool_or(pairs).only_enforce_if(lit.Not())

    return (_AUXILIARY, (assert_fn, reify_fn))


def _handle_count_if_node(visitor, node, *args):
    # CountIfExpression is numeric-valued (a count), not Boolean -- no
    # reification machinery needed at all.
    return (_GENERAL, sum(_get_literal(visitor, arg) for arg in args))


##
# Named expressions
##

# before_named_expression / handle_named_expression_node are shared with
# docplex_writer.py via repn/util.py (imported above, aliased to the names
# used in the dispatch tables below).


##
# Scheduling: precedence
##


def _handle_before_expression_node(visitor, node, time1, time2, delay):
    # CP-SAT has no analog of docplex's start_before_start/etc. specialized
    # functions, so precedence is always just a plain affine comparison --
    # no "maybe use the specialized form" fallback dance needed.
    lhs = (_GENERAL, _get_int_expr(visitor, time1) + _get_int_expr(visitor, delay))
    return _handle_inequality_node(visitor, node, lhs, time2)


def _handle_at_expression_node(visitor, node, time1, time2, delay):
    lhs = (_GENERAL, _get_int_expr(visitor, time1) + _get_int_expr(visitor, delay))
    return _handle_equality_node(visitor, node, lhs, time2)


##
# Scheduling: span / alternative / synchronize
##
#
# None of these three has a native CP-SAT primitive (unlike docplex's
# cp.span/cp.alternative/cp.synchronize). Each is decomposed below into
# reified presence + start/end/size (in)equalities. Only the "assert
# directly" case is implemented -- Pyomo only ever asserts these as a
# top-level scheduling requirement in practice, and reifying them soundly as
# a *nested* Boolean value would be substantially more work for a case that
# isn't expected to arise; that path raises NotImplementedError rather than
# silently doing something wrong.


def _assert_span(visitor, container, members):
    model = visitor.model
    c_present = _presence_literal(container)
    m_present = [_presence_literal(m) for m in members]

    # container present iff at least one member present
    model.add_bool_or(m_present).only_enforce_if(c_present)
    model.add_bool_and([p.Not() for p in m_present]).only_enforce_if(c_present.Not())

    c_start = container.start_expr()
    c_end = container.end_expr()
    at_min_start = []
    at_max_end = []
    for m, p in zip(members, m_present):
        # the container's span bounds every present member ...
        model.add(c_start <= m.start_expr()).only_enforce_if([c_present, p])
        model.add(c_end >= m.end_expr()).only_enforce_if([c_present, p])
        # ... and each of these literals is a one-directional witness that
        # some present member actually attains the min start / max end (the
        # <= / >= constraints above already guarantee c_start/c_end can't be
        # tighter than the true min/max, so it's sound to let the solver
        # pick whichever member(s) satisfy the equality; we don't need the
        # converse direction since these literals aren't consumed anywhere
        # else).
        is_min = model.new_bool_var('')
        model.add(c_start == m.start_expr()).only_enforce_if(is_min)
        at_min_start.append(is_min)
        is_max = model.new_bool_var('')
        model.add(c_end == m.end_expr()).only_enforce_if(is_max)
        at_max_end.append(is_max)
    model.add_bool_or(at_min_start).only_enforce_if(c_present)
    model.add_bool_or(at_max_end).only_enforce_if(c_present)


def _assert_alternative(visitor, container, members):
    model = visitor.model
    c_present = _presence_literal(container)
    m_present = [_presence_literal(m) for m in members]

    model.add(sum(m_present) == 1).only_enforce_if(c_present)
    model.add(sum(m_present) == 0).only_enforce_if(c_present.Not())

    c_start = container.start_expr()
    c_end = container.end_expr()
    c_size = container.size_expr()
    for m, p in zip(members, m_present):
        model.add(c_start == m.start_expr()).only_enforce_if(p)
        model.add(c_end == m.end_expr()).only_enforce_if(p)
        model.add(c_size == m.size_expr()).only_enforce_if(p)


def _assert_synchronize(visitor, container, members):
    # Pyomo's docstring for SynchronizeExpression only says "if the
    # container is present, the members start/end with it," but the
    # underlying CP Optimizer primitive this wraps (cp.synchronize) actually
    # forces full presence *equality* (a member is present iff the
    # container is), not just one-directional timing when a member happens
    # to be present -- matching that is what's implemented here.
    model = visitor.model
    c_present = _presence_literal(container)
    c_start = container.start_expr()
    c_end = container.end_expr()
    for m in members:
        model.add(_presence_literal(m) == c_present)
        model.add(m.start_expr() == c_start).only_enforce_if(c_present)
        model.add(m.end_expr() == c_end).only_enforce_if(c_present)


def _make_structural_scheduling_auxiliary(kind, assert_body):
    def assert_fn(visitor):
        assert_body(visitor)

    def reify_fn(visitor, lit):
        raise NotImplementedError(
            "The CP-SAT writer does not support using a %sExpression as a "
            "nested Boolean term (only asserting it directly)." % kind
        )

    return (_AUXILIARY, (assert_fn, reify_fn))


def _handle_span_expression_node(visitor, node, *args):
    container = args[0][1]
    members = [a[1] for a in args[1:]]
    return _make_structural_scheduling_auxiliary(
        'Span', lambda visitor: _assert_span(visitor, container, members)
    )


def _handle_alternative_expression_node(visitor, node, *args):
    container = args[0][1]
    members = [a[1] for a in args[1:]]
    return _make_structural_scheduling_auxiliary(
        'Alternative', lambda visitor: _assert_alternative(visitor, container, members)
    )


def _handle_synchronize_expression_node(visitor, node, *args):
    container = args[0][1]
    members = [a[1] for a in args[1:]]
    return _make_structural_scheduling_auxiliary(
        'Synchronize', lambda visitor: _assert_synchronize(visitor, container, members)
    )


##
# Scheduling: sequencing (no_overlap / first / last / before_in / predecessor_to)
##
#
# CP-SAT has no sequence/permutation-of-intervals primitive at all (Pyomo's
# SequenceVar is already just a plain Python list on the Pyomo side -- see
# sequence_var.py -- with no CP-SAT object counterpart here either). Whether
# first_in_sequence/last_in_sequence/before_in_sequence/predecessor_to can be
# encoded cheaply (as reified start-time comparisons) or need a heavier,
# fully general encoding (explicit rank/position variables + AllDifferent)
# depends on whether a NoOverlapExpression is *unconditionally* asserted
# elsewhere over the same SequenceVar -- and that can only be known once
# every LogicalConstraint in the model has been processed, regardless of the
# order constraints were declared in or whether they share an expression
# tree. So these four constraint types are never resolved at exitNode time:
# each just records a task and defers translation to the very end of
# CPSatWriter.write(), after every LogicalConstraint has been walked and
# asserted (see visitor.deferred_sequence_tasks / _resolve_sequence_task).


class _SequenceTask:
    __slots__ = ('kind', 'args', 'seq')

    def __init__(self, kind, args, seq):
        self.kind = kind
        self.args = args
        self.seq = seq


def _defer_sequence_task(kind, args, seq_var):
    task = _SequenceTask(kind, args, seq_var)

    def assert_fn(visitor):
        visitor.deferred_sequence_tasks.append(task)

    def reify_fn(visitor, lit):
        raise NotImplementedError(
            "The CP-SAT writer does not support using a %s sequencing "
            "expression as a nested Boolean term (only asserting it "
            "directly)." % kind
        )

    return (_AUXILIARY, (assert_fn, reify_fn))


def _handle_no_overlap_expression_node(visitor, node, seq_var):
    members = seq_var[1]
    seq = node.arg(0)

    def assert_fn(visitor):
        visitor.model.add_no_overlap(members)
        # Recorded here (in assert_fn), not at exitNode/walk time, so that a
        # NoOverlapExpression sitting inside e.g. Or(no_overlap(seq), foo)
        # (i.e. not actually guaranteed) correctly does *not* get credited --
        # that path goes through reify_fn below instead.
        visitor.sequences_with_no_overlap.add(id(seq))

    def reify_fn(visitor, lit):
        raise NotImplementedError(
            "The CP-SAT writer does not support using a NoOverlapExpression "
            "as a nested Boolean term (only asserting it directly)."
        )

    return (_AUXILIARY, (assert_fn, reify_fn))


def _handle_first_in_sequence_expression_node(visitor, node, interval_var, seq_var):
    return _defer_sequence_task('first_in_sequence', (node.arg(0),), node.arg(1))


def _handle_last_in_sequence_expression_node(visitor, node, interval_var, seq_var):
    return _defer_sequence_task('last_in_sequence', (node.arg(0),), node.arg(1))


def _handle_before_in_sequence_expression_node(
    visitor, node, before_var, after_var, seq_var
):
    return _defer_sequence_task(
        'before_in_sequence', (node.arg(0), node.arg(1)), node.arg(2)
    )


def _handle_predecessor_to_expression_node(
    visitor, node, before_var, after_var, seq_var
):
    return _defer_sequence_task(
        'predecessor_to', (node.arg(0), node.arg(1)), node.arg(2)
    )


def _get_sequence_positions(visitor, seq):
    # Explicit rank/position variables, giving each member of the sequence
    # a genuine (solver-chosen) position independent of its timing -- the
    # fully general fallback, built lazily and memoized per SequenceVar,
    # only for sequences that don't have an accompanying NoOverlap to
    # piggyback a cheaper encoding on (see the module comment above).
    _id = id(seq)
    if _id in visitor.sequence_positions:
        return visitor.sequence_positions[_id]

    model = visitor.model
    members = seq.interval_vars
    n = len(members)
    positions = {}
    pos_vars = []
    for i, iv in enumerate(members):
        p = _presence_literal(_get_cpsat_interval_var(visitor, iv))
        # A present member's position is one of 0..n-1; an absent member is
        # pinned to its own unique sentinel n+i (not a single shared
        # sentinel -- two simultaneously-absent members sharing one value
        # would otherwise violate add_all_different for no good reason).
        domain = cp_model.Domain.from_intervals([[0, n - 1], [n + i, n + i]])
        pos = model.new_int_var_from_domain(domain, '')
        model.add(pos < n).only_enforce_if(p)
        model.add(pos == n + i).only_enforce_if(p.Not())
        positions[id(iv)] = pos
        pos_vars.append(pos)
    model.add_all_different(pos_vars)

    visitor.sequence_positions[_id] = positions
    return positions


def _resolve_sequence_task(visitor, task):
    model = visitor.model
    seq = task.seq
    use_positions = id(seq) not in visitor.sequences_with_no_overlap

    def iv(pyomo_iv):
        return _get_cpsat_interval_var(visitor, pyomo_iv)

    def present(pyomo_iv):
        return _presence_literal(iv(pyomo_iv))

    if task.kind == 'first_in_sequence':
        (target,) = task.args
        others = [m for m in seq.interval_vars if m is not target]
        if use_positions:
            positions = _get_sequence_positions(visitor, seq)
            for m in others:
                model.add(positions[id(target)] < positions[id(m)]).only_enforce_if(
                    [present(target), present(m)]
                )
        else:
            for m in others:
                model.add(
                    iv(target).start_expr() <= iv(m).start_expr()
                ).only_enforce_if([present(target), present(m)])

    elif task.kind == 'last_in_sequence':
        (target,) = task.args
        others = [m for m in seq.interval_vars if m is not target]
        if use_positions:
            positions = _get_sequence_positions(visitor, seq)
            for m in others:
                model.add(positions[id(target)] > positions[id(m)]).only_enforce_if(
                    [present(target), present(m)]
                )
        else:
            for m in others:
                model.add(
                    iv(target).start_expr() >= iv(m).start_expr()
                ).only_enforce_if([present(target), present(m)])

    elif task.kind == 'before_in_sequence':
        before_iv, after_iv = task.args
        if use_positions:
            positions = _get_sequence_positions(visitor, seq)
            model.add(
                positions[id(before_iv)] < positions[id(after_iv)]
            ).only_enforce_if([present(before_iv), present(after_iv)])
        else:
            model.add(
                iv(before_iv).end_expr() <= iv(after_iv).start_expr()
            ).only_enforce_if([present(before_iv), present(after_iv)])

    elif task.kind == 'predecessor_to':
        before_iv, after_iv = task.args
        p_before = present(before_iv)
        p_after = present(after_iv)
        if use_positions:
            # Direct adjacency in rank is simpler here than the start-time
            # version below: no "nothing in between" loop is needed at all.
            positions = _get_sequence_positions(visitor, seq)
            model.add(
                positions[id(after_iv)] == positions[id(before_iv)] + 1
            ).only_enforce_if([p_before, p_after])
        else:
            model.add(
                iv(before_iv).end_expr() <= iv(after_iv).start_expr()
            ).only_enforce_if([p_before, p_after])
            # ... and no other present member of the sequence may sit
            # strictly between them.
            for m in seq.interval_vars:
                if m is before_iv or m is after_iv:
                    continue
                p_m = present(m)
                # One-directional reifications suffice here: these two
                # literals are consumed only inside the add_bool_or below,
                # so forcing "literal true => inequality holds" is enough to
                # make the disjunction actually imply what we want; we don't
                # need the converse.
                m_is_before = model.new_bool_var('')
                model.add(
                    iv(m).end_expr() <= iv(before_iv).start_expr()
                ).only_enforce_if(m_is_before)
                m_is_after = model.new_bool_var('')
                model.add(
                    iv(m).start_expr() >= iv(after_iv).end_expr()
                ).only_enforce_if(m_is_after)
                model.add_bool_or([m_is_before, m_is_after]).only_enforce_if(
                    [p_before, p_after, p_m]
                )
    else:
        raise DeveloperError("Unrecognized deferred sequence task kind %r" % task.kind)


##
# Scheduling: step functions / cumulative resources
##
#
# Confirmed scope for v1: a fast path for the common "sum of Pulses over the
# whole horizon, capacity bound" case (-> add_cumulative), a special case for
# a sum of Steps only (-> add_reservoir_constraint_with_active), and a clear
# NotImplementedError naming the failing condition for everything else
# (mixed Pulse+Step, non-constant heights/bounds, a partial sub-window). The
# general breakpoint/event-based encoding for the fully general case is
# deferred to a follow-up.
#
# AlwaysIn is intercepted in beforeChild (alongside Pulse/Step*/
# CumulativeFunction/NegatedStepFunction, all of which never reach exitNode
# on their own here -- see LogicalToCpSat.step_function_handles) so this
# handler sees the *raw*, unwalked node: deciding which fast path (if any)
# applies requires inspecting the original Pulse/Step term structure, which
# would already be lost by the time a normal walk finished reducing
# everything down to a single summed numeric expression.


def _try_constant(x):
    if x.__class__ in EXPR.native_types:
        return x
    return None


def _decompose_cumulative_function(cumul_func):
    # Returns a list of (term, sign) pairs: `term` is always an elementary
    # Pulse/StepAt/StepAtStart/StepAtEnd (a NegatedStepFunction is unwrapped,
    # its negation folded into `sign`, which is +1 or -1).
    raw = (
        cumul_func.args if cumul_func.__class__ is CumulativeFunction else [cumul_func]
    )
    terms = []
    for t in raw:
        if t.__class__ is NegatedStepFunction:
            terms.append((t.args[0], -1))
        else:
            terms.append((t, 1))
    return terms


def _term_interval_var_data(term):
    # Pulse keys off ._interval_var; StepAtStart/StepAtEnd key off ._time,
    # which Step.__new__ already resolved to the associated IntervalVarData.
    # StepAt's ._time is a bare constant, not tied to any interval.
    if term.__class__ is Pulse:
        return term._interval_var
    if term.__class__ in (StepAtStart, StepAtEnd):
        return term._time
    return None


def _covers_whole_horizon(terms, start_val, end_val):
    if start_val is None or end_val is None:
        return False
    for t, _sign in terms:
        iv_data = _term_interval_var_data(t)
        if iv_data is None:
            # StepAt: a bare constant trigger time, not tied to an interval.
            if not (start_val <= t._time):
                return False
            continue
        if not (
            start_val <= value(iv_data.start_time.lb)
            and end_val >= value(iv_data.end_time.ub)
        ):
            return False
    return True


def _handle_always_in_node(visitor, node):
    cumul_func, lb, ub, start, end = node.args
    terms = _decompose_cumulative_function(cumul_func)

    lb_val = _try_constant(lb)
    ub_val = _try_constant(ub)
    start_val = _try_constant(start)
    end_val = _try_constant(end)
    heights = []
    for t, sign in terms:
        h = _try_constant(t._height)
        heights.append(None if h is None else sign * h)

    if (
        lb_val == 0
        and all(t.__class__ is Pulse for t, _sign in terms)
        and all(h is not None and h >= 0 for h in heights)
        and ub_val is not None
        and _covers_whole_horizon(terms, start_val, end_val)
    ):
        intervals = [
            _get_cpsat_interval_var(visitor, t._interval_var) for t, _sign in terms
        ]
        visitor.model.add_cumulative(intervals, heights, ub_val)
        return False, (_GENERAL, 1)

    if (
        terms
        and all(t.__class__ in (StepAt, StepAtStart, StepAtEnd) for t, _sign in terms)
        and all(h is not None for h in heights)
        and lb_val is not None
        and ub_val is not None
        and _covers_whole_horizon(terms, start_val, end_val)
    ):
        times = []
        actives = []
        for t, _sign in terms:
            if t.__class__ is StepAt:
                times.append(t._time)
                actives.append(True)
            else:
                cpsat_iv = _get_cpsat_interval_var(visitor, t._time)
                times.append(
                    cpsat_iv.start_expr()
                    if t.__class__ is StepAtStart
                    else cpsat_iv.end_expr()
                )
                actives.append(_presence_literal(cpsat_iv))
        visitor.model.add_reservoir_constraint_with_active(
            times, heights, actives, lb_val, ub_val
        )
        return False, (_GENERAL, 1)

    raise NotImplementedError(
        "The CP-SAT writer only supports AlwaysIn constraints in two "
        "special cases: (1) a sum of Pulse terms only, with a zero lower "
        "bound, constant non-negative heights, and a window covering the "
        "full range of the referenced interval vars (translated to "
        "add_cumulative), or (2) a sum of Step terms only, with constant "
        "heights and a window similarly covering the full horizon "
        "(translated to add_reservoir_constraint_with_active). This "
        "model's AlwaysIn('%s') satisfies neither -- it may mix Pulse and "
        "Step terms, use a non-constant bound/height/window, or use a "
        "window that doesn't cover the full horizon of the terms involved. "
        "General AlwaysIn support is not yet implemented." % node
    )


_step_function_handles = {
    AlwaysIn: lambda visitor, node: _handle_always_in_node(visitor, node)
}


##
# Dispatch tables
##

_operator_handles = {
    EXPR.GetItemExpression: _handle_getitem,
    EXPR.GetAttrExpression: _handle_getattr,
    EXPR.CallExpression: _handle_call,
    EXPR.NegationExpression: _handle_negation_node,
    EXPR.ProductExpression: _handle_product_node,
    EXPR.DivisionExpression: _handle_division_node,
    EXPR.PowExpression: _handle_pow_node,
    EXPR.AbsExpression: _handle_abs_node,
    EXPR.MonomialTermExpression: _handle_monomial_expr,
    EXPR.SumExpression: _handle_sum_node,
    EXPR.MinExpression: _handle_min_node,
    EXPR.MaxExpression: _handle_max_node,
    EXPR.NotExpression: _handle_not_node,
    EXPR.EquivalenceExpression: _handle_equivalence_node,
    EXPR.ImplicationExpression: _handle_implication_node,
    EXPR.AndExpression: _handle_and_node,
    EXPR.OrExpression: _handle_or_node,
    EXPR.XorExpression: _handle_xor_node,
    EXPR.ExactlyExpression: _handle_exactly_node,
    EXPR.AtMostExpression: _handle_at_most_node,
    EXPR.AtLeastExpression: _handle_at_least_node,
    EXPR.AllDifferentExpression: _handle_all_diff_node,
    EXPR.CountIfExpression: _handle_count_if_node,
    EXPR.EqualityExpression: _handle_equality_node,
    EXPR.NotEqualExpression: _handle_not_equal_node,
    EXPR.InequalityExpression: _handle_inequality_node,
    EXPR.RangedExpression: _handle_ranged_inequality_node,
    BeforeExpression: _handle_before_expression_node,
    AtExpression: _handle_at_expression_node,
    ExpressionData: _handle_named_expression_node,
    ScalarExpression: _handle_named_expression_node,
    NoOverlapExpression: _handle_no_overlap_expression_node,
    FirstInSequenceExpression: _handle_first_in_sequence_expression_node,
    LastInSequenceExpression: _handle_last_in_sequence_expression_node,
    BeforeInSequenceExpression: _handle_before_in_sequence_expression_node,
    PredecessorToExpression: _handle_predecessor_to_expression_node,
    SpanExpression: _handle_span_expression_node,
    AlternativeExpression: _handle_alternative_expression_node,
    SynchronizeExpression: _handle_synchronize_expression_node,
}


class LogicalToCpSat(CPExpressionVisitorBase):
    exit_node_dispatcher = ExitNodeDispatcher(_operator_handles)

    var_handles = {
        IntervalVarStartTime: _before_interval_var_start_time,
        IntervalVarEndTime: _before_interval_var_end_time,
        IntervalVarLength: _before_interval_var_length,
        IntervalVarPresence: _before_interval_var_presence,
        ScalarIntervalVar: _before_interval_var,
        IntervalVarData: _before_interval_var,
        IndexedIntervalVar: _before_indexed_interval_var,
        ScalarSequenceVar: _before_sequence_var,
        SequenceVarData: _before_sequence_var,
        ScalarVar: _before_var,
        VarData: _before_var,
        IndexedVar: _before_indexed_var,
        ScalarBooleanVar: _before_boolean_var,
        BooleanVarData: _before_boolean_var,
        IndexedBooleanVar: _before_indexed_boolean_var,
        ExpressionData: _before_named_expression,
        ScalarExpression: _before_named_expression,
        IndexedParam: _before_indexed_param,
        ScalarParam: _before_param,
        ParamData: _before_param,
    }
    step_function_handles = _step_function_handles

    def __init__(self, cpsat_model, symbolic_solver_labels=False):
        super().__init__(symbolic_solver_labels=symbolic_solver_labels)
        self.model = cpsat_model
        # Populated as a side effect of asserting NoOverlapExpressions (see
        # _handle_no_overlap_expression_node); consulted while resolving
        # deferred_sequence_tasks, both below.
        self.sequences_with_no_overlap = set()
        self.deferred_sequence_tasks = []
        # Lazily built, memoized by id(SequenceVarData), only for sequences
        # that actually need the position-variable fallback.
        self.sequence_positions = {}


@WriterFactory.register(
    'cpsat_model', 'Generate the corresponding OR-Tools CP-SAT cp_model.CpModel object'
)
class CPSatWriter:
    CONFIG = ConfigDict('cpsat_model_writer')
    CONFIG.declare(
        'symbolic_solver_labels',
        ConfigValue(
            default=False,
            domain=bool,
            description='Write Pyomo Var and Constraint names to the CP-SAT model',
        ),
    )

    def __init__(self):
        self.config = self.CONFIG()

    def write(self, model, **options):
        config = options.pop('config', self.config)(options)

        components = categorize_cp_model(model, sort=SortComponents.deterministic)

        cpsat_model = cp_model.CpModel()
        visitor = LogicalToCpSat(
            cpsat_model, symbolic_solver_labels=config.symbolic_solver_labels
        )

        active_objs = components[Objective]
        if len(active_objs) > 1:
            raise ValueError(
                "More than one active objective defined for "
                "input model '%s': Cannot write to CP-SAT." % model.name
            )
        elif len(active_objs) == 1:
            obj = active_objs[0]
            obj_expr = visitor.walk_expression((obj.expr, obj, 0))
            obj_int_expr = _get_int_expr(visitor, obj_expr)
            if obj.sense is minimize:
                cpsat_model.minimize(obj_int_expr)
            else:
                cpsat_model.maximize(obj_int_expr)
        # No objective is fine too, this is CP after all...

        # Write algebraic constraints
        for cons in components[Constraint]:
            expr = visitor.walk_expression((cons.body, cons, 0))
            expr_val = _get_int_expr(visitor, expr)
            if cons.lower is not None:
                cpsat_model.add(cons.lb <= expr_val)
            if cons.upper is not None:
                cpsat_model.add(expr_val <= cons.ub)

        # Write interval vars (these are secretly constraints if they have
        # to be scheduled) -- walked once, purely for the side effect of
        # creating them, even if otherwise unreferenced.
        for var in components[IntervalVar]:
            visitor.walk_expression((var, var, 0))

        # Same idea for sequence vars, so their member lists are available
        # before any dependent sequencing constraint is processed.
        for var in components[SequenceVar]:
            visitor.walk_expression((var, var, 0))

        # Write logical constraints. This can't be a single "walk, then
        # immediately assert" loop the way docplex's writer does it: the
        # sequencing constraints deferred above need every LogicalConstraint
        # walked *and* asserted first, so that visitor.sequences_with_no_overlap
        # is complete regardless of declaration order (see the "Scheduling:
        # sequencing" section of cpsat_writer.py for the full explanation).
        walked = [
            (cons, visitor.walk_expression((cons.expr, cons, 0)))
            for cons in components[LogicalConstraint]
        ]
        for cons, expr in walked:
            _assert_true(visitor, expr)

        for task in visitor.deferred_sequence_tasks:
            _resolve_sequence_task(visitor, task)

        return cpsat_model, visitor.pyomo_to_native


@SolverFactory.register(
    'cp_sat', doc='Direct interface to Google OR-Tools CP-SAT solver'
)
class CPSatSolver:
    CONFIG = ConfigDict('cp_sat_solver')
    CONFIG.declare(
        'symbolic_solver_labels',
        ConfigValue(
            default=False,
            domain=bool,
            description='Write Pyomo Var and Constraint names to the CP-SAT model',
        ),
    )
    CONFIG.declare(
        'tee',
        ConfigValue(
            default=False, domain=bool, description="Stream solver output to terminal."
        ),
    )
    CONFIG.declare(
        'options', ConfigValue(default={}, description="Dictionary of solver options.")
    )

    def __init__(self, **kwds):
        self.config = self.CONFIG()
        self.config.set_value(kwds)
        # A flat 1:1 status map suffices here -- unlike docplex's separate
        # solve-status/stop-cause distinction, CP-SAT's solve() returns a
        # single status enum.
        if cp_model_available:
            self._status_map = {
                cp_model.OPTIMAL: TerminationCondition.optimal,
                cp_model.FEASIBLE: TerminationCondition.feasible,
                cp_model.INFEASIBLE: TerminationCondition.infeasible,
                cp_model.MODEL_INVALID: TerminationCondition.error,
                cp_model.UNKNOWN: TerminationCondition.unknown,
            }

    @property
    def options(self):
        return self.config.options

    # Support use as a context manager under current solver API
    def __enter__(self):
        return self

    def __exit__(self, t, v, traceback):
        pass

    def available(self, exception_flag=True):
        return bool(cp_model_available)

    def license_is_valid(self):
        # CP-SAT is open-source with no license restriction.
        return True

    def solve(self, model, **kwds):
        """Solve the model.

        Args:
            model (Block): a Pyomo model or block to be solved

        """
        config = self.config()
        config.set_value(kwds)

        writer = CPSatWriter()
        cpsat_model, var_map = writer.write(
            model, symbolic_solver_labels=config.symbolic_solver_labels
        )

        solver = cp_model.CpSolver()
        for key, val in self.options.items():
            setattr(solver.parameters, key, val)
        if config.tee:
            solver.parameters.log_search_progress = True

        status = solver.solve(cpsat_model)

        results = SolverResults()
        results.solver.name = "CP-SAT"
        results.problem.name = model.name
        results.solver.solve_time = solver.wall_time
        results.solver.termination_condition = self._status_map.get(
            status, TerminationCondition.unknown
        )

        if cpsat_model.has_objective():
            objs = list(model.component_data_objects(Objective, active=True))
            sense = objs[0].sense if objs else None
            val = solver.objective_value
            bound = solver.best_objective_bound
            results.problem.number_of_objectives = 1
            results.problem.sense = sense
            if sense is maximize:
                results.problem.lower_bound = val
                results.problem.upper_bound = bound
            else:
                results.problem.lower_bound = bound
                results.problem.upper_bound = val
        else:
            results.problem.number_of_objectives = 0
            results.problem.sense = None
            results.problem.lower_bound = None
            results.problem.upper_bound = None

        # Copy the variable values onto the Pyomo model, using the map we
        # stored on the writer.
        if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            for py_var, cpsat_var in var_map.items():
                if py_var.ctype is SequenceVar:
                    # They don't actually have values -- the IntervalVars
                    # will get set.
                    continue
                if py_var.ctype is IntervalVar:
                    p = solver.value(_presence_literal(cpsat_var))
                    if not p:
                        py_var.is_present.set_value(False)
                    else:
                        start = solver.value(cpsat_var.start_expr())
                        end = solver.value(cpsat_var.end_expr())
                        py_var.is_present.set_value(True)
                        py_var.start_time.set_value(start, skip_validation=True)
                        py_var.end_time.set_value(end, skip_validation=True)
                        py_var.length.set_value(end - start, skip_validation=True)
                elif py_var.ctype in {Var, BooleanVar}:
                    py_var.set_value(solver.value(cpsat_var), skip_validation=True)
                else:
                    raise DeveloperError(
                        "Unrecognized Pyomo type in pyomo-to-CP-SAT "
                        "variable map: %s" % type(py_var)
                    )

        return results
