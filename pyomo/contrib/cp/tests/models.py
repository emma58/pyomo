# ____________________________________________________________________________________
#
# Pyomo: Python Optimization Modeling Objects
# Copyright (c) 2008-2026 National Technology and Engineering Solutions of Sandia, LLC
# Under the terms of Contract DE-NA0003525 with National Technology and Engineering
# Solutions of Sandia, LLC, the U.S. Government retains certain rights in this
# software.  This software is distributed under the 3-clause BSD License.
# ____________________________________________________________________________________
"""Model builders shared by the pyomo.contrib.cp writer test suites (one per
solver backend). Keeping these here means a scheduling model that one CP
writer is tested against can be reused, unmodified, to test another, rather
than being retyped in each writer's own test file.

None of these functions solve or check anything -- see common_tests.py for
the solver-agnostic checks that go with each model.
"""

from pyomo.contrib.cp import IntervalVar, SequenceVar, Pulse, Step, AlwaysIn
from pyomo.contrib.cp.scheduling_expr.sequence_expressions import (
    first_in_sequence,
    predecessor_to,
    no_overlap,
)
from pyomo.core.expr.logical_expr import implies
from pyomo.environ import ConcreteModel, Set, Var, Integers, LogicalConstraint


def mice_and_cookies_model():
    """A satisfaction (no-objective) scheduling problem: eating cookies makes
    crumbs, which (if there are enough of them) require sweeping, and a mouse
    must be kept busy at all times by exactly one of these chores (or by
    reading a story, indefinitely, once the chores run out) -- otherwise it
    will get up to trouble doing the dishes instead.

    Exercises: optional and mandatory IntervalVars, precedence
    (start_time.after), implications, and a cumulative resource ("exactly one
    mouse occupied") built from Pulse and Step step functions and asserted
    with AlwaysIn.
    """
    m = ConcreteModel()
    m.eat_cookie = IntervalVar([0, 1], length=8, end=(0, 24), optional=False)
    m.eat_cookie[0].start_time.bounds = (0, 4)
    m.eat_cookie[1].start_time.bounds = (5, 20)

    m.read_story = IntervalVar(start=(15, 24), end=(0, 24), length=(2, 3))
    m.sweep_crumbs = IntervalVar(optional=True, length=1, end=(0, 24))
    m.do_dishes = IntervalVar(optional=True, length=5, end=(0, 24))

    m.num_crumbs = Var(domain=Integers, bounds=(0, 100))

    # Precedence
    m.cookies = LogicalConstraint(
        expr=m.eat_cookie[1].start_time.after(m.eat_cookie[0].end_time)
    )
    m.cookies_imply_crumbs = LogicalConstraint(
        expr=m.eat_cookie[0].is_present.implies(m.num_crumbs == 5)
    )
    m.good_mouse = LogicalConstraint(
        expr=implies(m.num_crumbs >= 3, m.sweep_crumbs.is_present)
    )
    m.sweep_after = LogicalConstraint(
        expr=m.sweep_crumbs.start_time.after(m.eat_cookie[1].end_time)
    )

    m.mice_occupied = (
        sum(Pulse((m.eat_cookie[i], 1)) for i in range(2))
        + Step(m.read_story.start_time, 1)
        + Pulse((m.sweep_crumbs, 1))
        - Pulse((m.do_dishes, 1))
    )

    # Must keep exactly one mouse occupied for a 25-hour day
    m.treat_your_mouse_well = LogicalConstraint(
        expr=AlwaysIn(cumul_func=m.mice_occupied, bounds=(1, 1), times=(0, 24))
    )

    return m


def three_step_sequence_model():
    """A sequencing problem over three tasks whose lengths increase with
    their index: task 1 must be first, and 1->2->3 must be immediate
    predecessors of each other, with no overlap allowed.

    Exercises: an indexed IntervalVar, a SequenceVar, first_in_sequence,
    predecessor_to, and no_overlap.
    """
    m = ConcreteModel()
    m.Steps = Set(initialize=[1, 2, 3])

    def length_rule(m, j):
        return 2 * j

    m.i = IntervalVar(m.Steps, start=(0, 12), end=(0, 12), length=length_rule)
    m.seq = SequenceVar(expr=[m.i[j] for j in m.Steps])
    m.first = LogicalConstraint(expr=first_in_sequence(m.i[1], m.seq))
    m.seq_order1 = LogicalConstraint(expr=predecessor_to(m.i[1], m.i[2], m.seq))
    m.seq_order2 = LogicalConstraint(expr=predecessor_to(m.i[2], m.i[3], m.seq))
    m.no_overlap = LogicalConstraint(expr=no_overlap(m.seq))

    return m


def three_step_sequence_model_no_overlap_constraint():
    """The same sequencing problem as three_step_sequence_model(), but
    without the no_overlap constraint. Since first_in_sequence and
    predecessor_to no longer have an accompanying NoOverlap to piggyback
    their ordering semantics on, a writer that only knows how to encode
    sequencing via temporal (start-time) comparisons cannot handle this
    model correctly -- it needs an explicit notion of sequence position
    instead. Used to exercise that fallback path specifically.
    """
    m = three_step_sequence_model()
    del m.no_overlap
    return m
