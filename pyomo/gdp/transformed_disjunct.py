from pyomo.core.base.block import _BlockData, Block
from pyomo.core.base.global_set import UnindexedComponent_index
from weakref import ref as weakref_ref

_ref_types = {type(None), weakref_ref}

class _TransformedDisjunctData(_BlockData):
    @property
    def src_disjunct(self):
        return self._src_disjunct

    def __init__(self, component):
        _BlockData.__init__(self, component)
        # pointer to the Disjunct whose transformation block this is.
        self._src_disjunct = None

    def __getstate__(self):
        """
        This method must be defined to support pickling because this class
        owns a weakref to the original Disjunct '_src_disjunct'
        """
        state = super().__getstate__()
        set_trace()
        if self._src_disjunct is not None:
            state['_src_disjunct'] = self._src_disjunct()
        return state

    def __setstate__(self):
        """
        This method must be defined to support pickling because this class
        owns a weakref to the original Disjunct '_src_disjunct'
        """
        if state['_src_disjunct'].__class__ not in _ref_types:
            state['_src_disjunct'] = weakref_ref(state['_src_disjunct'])

class _TransformedDisjunct(Block):
    def __new__(cls, *args, **kwds):
        if cls != _TransformedDisjunct:
            return super(_TransformedDisjunct, cls).__new__(cls)
        if args == ():
            return _ScalarTransformedDisjunct.__new__(
                _ScalarTransformedDisjunct)
        else:
            return _IndexedTransformedDisjunct.__new__(
                _IndexedTransformedDisjunct)

    def __init__(self, *args, **kwargs):
        kwargs.setdefault('ctype', Block)
        Block.__init__(self, *args, **kwargs)

class _ScalarTransformedDisjunct(_TransformedDisjunctData,
                                 _TransformedDisjunct):
    def __init__(self, *args, **kwds):
        _TransformedDisjunctData.__init__(self, self)
        _TransformedDisjunct.__init__(self, *args, **kwds)
        self._data[None] = self
        self._index = UnindexedComponent_index

class _IndexedTransformedDisjunct(_TransformedDisjunct):
    pass
