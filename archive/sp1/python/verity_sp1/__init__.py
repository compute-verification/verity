"""SP1 backend driver.  ``host`` speaks the ``veritor-zk-host`` subprocess protocol over opaque statement and
witness bytes; ``typed`` is the backend boundary -- a ``typed-obligation/v0`` object in, an :class:`Acceptance`
out -- and pins the approved guest identity.  Import from the owning module."""
