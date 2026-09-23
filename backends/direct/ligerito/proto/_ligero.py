"""Import shim for the reused ``backends/direct/ligero`` modules: ``backends.direct`` is a namespace directory, so
under pytest (rootdir ``backends/direct``) the sibling package is ``ligero``, under ``python -m backends.direct...``
it is ``backends.direct.ligero``."""
try:
    from backends.direct.ligero import encode_simt, field, merkle          # noqa: F401
except ImportError:                                                        # pragma: no cover
    from ligero import encode_simt, field, merkle                          # noqa: F401
