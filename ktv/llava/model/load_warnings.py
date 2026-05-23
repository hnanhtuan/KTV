import warnings


META_PARAMETER_COPY_WARNING = (
    r"for .*: copying from a non-meta parameter in the checkpoint to a meta "
    r"parameter in the current model, which is a no-op\..*"
)


def suppress_meta_parameter_copy_warning():
    """Hide the noisy PyTorch warning emitted by older Transformers loaders."""
    warnings.filterwarnings(
        "ignore",
        message=META_PARAMETER_COPY_WARNING,
        category=UserWarning,
    )
