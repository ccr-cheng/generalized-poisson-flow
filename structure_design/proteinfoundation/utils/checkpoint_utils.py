"""Resolve local checkpoint paths relative to the command's working directory."""

import os

from proteinfoundation.utils.fetch_last_ckpt import fetch_last_ckpt


def _expand_path(path):
    """Expand environment/home references without dereferencing checkpoint symlinks."""
    return os.path.expanduser(os.path.expandvars(os.fspath(path)))


def resolve_checkpoint_path(path, filename=None):
    """Return an existing checkpoint's absolute path.

    Inference can supply a directory and filename separately. Relative paths
    are based on the launch directory (the repository root in the README),
    and both components support ``~`` and environment variables. Preserve the
    final symlink name so EMA companion lookup uses the supplied checkpoint name.
    """
    if path is None or not os.fspath(path).strip():
        raise ValueError("A checkpoint path must be provided")
    checkpoint = _expand_path(path)
    if filename is not None:
        checkpoint = os.path.join(checkpoint, _expand_path(filename))
    checkpoint = os.path.abspath(checkpoint)
    if not os.path.isfile(checkpoint):
        raise FileNotFoundError(
            f"Checkpoint file not found: {checkpoint}. "
            f"Relative paths are resolved from the working directory: {os.getcwd()}"
        )
    return checkpoint


def select_resume_checkpoint(checkpoint_dir, checkpoint_path=None, required=False):
    """Select an explicit checkpoint or the run's latest automatic checkpoint."""
    if checkpoint_path is not None:
        return resolve_checkpoint_path(checkpoint_path)
    checkpoint_dir = os.path.abspath(_expand_path(checkpoint_dir))
    latest = fetch_last_ckpt(checkpoint_dir)
    if latest is not None:
        return resolve_checkpoint_path(checkpoint_dir, latest)
    if required:
        raise FileNotFoundError(
            f"No checkpoint to resume in {checkpoint_dir}. "
            "Set resume_ckpt_path=<checkpoint> to select one explicitly."
        )
    return None
