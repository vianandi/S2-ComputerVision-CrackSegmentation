import os


def resolve_output_dir(project_root, requested_dir=None, base_name="results"):
    """Resolve output directory.

    If requested_dir is provided, it will be used (absolute path or relative to project_root).
    Otherwise, creates an incremental directory: results, results_2, results_3, ...
    """
    if requested_dir:
        output_dir = requested_dir
        if not os.path.isabs(output_dir):
            output_dir = os.path.join(project_root, output_dir)
        os.makedirs(output_dir, exist_ok=True)
        return output_dir

    default_dir = os.path.join(project_root, base_name)
    if not os.path.exists(default_dir):
        os.makedirs(default_dir, exist_ok=True)
        return default_dir

    idx = 2
    while True:
        candidate = os.path.join(project_root, f"{base_name}_{idx}")
        if not os.path.exists(candidate):
            os.makedirs(candidate, exist_ok=True)
            return candidate
        idx += 1
