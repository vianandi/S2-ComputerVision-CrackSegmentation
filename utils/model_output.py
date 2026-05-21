def extract_logits(outputs):
    if isinstance(outputs, dict):
        if "logits" in outputs:
            return outputs["logits"]
        if "out" in outputs:
            return outputs["out"]
        if "main" in outputs:
            return outputs["main"]

    if isinstance(outputs, (list, tuple)) and len(outputs) > 0:
        return outputs[0]

    return outputs
