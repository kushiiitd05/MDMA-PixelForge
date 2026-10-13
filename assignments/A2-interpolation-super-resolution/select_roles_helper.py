"""Latency-record key for a pipeline id — mirrors method_development/select_roles.latency_key (kept in sync by hand)."""


def latency_key(s: int, mid: str) -> str:
    sfx = "+proj" if mid.endswith("+proj") else ""
    if mid.startswith("ours:"):
        return f"x{s}|ours:DirectLR w96 d12" + sfx
    if mid.startswith("finetuned:"):
        return f"x{s}|pretrained:SwinIR-M" + sfx
    return f"x{s}|{mid}"
