"""Codestral generative query → identifier expansion (Phase 3.0).

Uses the same loaded Codestral encoder model in generation mode. Returns a
list of identifier strings the model thinks are likely involved in the bug
described by the issue text.

Failure mode: if .generate() throws or output parses to no valid identifiers,
returns []. Caller logs and proceeds (Phase 3.0 degrades to Phase 2.9 on
that instance).
"""
from __future__ import annotations

import re

import torch

from src.encoder import MambaEncoder
from src.utils import get_logger

log = get_logger("llm_expansion")


PROMPT = """Given the following GitHub issue, list the specific, distinctive Python class names, method names, or constant names most likely involved in the underlying bug. Prefer:
- Specific class names (e.g., ResolverMatch, ChangeList)
- Distinctive method names (e.g., make_hashable, resolve_pattern)
- Specific constants (e.g., FILE_UPLOAD_PERMISSIONS)

Avoid:
- Generic Python terms (e.g., self, cls, init)
- Common standard library identifiers (e.g., dict, list)
- Common configuration terms unless distinctive (e.g., DEBUG, URL)

Output one identifier per line. No prose, no numbering, no explanations. Maximum 15 identifiers.

Issue:
{issue}

Identifiers:
"""


_IDENT_PAT = re.compile(r"^[A-Za-z_][A-Za-z0-9_.]*$")


def _parse_output(text: str, max_n: int) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for line in text.splitlines():
        line = line.strip().strip("-*•0123456789. \t`'\"")
        if not line:
            continue
        # Take first whitespace-separated token; many models emit "Name -- comment"
        tok = line.split()[0].strip("`'\".,()[]:")
        if _IDENT_PAT.match(tok) and tok not in seen:
            seen.add(tok)
            out.append(tok)
            if len(out) >= max_n:
                break
    return out


@torch.no_grad()
def expand_query_to_identifiers(issue_text: str,
                                 encoder: MambaEncoder,
                                 max_identifiers: int = 15,
                                 max_new_tokens: int = 200,
                                 max_issue_chars: int = 4000,
                                 ) -> tuple[list[str], str]:
    """Generate identifier candidates for an issue.

    Returns (identifiers, raw_decoded_output). Empty list on failure.
    """
    prompt = PROMPT.format(issue=issue_text[:max_issue_chars])
    tok = encoder.tokenizer(prompt, return_tensors="pt", truncation=True,
                            max_length=4096).to(encoder.device)

    pad_id = encoder.tokenizer.pad_token_id or encoder.tokenizer.eos_token_id

    try:
        gen = encoder.model.generate(
            **tok,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=1.0,
            pad_token_id=pad_id,
            eos_token_id=encoder.tokenizer.eos_token_id,
        )
    except Exception as e:
        log.warning("generate() failed: %s: %s", type(e).__name__, e)
        return [], ""

    new_ids = gen[0, tok["input_ids"].shape[1]:]
    raw = encoder.tokenizer.decode(new_ids, skip_special_tokens=True)
    idents = _parse_output(raw, max_identifiers)
    return idents, raw
