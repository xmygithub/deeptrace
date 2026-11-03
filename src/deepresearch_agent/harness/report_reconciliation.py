"""Remove only individually reviewed stale limitations, keeping an audit trail.

Research-stage gaps stay in checkpoints/events. A model's coverage verdict alone
does not authorize deleting every limitation in the delivered report.
"""
import copy
import hashlib
import re


_LIMIT_HEADING = re.compile(r"^(?:局限(?:性|与尚未解决的问题)?|研究局限|限制|尚未解决的问题|未解决问题|资料范围与限制|limitations?(?: and .*questions)?)$", re.I)
_ABSTRACT = re.compile(r"^(?:摘要|abstract|summary)$", re.I)


def report_paragraphs(report):
    paragraphs, eligible = {}, []
    excluded = False
    for block in report.split("\n\n"):
        body = []

        def finish():
            text = "\n".join(body).strip()
            if text and not text.startswith("> 运行说明："):
                pid = f"p{len(paragraphs) + 1}"
                paragraphs[pid] = text
                if not excluded:
                    eligible.append(pid)
            body.clear()

        for line in block.splitlines():
            heading = re.match(r"^#{1,6}\s+(.+?)\s*$", line)
            if heading:
                finish()
                title = heading.group(1).strip()
                excluded = bool(_LIMIT_HEADING.fullmatch(title) or _ABSTRACT.fullmatch(title))
            else:
                body.append(line)
        finish()
    return paragraphs, eligible


def limitation_entries(report):
    entries, active, offset, start, parts = [], False, 0, None, []

    def finish():
        nonlocal start, parts
        if start is not None:
            raw = "".join(parts)
            entries.append({"id": f"lim_{len(entries) + 1}", "text": raw.strip(), "start": start, "end": start + len(raw)})
        start, parts = None, []

    for line in report.splitlines(keepends=True):
        heading = re.match(r"^#{1,6}\s+(.+?)\s*$", line.strip())
        if heading:
            finish()
            active = bool(_LIMIT_HEADING.fullmatch(heading.group(1)))
        elif active:
            if not line.strip():
                finish()
            else:
                if re.match(r"^\s*(?:[-*]|\d+[.)])\s+", line):
                    finish()
                if start is None:
                    start = offset
                parts.append(line)
        offset += len(line)
    finish()
    return entries


def reconciliation_candidates(report, review, proposals):
    """Accept cleanup only with a covered requirement and exact body evidence."""
    paragraphs, eligible = report_paragraphs(report)
    entries = {item["id"]: item for item in limitation_entries(report)}
    research_ids = {item["id"] for item in review.get("requirements", []) if item.get("kind") != "format"}
    covered = {item["requirement_id"]: item for item in review["items"] if item["status"] == "covered"
               and (not review.get("requirements") or item["requirement_id"] in research_ids)}
    accepted, seen = [], set()
    for proposal in proposals[:30] if isinstance(proposals, list) else []:
        if not isinstance(proposal, dict):
            continue
        lid, rid = proposal.get("limitation_id"), proposal.get("requirement_id")
        pids = proposal.get("paragraph_ids", [])
        if lid in seen or lid not in entries or rid not in covered or not isinstance(pids, list) or not pids:
            continue
        if any(not isinstance(pid, str) or pid not in eligible or pid not in covered[rid].get("paragraph_ids", []) for pid in pids):
            continue
        reason = proposal.get("reason")
        if not isinstance(reason, str) or not reason.strip():
            continue
        accepted.append({"limitation_id": lid, "requirement_id": rid, "text": entries[lid]["text"],
                         "paragraph_ids": list(dict.fromkeys(pids)), "reason": reason[:1000],
                         "report_quotes": [{"paragraph_id": pid, "text": paragraphs[pid][:2000]} for pid in dict.fromkeys(pids)]})
        seen.add(lid)
    return accepted


def reconcile_report(report, review):
    """Rebase exact quote locations/hash after removing approved limitation blocks."""
    digest = lambda value: hashlib.sha256(value.encode("utf-8")).hexdigest()
    if review.get("report_hash") != digest(report):
        return report, review
    accepted = reconciliation_candidates(report, review, review.get("resolved_limitations", []))
    # Revalidation needs the model's IDs as well as its stored exact limitation.
    expected_text = {item.get("limitation_id"): item.get("text") for item in review.get("resolved_limitations", [])}
    accepted = [item for item in accepted if expected_text.get(item["limitation_id"]) == item["text"]]
    if not accepted:
        return report, review
    ids = {item["limitation_id"] for item in accepted}
    result = report
    for entry in reversed(limitation_entries(report)):
        if entry["id"] in ids:
            result = result[:entry["start"]] + result[entry["end"]:]
    # Drop only now-empty limitation headings. Body and other sections stay put.
    lines = result.splitlines(keepends=True)
    for index in reversed(range(len(lines))):
        heading = re.match(r"^#{1,6}\s+(.+?)\s*$", lines[index].strip())
        if heading and _LIMIT_HEADING.fullmatch(heading.group(1)):
            following = next((line for line in lines[index + 1:] if line.strip()), "")
            if not following or following.lstrip().startswith("#"):
                lines[index] = ""
    result = "".join(lines).strip()
    old_paragraphs, _ = report_paragraphs(report)
    new_paragraphs, _ = report_paragraphs(result)
    new_ids = {text: pid for pid, text in new_paragraphs.items()}
    rebased = copy.deepcopy(review)
    for item in rebased["items"]:
        item["paragraph_ids"] = [new_ids[old_paragraphs[pid]] for pid in item.get("paragraph_ids", [])
                                 if pid in old_paragraphs and old_paragraphs[pid] in new_ids]
        item["report_quotes"] = [{"paragraph_id": pid, "text": new_paragraphs[pid][:2000]} for pid in item["paragraph_ids"]]
        if item["report_quotes"]:
            item["report_quote"] = item["report_quotes"][0]["text"]
        elif item.get("report_quote") and item["report_quote"] not in result:
            item["report_quote"] = ""
    rebased.update(report_hash=digest(result), reviewed_report_hash=digest(report),
                   reconciliation={"removed_limitations": accepted, "method": "model-review+exact-body-validation"})
    return result, rebased
