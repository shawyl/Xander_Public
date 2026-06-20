"""Dynamic prompt assembly helper for Xander.

Builds focused prompt text from selected sections so specialist review can
receive relevant context without exposing private prompt contents here.

AI status: Created with AI.
"""

from __future__ import annotations

import argparse
import logging
import os
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable


TOC_SEPARATOR_REGEX = re.compile(r"^#\s*-{10,}\s*$", re.MULTILINE)
TOC_CHAPTER_REGEX = re.compile(r"^###\s+(?P<roman>[IVXLCDM]+)\.\s+(?P<title>.+?)\s*$")
TOC_ENTRY_REGEX = re.compile(r"^-\s+\[(?P<marker>[MO])\]\s+(?P<title>.+?)\s*$")
SECTION_HEADER_REGEX = re.compile(
    r"^#\s*=+\s+(?P<title>.+?)\s+=+\s*$",
    re.MULTILINE,
)
PRIORITY_RULE_PREFIX_REGEX = re.compile(
    r"^Priority Rule\s+(?:X|\d+):\s*(?P<title>.+?)\s*$",
    re.IGNORECASE,
)
PRIORITY_RULE_ANY_REGEX = re.compile(r"Priority Rule\s+X:", re.IGNORECASE)
TOC_MARKER_REGEX = re.compile(r"\[(?:M|O)\]")


class DynamicPromptError(RuntimeError):
    """Raised when the dynamic prompt cannot be safely generated."""


@dataclass(frozen=True)
class TocEntry:
    chapter_title: str
    marker: str
    title: str
    order: int

    @property
    def is_mandatory(self) -> bool:
        return self.marker == "M"

    @property
    def is_optional(self) -> bool:
        return self.marker == "O"

    @property
    def is_priority_rule(self) -> bool:
        return (
            self.chapter_title.lower().startswith("priority rules")
            or PRIORITY_RULE_PREFIX_REGEX.match(self.title) is not None
        )

    @property
    def base_title(self) -> str:
        return strip_priority_rule_prefix(self.title)


@dataclass
class TocChapter:
    title: str
    entries: list[TocEntry] = field(default_factory=list)


@dataclass
class SectionBlock:
    title: str
    text: str


@dataclass
class DynamicPromptResult:
    prompt: str
    selected_sections: list[str]
    unknown_sections: list[str]
    duplicate_sections: list[str]
    included_mandatory_count: int
    included_optional_count: int
    removed_optional_count: int
    temp_file_path: str | None
    prompt_size_bytes: int
    prompt_size_kb: float
    warnings: list[str] = field(default_factory=list)


def load_dynamic_template(prompts_dir: str | os.PathLike[str] | None = None) -> str:
    template_path = get_prompts_dir(prompts_dir) / "DYNAMIC.txt"
    with open(template_path, "r", encoding="utf-8") as file:
        return file.read()


def get_prompts_dir(prompts_dir: str | os.PathLike[str] | None = None) -> Path:
    if prompts_dir is not None:
        return Path(prompts_dir)
    return Path(__file__).resolve().parent / "PROMPTS"


def strip_priority_rule_prefix(section_name: str) -> str:
    match = PRIORITY_RULE_PREFIX_REGEX.match(section_name.strip())
    if match:
        return match.group("title").strip()
    return section_name.strip()


def canonical_section_name(section_name: str) -> str:
    cleaned = strip_priority_rule_prefix(section_name)
    cleaned = cleaned.replace("\u201c", '"').replace("\u201d", '"')
    cleaned = cleaned.replace("\u2018", "'").replace("\u2019", "'")
    cleaned = re.sub(r"\s+", " ", cleaned.strip())
    return cleaned.casefold()


def _windows_safe_filename(name: str) -> str:
    return re.sub(r'[<>:"/\\|?*]', "", name).strip()


def _section_file_candidates(entry: TocEntry) -> list[str]:
    names = [entry.base_title, entry.title]

    lowered = entry.base_title.lower()
    if lowered.startswith("hard exclusion:"):
        names.append(entry.base_title.split(":", 1)[1].strip())
    if lowered.startswith("priority rule") and ":" in entry.base_title:
        names.append(entry.base_title.split(":", 1)[1].strip())
    if lowered.startswith("gating rule") and ":" in entry.base_title:
        names.append(entry.base_title.split(":", 1)[1].strip())

    candidates = []
    for name in names:
        safe_name = _windows_safe_filename(name)
        if safe_name and safe_name not in candidates:
            candidates.append(safe_name)
    return candidates


def parse_dynamic_toc(template_text: str) -> list[TocChapter]:
    separators = list(TOC_SEPARATOR_REGEX.finditer(template_text))
    if len(separators) < 2:
        raise DynamicPromptError("DYNAMIC.txt must contain at least two table-of-contents separators.")

    toc_block = template_text[separators[0].end():separators[1].start()]
    chapters: list[TocChapter] = []
    current_chapter: TocChapter | None = None
    entry_order = 0

    for raw_line in toc_block.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        chapter_match = TOC_CHAPTER_REGEX.match(line)
        if chapter_match:
            current_chapter = TocChapter(title=chapter_match.group("title").strip())
            chapters.append(current_chapter)
            continue

        entry_match = TOC_ENTRY_REGEX.match(line)
        if entry_match and current_chapter is not None:
            current_chapter.entries.append(
                TocEntry(
                    chapter_title=current_chapter.title,
                    marker=entry_match.group("marker"),
                    title=entry_match.group("title").strip(),
                    order=entry_order,
                )
            )
            entry_order += 1

    if not chapters:
        raise DynamicPromptError("No table-of-contents chapters found in DYNAMIC.txt.")

    return chapters


def extract_section_blocks(template_text: str) -> dict[str, SectionBlock]:
    separators = list(TOC_SEPARATOR_REGEX.finditer(template_text))
    if len(separators) < 2:
        raise DynamicPromptError("DYNAMIC.txt must contain a body separator after the table of contents.")

    body = template_text[separators[1].end():]
    matches = list(SECTION_HEADER_REGEX.finditer(body))
    blocks: dict[str, SectionBlock] = {}

    for index, match in enumerate(matches):
        block_start = match.start()
        block_end = matches[index + 1].start() if index + 1 < len(matches) else len(body)
        title = match.group("title").strip()
        block_text = body[block_start:block_end].strip()
        blocks[canonical_section_name(title)] = SectionBlock(title=title, text=block_text)

    return blocks


def _build_entry_lookup(chapters: Iterable[TocChapter]) -> dict[str, TocEntry]:
    lookup: dict[str, TocEntry] = {}
    for chapter in chapters:
        for entry in chapter.entries:
            lookup[canonical_section_name(entry.title)] = entry
            lookup[canonical_section_name(entry.base_title)] = entry
    return lookup


def normalize_router_sections(
    router_response: Any,
    chapters: Iterable[TocChapter],
    logger: logging.Logger | None = None,
) -> tuple[set[int], list[str], list[str], list[str], list[str]]:
    warnings: list[str] = []
    unknown_sections: list[str] = []
    duplicate_sections: list[str] = []
    selected_entry_orders: set[int] = set()
    selected_display_names: list[str] = []
    entry_lookup = _build_entry_lookup(chapters)

    if router_response is None:
        _add_warning(warnings, logger, "Sections router response is empty; using mandatory sections only.")
        return selected_entry_orders, selected_display_names, unknown_sections, duplicate_sections, warnings

    if isinstance(router_response, dict):
        if "Sections" not in router_response:
            _add_warning(warnings, logger, 'Sections router response missing "Sections"; using mandatory sections only.')
            return selected_entry_orders, selected_display_names, unknown_sections, duplicate_sections, warnings
        raw_sections = router_response.get("Sections")
    else:
        raw_sections = router_response

    if raw_sections is None:
        raw_sections = []

    if not isinstance(raw_sections, list):
        _add_warning(warnings, logger, 'Sections router response "Sections" is not a list; using mandatory sections only.')
        return selected_entry_orders, selected_display_names, unknown_sections, duplicate_sections, warnings

    seen_raw_names: set[str] = set()
    for raw_section in raw_sections:
        if not isinstance(raw_section, str):
            unknown_sections.append(str(raw_section))
            _add_warning(warnings, logger, f"Ignoring non-string router section: {raw_section!r}")
            continue

        section = raw_section.strip()
        if not section:
            continue

        raw_key = canonical_section_name(section)
        if raw_key in seen_raw_names:
            duplicate_sections.append(section)
            _add_warning(warnings, logger, f"Ignoring duplicate router section: {section}")
            continue
        seen_raw_names.add(raw_key)

        entry = entry_lookup.get(raw_key)
        if entry is None:
            unknown_sections.append(section)
            _add_warning(warnings, logger, f"Ignoring unknown router section: {section}")
            continue

        if entry.order in selected_entry_orders:
            duplicate_sections.append(section)
            _add_warning(warnings, logger, f"Ignoring duplicate router section alias: {section}")
            continue

        selected_entry_orders.add(entry.order)
        selected_display_names.append(entry.base_title)

    return selected_entry_orders, selected_display_names, unknown_sections, duplicate_sections, warnings


def _add_warning(warnings: list[str], logger: logging.Logger | None, message: str) -> None:
    warnings.append(message)
    if logger is not None:
        logger.warning(f"[DynamicPrompt] {message}")


def _load_section_file_block(
    entry: TocEntry,
    prompts_dir: Path,
    logger: logging.Logger | None,
    warnings: list[str],
) -> SectionBlock | None:
    sections_dir = prompts_dir / "Sections"
    for candidate in _section_file_candidates(entry):
        section_path = sections_dir / f"{candidate}.txt"
        if section_path.exists():
            with open(section_path, "r", encoding="utf-8") as file:
                section_content = file.read().strip()
            block_text = (
                f"# ============================== {entry.title.upper()} ==============================\n\n"
                f"{section_content}"
            )
            return SectionBlock(title=entry.title.upper(), text=block_text)

    _add_warning(
        warnings,
        logger,
        f'Selected optional section "{entry.base_title}" was not found in DYNAMIC.txt or PROMPTS/Sections.',
    )
    return None


def _get_entry_block(
    entry: TocEntry,
    section_blocks: dict[str, SectionBlock],
    prompts_dir: Path,
    logger: logging.Logger | None,
    warnings: list[str],
) -> SectionBlock | None:
    block = section_blocks.get(canonical_section_name(entry.title))
    if block is not None:
        return block

    block = section_blocks.get(canonical_section_name(entry.base_title))
    if block is not None:
        return block

    return _load_section_file_block(entry, prompts_dir, logger, warnings)


def cleanup_toc_markers(line: str) -> str:
    return TOC_MARKER_REGEX.sub("", line).replace("-  ", "- ")


def remove_empty_chapters(chapters: Iterable[TocChapter], included_orders: set[int]) -> list[TocChapter]:
    filtered_chapters: list[TocChapter] = []
    for chapter in chapters:
        included_entries = [entry for entry in chapter.entries if entry.order in included_orders]
        if included_entries:
            filtered_chapters.append(TocChapter(title=chapter.title, entries=included_entries))
    return filtered_chapters


def _to_roman(number: int) -> str:
    values = [
        (1000, "M"),
        (900, "CM"),
        (500, "D"),
        (400, "CD"),
        (100, "C"),
        (90, "XC"),
        (50, "L"),
        (40, "XL"),
        (10, "X"),
        (9, "IX"),
        (5, "V"),
        (4, "IV"),
        (1, "I"),
    ]
    result = []
    remaining = number
    for value, numeral in values:
        while remaining >= value:
            result.append(numeral)
            remaining -= value
    return "".join(result)


def renumber_chapters(chapters: Iterable[TocChapter]) -> str:
    lines = ["## TABLE OF CONTENTS ##", ""]
    for index, chapter in enumerate(chapters, start=1):
        lines.append(f"### {_to_roman(index)}. {chapter.title}")
        for entry in chapter.entries:
            lines.append(f"- {entry.title}")
        lines.append("")
    return "\n".join(lines).rstrip()


def _renumber_priority_text(text: str, priority_number: int) -> str:
    return PRIORITY_RULE_ANY_REGEX.sub(f"Priority Rule {priority_number}:", text)


def renumber_priority_rules(
    chapters: Iterable[TocChapter],
    body_blocks: list[tuple[TocEntry, str]],
) -> tuple[list[TocChapter], list[str]]:
    priority_numbers: dict[int, int] = {}
    next_priority_number = 1

    for chapter in chapters:
        for entry in chapter.entries:
            if entry.is_priority_rule:
                priority_numbers[entry.order] = next_priority_number
                next_priority_number += 1

    renumbered_chapters: list[TocChapter] = []
    for chapter in chapters:
        new_entries = []
        for entry in chapter.entries:
            title = entry.title
            if entry.order in priority_numbers:
                title = _renumber_priority_text(title, priority_numbers[entry.order])
            new_entries.append(
                TocEntry(
                    chapter_title=entry.chapter_title,
                    marker=entry.marker,
                    title=title,
                    order=entry.order,
                )
            )
        renumbered_chapters.append(TocChapter(title=chapter.title, entries=new_entries))

    renumbered_blocks = []
    for entry, block_text in body_blocks:
        if entry.order in priority_numbers:
            block_text = _renumber_priority_text(block_text, priority_numbers[entry.order])
        renumbered_blocks.append(block_text)

    return renumbered_chapters, renumbered_blocks


def _validate_generated_prompt(prompt: str, mandatory_count: int) -> None:
    if not prompt.strip():
        raise DynamicPromptError("Generated dynamic prompt is empty.")
    if mandatory_count <= 0:
        raise DynamicPromptError("Generated dynamic prompt is missing mandatory sections.")

    separators = list(TOC_SEPARATOR_REGEX.finditer(prompt))
    if len(separators) < 2:
        raise DynamicPromptError("Generated dynamic prompt is missing table-of-contents separators.")

    toc_text = prompt[separators[0].end():separators[1].start()]
    body_text = prompt[separators[1].end():]

    if TOC_MARKER_REGEX.search(toc_text):
        raise DynamicPromptError("Generated dynamic prompt table of contents still contains [M] or [O] markers.")
    if PRIORITY_RULE_ANY_REGEX.search(toc_text):
        raise DynamicPromptError("Generated dynamic prompt table of contents still contains Priority Rule X.")
    if PRIORITY_RULE_ANY_REGEX.search(body_text):
        raise DynamicPromptError("Generated dynamic prompt body still contains Priority Rule X.")

    toc_lines = [line.strip() for line in toc_text.splitlines() if line.strip()]
    chapter_indexes = [index for index, line in enumerate(toc_lines) if TOC_CHAPTER_REGEX.match(line)]
    for index, line_index in enumerate(chapter_indexes):
        next_line_index = chapter_indexes[index + 1] if index + 1 < len(chapter_indexes) else len(toc_lines)
        chapter_contents = toc_lines[line_index + 1:next_line_index]
        if not any(line.startswith("- ") for line in chapter_contents):
            raise DynamicPromptError(f"Generated dynamic prompt contains an empty chapter: {toc_lines[line_index]}")


def write_temp_dynamic_prompt(
    prompt: str,
    prompts_dir: str | os.PathLike[str] | None = None,
    timestamp: datetime | None = None,
) -> str:
    output_dir = get_prompts_dir(prompts_dir) / "TEMP_DYNAMIC"
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = timestamp or datetime.now()
    filename = f"dynamic_prompt_{timestamp.strftime('%Y%m%d_%H%M%S_%f')}.txt"
    output_path = output_dir / filename
    with open(output_path, "w", encoding="utf-8", newline="\n") as file:
        file.write(prompt)
    return str(output_path)


def log_dynamic_prompt_stats(result: DynamicPromptResult, logger: logging.Logger | None = None) -> None:
    lines = [
        f"[DynamicPrompt] Selected sections: {result.selected_sections}",
        f"[DynamicPrompt] Included mandatory sections count: {result.included_mandatory_count}",
        f"[DynamicPrompt] Included optional sections count: {result.included_optional_count}",
        f"[DynamicPrompt] Removed optional sections count: {result.removed_optional_count}",
        f"[DynamicPrompt] Final prompt file path: {result.temp_file_path}",
        f"[DynamicPrompt] Final prompt size in bytes: {result.prompt_size_bytes}",
        f"[DynamicPrompt] Approximate final prompt size in KB: {result.prompt_size_kb:.2f}",
    ]
    for line in lines:
        if logger is not None:
            logger.info(line)
        else:
            print(line)


def build_dynamic_prompt(
    selected_sections: Any,
    prompts_dir: str | os.PathLike[str] | None = None,
    logger: logging.Logger | None = None,
    write_temp: bool = True,
) -> DynamicPromptResult:
    prompts_path = get_prompts_dir(prompts_dir)
    template_text = load_dynamic_template(prompts_path)
    chapters = parse_dynamic_toc(template_text)
    section_blocks = extract_section_blocks(template_text)

    (
        selected_entry_orders,
        selected_display_names,
        unknown_sections,
        duplicate_sections,
        warnings,
    ) = normalize_router_sections(selected_sections, chapters, logger)

    included_orders: set[int] = set()
    included_mandatory_count = 0
    included_optional_count = 0
    removed_optional_count = 0
    body_blocks: list[tuple[TocEntry, str]] = []
    missing_mandatory_sections: list[str] = []

    for chapter in chapters:
        for entry in chapter.entries:
            include_entry = False
            if entry.is_mandatory:
                include_entry = True
            elif entry.order in selected_entry_orders:
                include_entry = True

            if not include_entry:
                if entry.is_optional:
                    removed_optional_count += 1
                continue

            block = _get_entry_block(entry, section_blocks, prompts_path, logger, warnings)
            if block is None:
                if entry.is_mandatory:
                    missing_mandatory_sections.append(entry.base_title)
                else:
                    removed_optional_count += 1
                continue

            included_orders.add(entry.order)
            body_blocks.append((entry, block.text))
            if entry.is_mandatory:
                included_mandatory_count += 1
            else:
                included_optional_count += 1

    if missing_mandatory_sections:
        raise DynamicPromptError(
            "Generated dynamic prompt is missing mandatory sections: "
            + ", ".join(missing_mandatory_sections)
        )

    filtered_chapters = remove_empty_chapters(chapters, included_orders)
    renumbered_chapters, renumbered_body_blocks = renumber_priority_rules(filtered_chapters, body_blocks)
    toc_text = cleanup_toc_markers(renumber_chapters(renumbered_chapters))
    body_text = "\n\n\n\n\n\n".join(block.strip() for block in renumbered_body_blocks if block.strip())
    prompt = (
        "# ------------------------------------------------------------------------------------------\n\n"
        f"{toc_text}\n\n"
        "# ------------------------------------------------------------------------------------------\n\n"
        f"{body_text}\n"
    )

    _validate_generated_prompt(prompt, included_mandatory_count)

    temp_file_path = write_temp_dynamic_prompt(prompt, prompts_path) if write_temp else None
    prompt_size_bytes = os.path.getsize(temp_file_path) if temp_file_path else len(prompt.encode("utf-8"))
    result = DynamicPromptResult(
        prompt=prompt,
        selected_sections=selected_display_names,
        unknown_sections=unknown_sections,
        duplicate_sections=duplicate_sections,
        included_mandatory_count=included_mandatory_count,
        included_optional_count=included_optional_count,
        removed_optional_count=removed_optional_count,
        temp_file_path=temp_file_path,
        prompt_size_bytes=prompt_size_bytes,
        prompt_size_kb=prompt_size_bytes / 1024,
        warnings=warnings,
    )
    log_dynamic_prompt_stats(result, logger)
    return result


def _dry_run_case(name: str, selected_sections: Any, prompts_dir: Path) -> DynamicPromptResult:
    print(f"\n--- Dynamic prompt dry run: {name} ---")
    result = build_dynamic_prompt(selected_sections, prompts_dir=prompts_dir, logger=None, write_temp=True)
    print(f"Warnings: {len(result.warnings)}")
    for warning in result.warnings:
        print(f"- {warning}")
    return result


def dry_run_dynamic_prompt_tests(prompts_dir: str | os.PathLike[str] | None = None) -> None:
    prompts_path = get_prompts_dir(prompts_dir)
    selected_sections = [
        "Systemic Actor Ruleset: Donald Trump",
        "Handling Regulatory or Policy Announcements",
        "Market Context Integration Requirement (Global Rule)",
    ]
    _dry_run_case("selected optionals", selected_sections, prompts_path)
    _dry_run_case("mandatory only", [], prompts_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build and validate dynamic GPT-4o prompts.")
    parser.add_argument("--dry-run", action="store_true", help="Run local dry-run prompt generation checks.")
    parser.add_argument("--prompts-dir", default=None, help="Path to the SocialMarket PROMPTS directory.")
    args = parser.parse_args()

    if args.dry_run:
        dry_run_dynamic_prompt_tests(args.prompts_dir)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
