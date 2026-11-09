import re

from comfy_api.latest import io


LORA_PATTERN = re.compile(r"<lora:[^>]+>")
WEIGHTED_TAG_PATTERN = re.compile(r"^\((.+):(-?(?:\d+\.?\d*|\.\d+))\)$", re.DOTALL)


class MagikTextCleanup(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="MagikTextCleanup",
            display_name="Magik Text Cleanup",
            category="adaptiveprompts/processing",
            inputs=[
                io.String.Input("string"),
                io.Boolean.Input(
                    "cleanup_commas",
                    default=True,
                    tooltip="cleans up extra commas if there is no tag present between them",
                ),
                io.Boolean.Input(
                    "cleanup_whitespace",
                    default=True,
                    tooltip="cleans up leading/trailing whitespace, then excessive whitespace",
                ),
                io.Boolean.Input(
                    "remove_lora_tags",
                    default=False,
                    tooltip="completely removes lora tags from the string",
                ),
                io.Combo.Input(
                    "cleanup_newlines",
                    options=["false", "space", "comma"],
                    default="false",
                    tooltip="replaces newlines (\\n) with a space or comma",
                ),
                io.Boolean.Input(
                    "remove_empty_lines",
                    default=False,
                    tooltip="removes every empty line from the string",
                ),
                io.Combo.Input(
                    "fix_brackets",
                    options=["false", "(parenthesis)", "[brackets]", "([both])"],
                    default="([both])",
                    tooltip="removes stray, unpaired brackets/parentheses",
                ),
                # NOTE: keep new inputs at the END so saved V1 workflows (positional
                # widget values) keep loading correctly.
                io.Boolean.Input(
                    "remove_duplicates",
                    default=False,
                    tooltip=(
                        "removes duplicate tags (comma-separated). Case-insensitive; "
                        "tag, (tag), [tag] and (tag:1.2) count as the same tag and the "
                        "highest weight is kept ((x)=1.1, [x]=0.91). Commas inside "
                        "(...) or [...] do not split tags"
                    ),
                ),
            ],
            outputs=[
                io.String.Output(display_name="STRING"),
            ],
        )

    @staticmethod
    def _remove_unmatched(s: str, open_ch: str, close_ch: str) -> str:
        """Removes unmatched brackets of one type while preserving valid pairs."""
        stack = []
        remove_idx = set()

        for i, ch in enumerate(s):
            if ch == open_ch:
                stack.append(i)
            elif ch == close_ch:
                if stack:
                    stack.pop()  # matched → keep both
                else:
                    remove_idx.add(i)  # unmatched close → remove later

        # any opens still in stack are unmatched → remove them
        remove_idx.update(stack)

        # build cleaned string
        return "".join(ch for i, ch in enumerate(s) if i not in remove_idx)

    @staticmethod
    def _split_top_level_commas(s: str) -> list:
        """Splits on commas outside of (...) / [...]; backslash-escaped chars are ignored.

        Segments keep their original surrounding whitespace so that
        ",".join(segments) reproduces the input exactly.
        """
        parts = []
        depth = 0
        start = 0
        i = 0
        while i < len(s):
            ch = s[i]
            if ch == "\\":
                i += 2  # skip escaped char, e.g. \( or \)
                continue
            if ch in "([":
                depth += 1
            elif ch in ")]":
                depth = max(0, depth - 1)
            elif ch == "," and depth == 0:
                parts.append(s[start:i])
                start = i + 1
            i += 1
        parts.append(s[start:])
        return parts

    @staticmethod
    def _wraps_whole(s: str, open_ch: str, close_ch: str) -> bool:
        """True if s[0] is open_ch and it closes at the very last char (escapes ignored)."""
        if len(s) < 2 or s[0] != open_ch or s[-1] != close_ch:
            return False
        depth = 0
        i = 0
        while i < len(s):
            ch = s[i]
            if ch == "\\":
                i += 2
                continue
            if ch == open_ch:
                depth += 1
            elif ch == close_ch:
                depth -= 1
                if depth < 0 or (depth == 0 and i != len(s) - 1):
                    return False
            i += 1
        return depth == 0

    @classmethod
    def _parse_tag(cls, part: str):
        """Returns (normalized_key, weight), A1111-style.

        '(tag:1.2)' -> 1.2 | '(tag)' -> 1.1 | '[tag]' -> 1/1.1 | plain -> 1.0.
        Nested layers multiply, e.g. '((tag))' -> 1.21.
        """
        text = part.strip()
        weight = 1.0
        while True:
            if cls._wraps_whole(text, "(", ")"):
                m = WEIGHTED_TAG_PATTERN.match(text)
                inner = m.group(1) if m else text[1:-1]
                factor = float(m.group(2)) if m else 1.1
            elif cls._wraps_whole(text, "[", "]"):
                inner, factor = text[1:-1], 1 / 1.1
            else:
                break
            if not inner.strip():  # '()' / '[]' stay literal
                break
            weight *= factor
            text = inner.strip()
        return " ".join(text.split()).lower(), round(weight, 4)

    @classmethod
    def _remove_duplicate_tags(cls, s: str) -> str:
        """Removes repeated tags, ignoring '(x:weight)' wrappers.

        The first occurrence keeps its position; its content is replaced by the
        highest-weighted variant (ties -> earliest). Plain tags weigh 1.0.
        """
        parts = cls._split_top_level_commas(s)
        parsed = [cls._parse_tag(p) if p.strip() else None for p in parts]

        # Pass 1: best (highest-weight) variant per tag
        best = {}
        for part, info in zip(parts, parsed):
            if info is None:
                continue
            key, weight = info
            if key not in best or weight > best[key][0]:
                best[key] = (weight, part.strip())

        # Pass 2: emit first occurrence (with best content), drop the rest
        seen = set()
        kept = []
        for part, info in zip(parts, parsed):
            if info is None:  # empty segments are left to cleanup_commas
                kept.append(part)
                continue
            key = info[0]
            if key in seen:
                continue
            seen.add(key)
            lead = part[:len(part) - len(part.lstrip())]
            trail = part[len(part.rstrip()):]
            kept.append(lead + best[key][1] + trail)
        return ",".join(kept)

    @classmethod
    def execute(cls, string, cleanup_commas, cleanup_newlines, remove_empty_lines,
                cleanup_whitespace, remove_lora_tags, fix_brackets,
                remove_duplicates=False):
        # Stage 1: Remove LoRA tags
        if remove_lora_tags:
            string = re.sub(LORA_PATTERN, "", string)

        # Stage 2: Replace newlines with space
        if cleanup_newlines == "space":
            string = string.replace("\n", " ")
        elif cleanup_newlines == "comma":
            string = string.replace("\n", ", ")

        # Stage 2b: Remove empty lines
        if remove_empty_lines:
            string = "\n".join(
                line for line in string.split("\n") if line.strip() != ""
            )

        # Stage 3: Remove empty comma sections
        if cleanup_commas:
            # Iteratively remove leading commas
            while re.match(r"^[ \t]*,[ \t]*", string):
                string = re.sub(r"^[ \t]*,[ \t]*", "", string)

            # Iteratively remove trailing commas
            while re.search(r"[ \t]*,[ \t]*$", string):
                string = re.sub(r"[ \t]*,[ \t]*$", "", string)

            # Remove empty comma sections inside the string
            while re.search(r",[ \t]*,", string):
                string = re.sub(r",[ \t]*,", ",", string)

        # Stage 4: Fix stray brackets
        if fix_brackets != "false":
            if fix_brackets in ("(parenthesis)", "([both])"):
                string = cls._remove_unmatched(string, "(", ")")
            if fix_brackets in ("[brackets]", "([both])"):
                string = cls._remove_unmatched(string, "[", "]")

        # Stage 4b: Remove duplicate tags
        if remove_duplicates:
            string = cls._remove_duplicate_tags(string)

        # Stage 5: Whitespace cleanup
        if cleanup_whitespace:
            string = string.strip(" \t")
            string = re.sub(r"[ \t]{2,}", " ", string)              # collapse spaces/tabs
            string = re.sub(r"[ \t]*,[ \t]*", ", ", string)         # normalize comma spacing

        return io.NodeOutput(string)
