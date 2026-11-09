from typing_extensions import override
from comfy_api.latest import ComfyExtension, io

from .magik_text_cleanup import MagikTextCleanup


class MagikTextCleanupExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [MagikTextCleanup]


async def comfy_entrypoint() -> MagikTextCleanupExtension:
    return MagikTextCleanupExtension()


__all__ = ["comfy_entrypoint"]
