from typing import Any

from pydantic import BaseModel


class ToolRegistryEntry(BaseModel):
    namespace: str
    name: str
    description: str
    input_schema: dict


def entry_text(entry: ToolRegistryEntry) -> str:
    """Text representation embedded for retrieval."""
    fields = list(entry.input_schema.get("properties", {}).keys())
    return (
        f"{entry.namespace}.{entry.name}: {entry.description}. "
        f"Input fields: {fields}"
    )


def build_registry(tools_by_namespace: dict[str, list[Any]]) -> list[ToolRegistryEntry]:
    """Build a typed registry from per-namespace tool lists.

    Each tool is expected to be a LangChain StructuredTool with:
      .name (str), .description (str), .args_schema (Pydantic model class).
    """
    entries: list[ToolRegistryEntry] = []
    for namespace, tools in tools_by_namespace.items():
        for tool in tools:
            try:
                schema = tool.args_schema.model_json_schema()
            except AttributeError:
                schema = {}
            entries.append(
                ToolRegistryEntry(
                    namespace=namespace,
                    name=tool.name,
                    description=tool.description or "",
                    input_schema=schema,
                )
            )
    return entries
