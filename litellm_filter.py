# litellm_filter.py
# Strips Claude Code-specific tools that local models don't understand
# Reference in litellm config:
#   litellm_settings:
#     callbacks: ["litellm_filter.ToolFilter"]

from litellm.integrations.custom_logger import CustomLogger

class ToolFilter(CustomLogger):
    # Tools only understood by Claude (not local models)
    SKIP_TOOLS = {'Skill', 'EnterPlanMode', 'ExitPlanMode'}

    async def async_pre_call_hook(self, user_api_key_dict, cache, data, call_type):
        if 'tools' in data and data['tools']:
            data['tools'] = [
                t for t in data['tools']
                if not (
                    isinstance(t, dict)
                    and t.get('type') == 'function'
                    and t.get('function', {}).get('name') in self.SKIP_TOOLS
                )
            ]
            if not data['tools']:
                data.pop('tools', None)
                data.pop('tool_choice', None)
        return data

proxy_handler_instance = ToolFilter()
