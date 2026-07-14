from datetime import datetime, timezone
from langchain_core.tools import tool
from datetime import timedelta



@tool("get_current_time")
def get_current_time() -> str:
    """
    获取当前日期和时间
    
    Returns:
        str: 格式化的当前时间字符串，格式为 YYYY-MM-DD HH:MM:SS
    """
    return datetime.now(timezone.utc).astimezone(timezone(timedelta(hours=8))).strftime("%Y-%m-%d %H:%M:%S")
