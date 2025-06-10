from datetime import datetime
import pytz
def get_current_time_str(format_str: str = '%Y-%m-%d %H:%M:%S'):
    utc_now = datetime.utcnow()
    utc_plus_8_timezone = pytz.timezone('Asia/Shanghai')
    utc_plus_8_now = utc_now.replace(tzinfo=pytz.utc).astimezone(utc_plus_8_timezone).strftime(format_str)
    return utc_plus_8_now