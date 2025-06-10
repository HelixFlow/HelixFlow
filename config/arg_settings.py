import argparse
from config.app_config import *



parser = argparse.ArgumentParser(prog='helixflow',
                                 description='海力克斯工作流')
parser.add_argument("--host", type=str, default="0.0.0.0", help='默认的Ip')
parser.add_argument("--port", type=int, default=11110, help='默认的端口')
parser.add_argument("--database-url", type=str, default=APP_TABLE_URL, help='默认数据库地址')




args = parser.parse_args()