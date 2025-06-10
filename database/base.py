from sqlmodel import Session, create_engine

from config.arg_settings import args

table_engine = create_engine(args.database_url, pool_pre_ping=True,pool_size=5, pool_timeout=1000, pool_recycle=3600)
def get_table_session():
    with Session(table_engine) as table_session:
        yield table_session