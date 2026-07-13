import langchain as _lc

# 兼容补丁：为当前 langchain 版本提供缺失的 debug 属性，防止崩溃
if not hasattr(_lc, "debug"):
    _lc.debug = False

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from config.arg_settings import args
from router.route import router


def create_app():

    app = FastAPI()

    origins = [
        '*',
    ]

    @app.get('/helixflow/health')
    def get_health():
        return {'status': 'OK'}

    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=['*'],
        allow_headers=['*'],
    )

    app.include_router(router)
    # app.on_event('startup')(load_config)
    return app


if __name__ == "__main__":
    import uvicorn
    app = create_app()
    uvicorn.run(app, host=args.host, port=args.port)