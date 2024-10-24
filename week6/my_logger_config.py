import logging
import colorlog


def config_my_logger():

    # 로거 생성
    logger = logging.getLogger()

    # 기존 핸들러 삭제
    if logger.hasHandlers():
        logger.handlers.clear()

    logger.setLevel(logging.DEBUG)

    # 콘솔 핸들러 생성
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)

    # Colorlog 포매터 추가 (로그 레벨만 색상 적용)
    formatter = colorlog.ColoredFormatter(
        "%(asctime)s %(log_color)s%(levelname)s%(reset)s: %(message)s",
        log_colors={
            "DEBUG": "blue",
            "INFO": "cyan",
            "WARNING": "yellow",
            "ERROR": "bold_red",
        },
        datefmt="%m/%d/%Y %I:%M:%S %p",  # 날짜 형식 지정
    )

    console_handler.setFormatter(formatter)

    # 핸들러를 로거에 추가
    logger.addHandler(console_handler)
    return logger
