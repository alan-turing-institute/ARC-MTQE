from datetime import datetime


def create_now_str():
    """
    Return datetime string in YYYYMMDD_HHMMSS format.
    """

    now = datetime.now()
    return now.strftime("%Y%m%d_%H%M%S")
