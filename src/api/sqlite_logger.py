import logging
from src.api.database import SessionLocal, AppLog

class SQLiteHandler(logging.Handler):
    def emit(self, record):
        try:
            db = SessionLocal()
            log_entry = AppLog(
                level=record.levelname,
                message=self.format(record),
                module=record.module
            )
            db.add(log_entry)
            db.commit()
            db.close()
        except Exception:
            pass
