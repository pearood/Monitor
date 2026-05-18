import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont, QFontDatabase, QIcon

from app.core.database import Database
from app.core.remote_database import RemoteDatabase
from app.ui.new_login_window import NewLoginRegisterWindow
from config.config import API_BASE_URL, APP_NAME, ICON_PATH, LOGO_PATH, REMOTE_API_ENABLED


class Application:
    def __init__(self):
        self.app = QApplication(sys.argv)
        self.app.setApplicationName(APP_NAME)
        self.app.setWindowIcon(self._load_app_icon())
        self.app.setFont(self._pick_default_font())
        self.app.aboutToQuit.connect(self.cleanup)
        
        self.db = self._build_database()
        
        self.login_window = None
        self.main_window = None

    def _build_database(self):
        if REMOTE_API_ENABLED:
            return RemoteDatabase(API_BASE_URL)
        return Database()

    def _pick_default_font(self):
        available_fonts = set(QFontDatabase().families())
        for family in ["Microsoft YaHei", "PingFang SC", "Hiragino Sans GB", "Arial Unicode MS"]:
            if family in available_fonts:
                return QFont(family, 10)
        return QFont(self.app.font().family(), 10)

    def _load_app_icon(self):
        for path in [ICON_PATH, LOGO_PATH]:
            if path and os.path.exists(path):
                icon = QIcon(path)
                if not icon.isNull():
                    return icon
        return self.app.windowIcon()
    
    def run(self):
        self.show_login()
        return self.app.exec_()
    
    def show_login(self):
        if self.main_window:
            self.main_window.close()
            self.main_window = None
        
        self.login_window = NewLoginRegisterWindow(self.db)
        self.login_window.login_success.connect(self.handle_login_success)
        self.login_window.show()
    
    def handle_login_success(self, user_info):
        self.login_window.close()
        self.login_window = None
        
        if user_info['user_type'] == 'student':
            from app.ui.student_window import StudentMainWindow

            self.main_window = StudentMainWindow(user_info, self.db)
        elif user_info['user_type'] == 'parent':
            from app.ui.parent_window import ParentMainWindow

            self.main_window = ParentMainWindow(user_info, self.db)
        else:
            from app.ui.teacher_window import TeacherMainWindow

            self.main_window = TeacherMainWindow(user_info, self.db)
        
        self.main_window.logout_signal.connect(self.show_login)
        self.main_window.show()

    def cleanup(self):
        if self.main_window and hasattr(self.main_window, "voice_llm_controller"):
            try:
                self.main_window.voice_llm_controller.shutdown()
            except Exception:
                pass


def main():
    app = Application()
    sys.exit(app.run())


if __name__ == '__main__':
    main()
