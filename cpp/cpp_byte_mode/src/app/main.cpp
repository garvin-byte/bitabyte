#include <QApplication>

#include "ui/main_window.h"

int main(int argumentCount, char* argumentValues[]) {
    QApplication application(argumentCount, argumentValues);
    bitabyte::ui::MainWindow mainWindow;
    mainWindow.show();
    return application.exec();
}
