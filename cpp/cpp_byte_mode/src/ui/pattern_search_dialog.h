#pragma once

#include <QDialog>
#include <QPair>
#include <QVector>
#include <QtTypes>

class QLabel;
class QLineEdit;
class QPushButton;
class QComboBox;
class QSpinBox;
class QTableWidget;
class QCloseEvent;

namespace bitabyte::data {
class ByteDataSource;
}

namespace bitabyte::ui {

class PatternSearchDialog final : public QDialog {
    Q_OBJECT

public:
    explicit PatternSearchDialog(
        data::ByteDataSource* dataSource,
        const QString& initialPatternText = {},
        QWidget* parent = nullptr
    );

signals:
    void highlightRangesChanged(const QVector<QPair<qsizetype, qsizetype>>& absoluteBitRanges);
    void focusMatchRequested(qsizetype startBit, qsizetype bitCount);
    void frameByPatternRequested(const QString& patternText);

protected:
    void closeEvent(QCloseEvent* closeEvent) override;

private slots:
    void runSearch();
    void handleCurrentMatchChanged();
    void requestFrameByPattern();
    void discoverRepeatedPatterns();
    void handleDiscoveredPatternChanged();

private:
    void refreshResultsTable();
    void clearResults(const QString& summaryText);
    void clearDiscoveredPatterns(const QString& summaryText);
    [[nodiscard]] bool hasCurrentMatch() const;
    [[nodiscard]] qsizetype currentMatchStartBit() const;

    data::ByteDataSource* dataSource_ = nullptr;
    QLineEdit* patternLineEdit_ = nullptr;
    QLabel* summaryLabel_ = nullptr;
    QTableWidget* resultsTableWidget_ = nullptr;
    QPushButton* findButton_ = nullptr;
    QPushButton* frameButton_ = nullptr;
    QSpinBox* repeatedPatternSizeSpinBox_ = nullptr;
    QComboBox* repeatedPatternUnitComboBox_ = nullptr;
    QLabel* repeatedPatternSummaryLabel_ = nullptr;
    QTableWidget* repeatedPatternsTableWidget_ = nullptr;
    QPushButton* discoverPatternsButton_ = nullptr;
    qsizetype patternBitCount_ = 0;
    QVector<qsizetype> matchStartBits_;
};

}  // namespace bitabyte::ui
