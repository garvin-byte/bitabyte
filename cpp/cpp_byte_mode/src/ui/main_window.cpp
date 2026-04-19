#include "ui/main_window.h"

#include "models/byte_table_model.h"
#include "models/frame_group_tree_model.h"
#include "ui/byte_table_view.h"
#include "ui/column_definitions_panel.h"
#include "ui/frame_browser_controller.h"
#include "ui/frame_field_hints_panel.h"
#include "ui/field_current_value_panel.h"
#include "ui/field_inspector_panel.h"
#include "ui/frame_grouping_panel.h"
#include "ui/framing_controller.h"
#include "ui/inspection_controller.h"
#include "ui/live_bit_viewer_widget.h"
#include "ui/main_window_internal.h"
#include "ui/pattern_search_dialog.h"

#include <QAbstractButton>
#include <QAbstractItemView>
#include <QAction>
#include <QButtonGroup>
#include <QComboBox>
#include <QDockWidget>
#include <QFileInfo>
#include <QFontMetrics>
#include <QGroupBox>
#include <QGuiApplication>
#include <QHBoxLayout>
#include <QHeaderView>
#include <QItemSelection>
#include <QItemSelectionModel>
#include <QLabel>
#include <QLineEdit>
#include <QMenu>
#include <QMenuBar>
#include <QMessageBox>
#include <QPushButton>
#include <QRadioButton>
#include <QResizeEvent>
#include <QSignalBlocker>
#include <QScrollArea>
#include <QScreen>
#include <QSpinBox>
#include <QSplitter>
#include <QStatusBar>
#include <QVBoxLayout>
#include <QWidget>
#include <QtGlobal>

#include <cmath>
#include <limits>

namespace bitabyte::ui {

MainWindow::MainWindow(QWidget* parent)
    : QMainWindow(parent),
      byteTableModel_(new models::ByteTableModel(&dataSource_, &frameLayout_, &columnDefinitions_, this)),
      frameGroupTreeModel_(new models::FrameGroupTreeModel(this)) {
    setWindowTitle(QStringLiteral("Bitabyte C++ Byte Mode"));
    setAcceptDrops(true);
    buildMenus();
    buildCentralWidget();
    frameBrowserController_ = std::make_unique<FrameBrowserController>(
        dataSource_,
        frameLayout_,
        columnDefinitions_,
        *byteTableModel_,
        *byteTableView_,
        *frameGroupTreeModel_,
        FrameBrowserController::Callbacks{
            .syncFramingControlsFromState = [this]() {
                if (framingController_ != nullptr) {
                    framingController_->syncControlsFromState();
                }
            },
            .resizeTableColumns = [this]() { resizeTableColumns(); },
            .updateLoadedFileState = [this]() { updateLoadedFileState(); },
            .updateSelectionStatus = [this]() {
                if (inspectionController_ != nullptr) {
                    inspectionController_->updateSelectionStatus();
                }
            },
            .refreshLiveBitViewer = [this]() {
                if (inspectionController_ != nullptr) {
                    inspectionController_->refreshLiveBitViewer();
                }
            },
        }
    );
    buildColumnDefinitionsDock();
    buildLiveBitViewerDock();
    inspectionController_ = std::make_unique<InspectionController>(
        dataSource_,
        frameLayout_,
        columnDefinitions_,
        *byteTableModel_,
        *byteTableView_,
        *liveBitViewerWidget_,
        *frameFieldHintsPanel_,
        *fieldInspectorPanel_,
        *fieldCurrentValuePanel_,
        *selectionInfoLabel_,
        *liveBitViewerModeGroup_,
        *liveBitViewerSizeSpinBox_,
        InspectionController::SelectionCallbacks{
            .selectedVisibleColumns = [this]() { return selectedVisibleColumns(); },
            .editableVisibleColumnsForSeed = [this](int visibleColumnIndex) {
                return editableVisibleColumnsForSeed(visibleColumnIndex);
            },
            .definitionIndexForVisibleColumns = [this](const QSet<int>& visibleColumns) {
                return definitionIndexForVisibleColumns(visibleColumns);
            },
            .topSelectedDataIndex = [this]() { return topSelectedDataIndex(); },
        },
        this
    );
    framingController_ = std::make_unique<FramingController>(
        dataSource_,
        frameLayout_,
        columnDefinitions_,
        *byteTableModel_,
        *byteTableView_,
        *frameBrowserController_,
        *inspectionController_,
        *syncPatternLineEdit_,
        *frameWidthSpinBox_,
        *frameBitOffsetSpinBox_,
        *frameSortComboBox_,
        *this,
        *statusBar(),
        FramingController::Callbacks{
            .isBitModeEnabled = [this]() {
                return byteTableModel_ != nullptr && byteTableModel_->isBitDisplayMode();
            },
            .selectedVisibleColumns = [this]() { return selectedVisibleColumns(); },
            .extractSelectionPattern = [this](QString* patternText, QString* errorMessage) {
                return extractSelectionPattern(patternText, errorMessage);
            },
            .buildDefinitionFromSelection =
                [this](const QSet<int>& visibleColumns,
                       features::columns::ByteColumnDefinition* definition,
                       QString* errorMessage) {
                    return buildDefinitionFromSelection(visibleColumns, definition, errorMessage);
                },
            .resizeTableColumns = [this]() { resizeTableColumns(); },
            .updateLoadedFileState = [this]() { updateLoadedFileState(); },
            .refreshColumnDefinitionsPanel = [this]() { refreshColumnDefinitionsPanel(); },
        }
    );
    framingController_->syncControlsFromState();
    syncFrameWidthControls();
    inspectionController_->refreshLiveBitViewer();
    inspectionController_->refreshFieldInspector();
    inspectionController_->scheduleFrameFieldHintsRefresh();
    connect(frameFieldHintsPanel_, &FrameFieldHintsPanel::bitRangeRequested, this, [this](int startBit, int endBit) {
        selectVisibleColumns(visibleColumnsForAbsoluteBitRange(startBit, endBit));
        byteTableView_->setFocus();
        if (inspectionController_ != nullptr) {
            inspectionController_->updateSelectionStatus();
            inspectionController_->refreshLiveBitViewer();
            inspectionController_->refreshFieldInspector();
        }
    });
    connect(
        frameFieldHintsPanel_,
        &FrameFieldHintsPanel::addColumnRequested,
        this,
        [this](int startBit, int endBit, bool isConstant, const QString& label, const QString& valueText) {
            addColumnDefinitionFromHint(startBit, endBit, isConstant, label, valueText);
        }
    );
    applyInitialWindowLayout();
    updateLoadedFileState();
}

MainWindow::~MainWindow() = default;

void MainWindow::buildMenus() {
    QMenu* fileMenu = menuBar()->addMenu(QStringLiteral("&File"));

    openFileAction_ = fileMenu->addAction(QStringLiteral("&Open File..."));
    connect(openFileAction_, &QAction::triggered, this, &MainWindow::openFile);

    reloadFileAction_ = fileMenu->addAction(QStringLiteral("&Reload File"));
    connect(reloadFileAction_, &QAction::triggered, this, &MainWindow::reloadFile);

    fileMenu->addSeparator();

    exportCsvAction_ = fileMenu->addAction(QStringLiteral("Export &CSV..."));
    connect(exportCsvAction_, &QAction::triggered, this, &MainWindow::exportCsv);

    fileMenu->addSeparator();
    fileMenu->addAction(QStringLiteral("E&xit"), this, &QWidget::close);

    QMenu* framingMenu = menuBar()->addMenu(QStringLiteral("&Framing"));

    applyFramingAction_ = framingMenu->addAction(QStringLiteral("&Apply Sync Framing"));
    connect(applyFramingAction_, &QAction::triggered, this, [this]() {
        if (framingController_ != nullptr) {
            framingController_->applySyncFraming();
        }
    });

    applyFixedFramingAction_ = framingMenu->addAction(QStringLiteral("Set &Fixed Framing"));
    connect(applyFixedFramingAction_, &QAction::triggered, this, [this]() {
        if (framingController_ != nullptr) {
            framingController_->applyFixedFraming();
        }
    });

    bitstreamSyncDiscoveryAction_ = framingMenu->addAction(QStringLiteral("Find &Framing..."));
    connect(bitstreamSyncDiscoveryAction_, &QAction::triggered, this, [this]() {
        if (framingController_ != nullptr) {
            framingController_->openBitstreamSyncDiscovery();
        }
    });

    patternSearchAction_ = framingMenu->addAction(QStringLiteral("Find &Pattern..."));
    connect(patternSearchAction_, &QAction::triggered, this, &MainWindow::openPatternSearchDialog);

    frameSelectionAction_ = framingMenu->addAction(QStringLiteral("Frame from &Selection"));
    connect(frameSelectionAction_, &QAction::triggered, this, [this]() {
        if (framingController_ != nullptr) {
            framingController_->frameSelection();
        }
    });

    framingMenu->addSeparator();

    clearFramingAction_ = framingMenu->addAction(QStringLiteral("&Clear Framing"));
    connect(clearFramingAction_, &QAction::triggered, this, [this]() {
        if (framingController_ != nullptr) {
            framingController_->clearFraming();
        }
    });

    QMenu* columnsMenu = menuBar()->addMenu(QStringLiteral("&Columns"));

    addColumnDefinitionAction_ = columnsMenu->addAction(QStringLiteral("&Add Column..."));
    connect(addColumnDefinitionAction_, &QAction::triggered, this, &MainWindow::addColumnDefinition);

    defineSelectionColumnAction_ = columnsMenu->addAction(QStringLiteral("Define Column from &Selection"));
    connect(defineSelectionColumnAction_, &QAction::triggered, this, &MainWindow::defineColumnFromSelection);

    columnsMenu->addSeparator();

    combineSelectionAction_ = columnsMenu->addAction(QStringLiteral("&Combine Selection"));
    connect(combineSelectionAction_, &QAction::triggered, this, &MainWindow::combineSelection);

    splitBinaryAction_ = columnsMenu->addAction(QStringLiteral("Split Selection as &Binary"));
    connect(splitBinaryAction_, &QAction::triggered, this, &MainWindow::splitSelectionAsBinary);

    splitNibblesAction_ = columnsMenu->addAction(QStringLiteral("Split Selection as &Nibbles"));
    connect(splitNibblesAction_, &QAction::triggered, this, &MainWindow::splitSelectionAsNibbles);

    clearSelectionSplitsAction_ = columnsMenu->addAction(QStringLiteral("Clear Selected Splits"));
    connect(clearSelectionSplitsAction_, &QAction::triggered, this, &MainWindow::clearSelectionSplits);

    columnsMenu->addSeparator();

    highlightConstantColumnsAction_ = columnsMenu->addAction(QStringLiteral("Highlight Constant Columns"));
    highlightConstantColumnsAction_->setCheckable(true);
    highlightConstantColumnsAction_->setChecked(byteTableModel_->constantColumnHighlightEnabled());
    connect(highlightConstantColumnsAction_, &QAction::toggled, this, [this](bool highlightEnabled) {
        if (byteTableModel_ == nullptr) {
            return;
        }
        byteTableModel_->setConstantColumnHighlightEnabled(highlightEnabled);
    });
}

void MainWindow::buildCentralWidget() {
    QWidget* centralWidget = new QWidget(this);
    QVBoxLayout* rootLayout = new QVBoxLayout(centralWidget);
    rootLayout->setContentsMargins(8, 8, 8, 8);
    rootLayout->setSpacing(8);

    QHBoxLayout* controlsLayout = new QHBoxLayout();
    controlsLayout->setSpacing(8);

    QPushButton* openFileButton = new QPushButton(QStringLiteral("Open File..."), centralWidget);
    connect(openFileButton, &QPushButton::clicked, this, &MainWindow::openFile);
    controlsLayout->addWidget(openFileButton);

    QPushButton* exportCsvButton = new QPushButton(QStringLiteral("Export CSV..."), centralWidget);
    connect(exportCsvButton, &QPushButton::clicked, this, &MainWindow::exportCsv);
    controlsLayout->addWidget(exportCsvButton);

    controlsLayout->addWidget(new QLabel(QStringLiteral("Mode:"), centralWidget));
    tableDisplayModeGroup_ = new QButtonGroup(centralWidget);
    QRadioButton* byteModeButton = new QRadioButton(QStringLiteral("Byte"), centralWidget);
    QRadioButton* bitModeButton = new QRadioButton(QStringLiteral("Bit"), centralWidget);
    byteModeButton->setChecked(true);
    tableDisplayModeGroup_->addButton(byteModeButton);
    tableDisplayModeGroup_->addButton(bitModeButton);
    controlsLayout->addWidget(byteModeButton);
    controlsLayout->addWidget(bitModeButton);
    connect(tableDisplayModeGroup_, QOverload<QAbstractButton*, bool>::of(&QButtonGroup::buttonToggled), this,
        [this, bitModeButton](QAbstractButton* toggledButton, bool checked) {
            if (!checked || toggledButton == nullptr) {
                return;
            }
            setBitModeEnabled(toggledButton == bitModeButton);
        });

    controlsLayout->addStretch();
    rootLayout->addLayout(controlsLayout);

    syncPatternLineEdit_ = new QLineEdit(centralWidget);
    syncPatternLineEdit_->setPlaceholderText(QStringLiteral("0x1ACF, 1011011, or 0b1011011"));
    syncPatternLineEdit_->setVisible(false);

    QHBoxLayout* framingLayout = new QHBoxLayout();
    framingLayout->setSpacing(8);
    QPushButton* discoveryButton = new QPushButton(QStringLiteral("Find Framing"), centralWidget);
    connect(discoveryButton, &QPushButton::clicked, this, [this]() {
        if (framingController_ != nullptr) {
            framingController_->openBitstreamSyncDiscovery();
        }
    });
    framingLayout->addWidget(discoveryButton);

    QPushButton* patternSearchButton = new QPushButton(QStringLiteral("Find Pattern"), centralWidget);
    connect(patternSearchButton, &QPushButton::clicked, this, &MainWindow::openPatternSearchDialog);
    framingLayout->addWidget(patternSearchButton);

    frameWidthLabel_ = new QLabel(QStringLiteral("Frame Width:"), centralWidget);
    framingLayout->addWidget(frameWidthLabel_);

    frameWidthSpinBox_ = new QSpinBox(centralWidget);
    frameWidthSpinBox_->setRange(1, 65536);
    frameWidthSpinBox_->setValue(dataSource_.bytesPerRow());
    frameWidthSpinBox_->setToolTip(QStringLiteral("Fixed frame width in bytes"));
    connect(frameWidthSpinBox_, qOverload<int>(&QSpinBox::valueChanged), this, [this](int) {
        if (!frameLayout_.isFramed()) {
            applyRawLayoutControls();
            return;
        }

        if (framingController_ != nullptr && framingController_->fixedFramingControlsEditable()) {
            framingController_->applyFixedFraming();
        }
    });
    framingLayout->addWidget(frameWidthSpinBox_);

    framingLayout->addWidget(new QLabel(QStringLiteral("Bit Offset:"), centralWidget));

    frameBitOffsetSpinBox_ = new QSpinBox(centralWidget);
    frameBitOffsetSpinBox_->setRange(0, 7);
    frameBitOffsetSpinBox_->setToolTip(QStringLiteral("Bit offset applied before fixed framing"));
    connect(frameBitOffsetSpinBox_, qOverload<int>(&QSpinBox::valueChanged), this, [this](int) {
        if (!frameLayout_.isFramed()) {
            applyRawLayoutControls();
            return;
        }

        if (framingController_ != nullptr && framingController_->fixedFramingControlsEditable()) {
            framingController_->applyFixedFraming();
        }
    });
    framingLayout->addWidget(frameBitOffsetSpinBox_);

    setFrameButton_ = new QPushButton(QStringLiteral("Set Frame"), centralWidget);
    connect(setFrameButton_, &QPushButton::clicked, this, [this]() {
        if (framingController_ != nullptr) {
            framingController_->applyFixedFraming();
        }
    });
    framingLayout->addWidget(setFrameButton_);

    framingLayout->addStretch();
    rootLayout->addLayout(framingLayout);

    QHBoxLayout* rowOrderLayout = new QHBoxLayout();
    rowOrderLayout->setSpacing(8);
    rowOrderLayout->addWidget(new QLabel(QStringLiteral("Sort Frame By:"), centralWidget));
    frameSortComboBox_ = new QComboBox(centralWidget);
    frameSortComboBox_->addItem(QStringLiteral("Chronological (Ascending)"));
    frameSortComboBox_->addItem(QStringLiteral("Chronological (Descending)"));
    frameSortComboBox_->addItem(QStringLiteral("Frame Size (Ascending)"));
    frameSortComboBox_->addItem(QStringLiteral("Frame Size (Descending)"));
    connect(frameSortComboBox_, qOverload<int>(&QComboBox::currentIndexChanged), this, [this](int index) {
        if (framingController_ != nullptr) {
            framingController_->setFrameSortOption(index);
        }
    });
    rowOrderLayout->addWidget(frameSortComboBox_);

    rowOrderLayout->addSpacing(16);
    rowOrderLayout->addWidget(new QLabel(QStringLiteral("Size:"), centralWidget));
    tableBitSizeSpinBox_ = new QSpinBox(centralWidget);
    tableBitSizeSpinBox_->setRange(2, 24);
    tableBitSizeSpinBox_->setValue(12);
    tableBitSizeSpinBox_->setEnabled(false);
    connect(tableBitSizeSpinBox_, qOverload<int>(&QSpinBox::valueChanged), this, [this](int) {
        if (byteTableModel_ != nullptr) {
            resizeTableColumns();
            if (byteTableView_ != nullptr) {
                byteTableView_->viewport()->update();
            }
        }
    });
    rowOrderLayout->addWidget(tableBitSizeSpinBox_);

    rowOrderLayout->addStretch();
    rootLayout->addLayout(rowOrderLayout);

    fileInfoLabel_ = new QLabel(centralWidget);
    fileInfoLabel_->setTextInteractionFlags(Qt::TextInteractionFlag::TextSelectableByMouse);
    rootLayout->addWidget(fileInfoLabel_);

    frameBrowserInfoLabel_ = new QLabel(centralWidget);
    frameBrowserInfoLabel_->setWordWrap(true);
    frameBrowserInfoLabel_->setTextInteractionFlags(Qt::TextInteractionFlag::TextSelectableByMouse);
    rootLayout->addWidget(frameBrowserInfoLabel_);

    byteTableView_ = new ByteTableView(centralWidget);
    byteTableView_->setModel(byteTableModel_);
    byteTableView_->setContextMenuPolicy(Qt::ContextMenuPolicy::CustomContextMenu);
    rootLayout->addWidget(byteTableView_, 1);
    connect(byteTableView_, &QWidget::customContextMenuRequested, this, &MainWindow::showTableContextMenu);
    byteTableView_->horizontalHeader()->setContextMenuPolicy(Qt::ContextMenuPolicy::CustomContextMenu);
    connect(
        byteTableView_->horizontalHeader(),
        &QWidget::customContextMenuRequested,
        this,
        [this](const QPoint& pos) {
            if (frameBrowserController_ != nullptr) {
                frameBrowserController_->showTableHeaderContextMenu(pos);
            }
        }
    );

    connect(
        byteTableView_->selectionModel(),
        &QItemSelectionModel::selectionChanged,
        this,
        [this](const QItemSelection&, const QItemSelection&) {
            if (inspectionController_ != nullptr) {
                inspectionController_->updateSelectionStatus();
            }
            refreshColumnDefinitionsPanel();
            if (inspectionController_ != nullptr) {
                inspectionController_->scheduleLiveBitViewerRefresh();
            }
        }
    );
    connect(
        byteTableView_->selectionModel(),
        &QItemSelectionModel::currentChanged,
        this,
        [this](const QModelIndex&, const QModelIndex&) {
            if (inspectionController_ != nullptr) {
                inspectionController_->updateSelectionStatus();
            }
            refreshColumnDefinitionsPanel();
        }
    );
    connect(byteTableView_, &QTableView::doubleClicked, this, [this](const QModelIndex& modelIndex) {
        if (byteTableModel_ == nullptr) {
            return;
        }

        const int visibleColumnIndex = byteTableModel_->visibleColumnIndexForModelIndex(modelIndex);
        if (visibleColumnIndex < 0) {
            return;
        }

        editVisibleColumns(editableVisibleColumnsForSeed(visibleColumnIndex));
    });
    connect(byteTableView_->horizontalHeader(), &QHeaderView::sectionDoubleClicked, this, [this](int logicalIndex) {
        if (byteTableModel_ == nullptr) {
            return;
        }

        const int visibleColumnIndex = logicalIndex - (byteTableModel_->hasFrameLengthColumn() ? 1 : 0);
        if (visibleColumnIndex < 0 || visibleColumnIndex >= byteTableModel_->visibleColumnCount()) {
            return;
        }

        editVisibleColumns(editableVisibleColumnsForSeed(visibleColumnIndex));
    });

    setCentralWidget(centralWidget);
    selectionInfoLabel_ = new QLabel(QStringLiteral("Selection: none"), this);
    statusBar()->addPermanentWidget(selectionInfoLabel_, 1);
    statusBar()->showMessage(QStringLiteral("No file loaded"));
}

void MainWindow::openPatternSearchDialog() {
    if (byteTableModel_ == nullptr) {
        return;
    }

    PatternSearchDialog dialog(
        &dataSource_,
        syncPatternLineEdit_ != nullptr ? syncPatternLineEdit_->text() : QString{},
        this
    );
    connect(
        &dialog,
        &PatternSearchDialog::highlightRangesChanged,
        this,
        [this](const QVector<QPair<qsizetype, qsizetype>>& absoluteBitRanges) {
            if (byteTableModel_ != nullptr) {
                byteTableModel_->setPatternHighlightRanges(absoluteBitRanges);
            }
        }
    );
    connect(&dialog, &PatternSearchDialog::focusMatchRequested, this, &MainWindow::focusPatternMatch);
    connect(&dialog, &PatternSearchDialog::frameByPatternRequested, this, [this](const QString& patternText) {
        if (framingController_ == nullptr) {
            return;
        }

        QString errorMessage;
        if (!framingController_->frameByPattern(patternText, &errorMessage)) {
            QMessageBox::warning(this, QStringLiteral("Frame by Pattern"), errorMessage);
        }
    });
    dialog.exec();
}

void MainWindow::buildColumnDefinitionsDock() {
    columnDefinitionsDock_ = new QDockWidget(QStringLiteral("Column Definitions"), this);
    columnDefinitionsDock_->setAllowedAreas(Qt::LeftDockWidgetArea | Qt::RightDockWidgetArea);
    QSplitter* leftDockSplitter = new QSplitter(Qt::Vertical, columnDefinitionsDock_);
    columnDefinitionsPanel_ = new ColumnDefinitionsPanel(leftDockSplitter);
    frameGroupingPanel_ = new FrameGroupingPanel(leftDockSplitter);
    frameGroupingPanel_->setTreeModel(frameGroupTreeModel_);
    if (frameBrowserController_ != nullptr) {
        frameBrowserController_->setPanel(frameGroupingPanel_);
    }
    QGroupBox* frameFieldHintsGroup = new QGroupBox(QStringLiteral("Framing Hints"), leftDockSplitter);
    QVBoxLayout* frameFieldHintsLayout = new QVBoxLayout(frameFieldHintsGroup);
    frameFieldHintsLayout->setContentsMargins(8, 8, 8, 8);
    frameFieldHintsLayout->setSpacing(6);
    frameFieldHintsPanel_ = new FrameFieldHintsPanel(
        FrameFieldHintsPanel::Mode::BrowseOnly,
        frameFieldHintsGroup
    );
    frameFieldHintsLayout->addWidget(frameFieldHintsPanel_);
    leftDockSplitter->addWidget(columnDefinitionsPanel_);
    leftDockSplitter->addWidget(frameGroupingPanel_);
    leftDockSplitter->addWidget(frameFieldHintsGroup);
    leftDockSplitter->setStretchFactor(0, 3);
    leftDockSplitter->setStretchFactor(1, 2);
    leftDockSplitter->setStretchFactor(2, 2);
    leftDockSplitter->setSizes({300, 180, 180});
    columnDefinitionsDock_->setWidget(leftDockSplitter);
    addDockWidget(Qt::LeftDockWidgetArea, columnDefinitionsDock_);

    connect(columnDefinitionsPanel_, &ColumnDefinitionsPanel::addRequested, this, &MainWindow::addColumnDefinition);
    connect(
        columnDefinitionsPanel_,
        &ColumnDefinitionsPanel::defineSelectionRequested,
        this,
        &MainWindow::defineColumnFromSelection
    );
    connect(
        columnDefinitionsPanel_,
        &ColumnDefinitionsPanel::editRequested,
        this,
        &MainWindow::editColumnDefinition
    );
    connect(columnDefinitionsPanel_, &ColumnDefinitionsPanel::editSplitRequested, this, [this](int startBit, int endBit) {
        const QSet<int> visibleColumns = visibleColumnsForAbsoluteBitRange(startBit, endBit);
        if (visibleColumns.isEmpty()) {
            return;
        }

        editVisibleColumns(visibleColumns);
    });
    connect(
        columnDefinitionsPanel_,
        &ColumnDefinitionsPanel::removeRequested,
        this,
        &MainWindow::removeColumnDefinition
    );
    connect(columnDefinitionsPanel_, &ColumnDefinitionsPanel::removeSplitRequested, this, [this](int byteIndex) {
        if (byteTableModel_ == nullptr || byteIndex < 0) {
            return;
        }
        if (!byteTableModel_->clearSplits(QSet<int>{byteIndex})) {
            return;
        }

        if (frameBrowserController_ != nullptr) {
            frameBrowserController_->saveCurrentSplitState();
        }
        resizeTableColumns();
        updateLoadedFileState();
        if (inspectionController_ != nullptr) {
            inspectionController_->updateSelectionStatus();
        }
        refreshColumnDefinitionsPanel();
        if (inspectionController_ != nullptr) {
            inspectionController_->refreshLiveBitViewer();
        }
        statusBar()->showMessage(QStringLiteral("Removed split"), 3000);
    });

    connect(frameGroupingPanel_, &FrameGroupingPanel::removeGroupingKeyRequested, this, [this](int keyIndex) {
        if (frameBrowserController_ != nullptr) {
            frameBrowserController_->removeGroupingKey(keyIndex);
        }
    });
    connect(frameGroupingPanel_, &FrameGroupingPanel::groupingOrderChanged, this, [this](const QVector<int>& reorderedIndexes) {
        if (frameBrowserController_ != nullptr) {
            frameBrowserController_->reorderGroupingKeys(reorderedIndexes);
        }
    });
    connect(frameGroupingPanel_, &FrameGroupingPanel::clearGroupingRequested, this, [this]() {
        if (frameBrowserController_ != nullptr) {
            frameBrowserController_->clearGrouping();
        }
    });
    connect(frameGroupingPanel_, &FrameGroupingPanel::clearFiltersRequested, this, [this]() {
        if (frameBrowserController_ != nullptr) {
            frameBrowserController_->clearFilters();
        }
    });
    connect(frameGroupingPanel_, &FrameGroupingPanel::clearScopeRequested, this, [this]() {
        if (frameBrowserController_ != nullptr) {
            frameBrowserController_->clearScope();
        }
    });
    connect(frameGroupingPanel_, &FrameGroupingPanel::scopeRequested, this, [this](const QModelIndex& treeIndex) {
        if (frameBrowserController_ != nullptr) {
            frameBrowserController_->applyScopeForTreeIndex(treeIndex);
        }
    });
    connect(frameGroupingPanel_, &FrameGroupingPanel::leafActivated, this, [this](const QModelIndex& treeIndex) {
        if (frameGroupTreeModel_ == nullptr || frameBrowserController_ == nullptr) {
            return;
        }
        frameBrowserController_->focusFrameByStartBit(frameGroupTreeModel_->leafFrameStartBit(treeIndex));
    });

    refreshColumnDefinitionsPanel();
    if (frameBrowserController_ != nullptr) {
        frameBrowserController_->refreshPanel();
    }
}

void MainWindow::buildLiveBitViewerDock() {
    liveBitViewerDock_ = new QDockWidget(QStringLiteral("Live Bit Viewer"), this);
    liveBitViewerDock_->setAllowedAreas(
        Qt::LeftDockWidgetArea | Qt::RightDockWidgetArea | Qt::BottomDockWidgetArea
    );

    QWidget* container = new QWidget(liveBitViewerDock_);
    QVBoxLayout* rootLayout = new QVBoxLayout(container);
    rootLayout->setContentsMargins(8, 8, 8, 8);
    rootLayout->setSpacing(8);

    QGroupBox* displayGroup = new QGroupBox(QStringLiteral("Display"), container);
    QVBoxLayout* displayLayout = new QVBoxLayout(displayGroup);

    QHBoxLayout* modeLayout = new QHBoxLayout();
    liveBitViewerModeGroup_ = new QButtonGroup(displayGroup);
    QRadioButton* squaresButton = new QRadioButton(QStringLiteral("Squares"), displayGroup);
    QRadioButton* circlesButton = new QRadioButton(QStringLiteral("Circles"), displayGroup);
    QRadioButton* digitsButton = new QRadioButton(QStringLiteral("Digits"), displayGroup);
    squaresButton->setChecked(true);
    liveBitViewerModeGroup_->addButton(squaresButton);
    liveBitViewerModeGroup_->addButton(circlesButton);
    liveBitViewerModeGroup_->addButton(digitsButton);
    modeLayout->addWidget(squaresButton);
    modeLayout->addWidget(circlesButton);
    modeLayout->addWidget(digitsButton);
    modeLayout->addStretch();
    displayLayout->addLayout(modeLayout);

    QHBoxLayout* sizeLayout = new QHBoxLayout();
    sizeLayout->addWidget(new QLabel(QStringLiteral("Size:"), displayGroup));
    liveBitViewerSizeSpinBox_ = new QSpinBox(displayGroup);
    liveBitViewerSizeSpinBox_->setRange(4, 24);
    liveBitViewerSizeSpinBox_->setValue(10);
    sizeLayout->addWidget(liveBitViewerSizeSpinBox_);
    sizeLayout->addStretch();
    displayLayout->addLayout(sizeLayout);

    rootLayout->addWidget(displayGroup);

    QScrollArea* liveViewerScrollArea = new QScrollArea(container);
    liveViewerScrollArea->setWidgetResizable(false);
    liveViewerScrollArea->setHorizontalScrollBarPolicy(Qt::ScrollBarAsNeeded);
    liveViewerScrollArea->setVerticalScrollBarPolicy(Qt::ScrollBarAsNeeded);
    liveBitViewerWidget_ = new LiveBitViewerWidget(liveViewerScrollArea);
    liveViewerScrollArea->setWidget(liveBitViewerWidget_);
    rootLayout->addWidget(liveViewerScrollArea, 2);

    QGroupBox* distributionGroup = new QGroupBox(QStringLiteral("Distribution"), container);
    QVBoxLayout* distributionLayout = new QVBoxLayout(distributionGroup);
    distributionLayout->setContentsMargins(8, 8, 8, 8);
    distributionLayout->setSpacing(6);
    fieldInspectorPanel_ = new FieldInspectorPanel(distributionGroup);
    distributionLayout->addWidget(fieldInspectorPanel_);
    rootLayout->addWidget(distributionGroup, 3);

    QGroupBox* currentValueGroup = new QGroupBox(QStringLiteral("Current Value"), container);
    QVBoxLayout* currentValueLayout = new QVBoxLayout(currentValueGroup);
    currentValueLayout->setContentsMargins(8, 8, 8, 8);
    currentValueLayout->setSpacing(6);
    fieldCurrentValuePanel_ = new FieldCurrentValuePanel(currentValueGroup);
    currentValueLayout->addWidget(fieldCurrentValuePanel_);
    rootLayout->addWidget(currentValueGroup);

    liveBitViewerDock_->setWidget(container);
    addDockWidget(Qt::RightDockWidgetArea, liveBitViewerDock_);
    liveBitViewerDock_->setMinimumWidth(detail::kMinimumLiveBitViewerDockWidth);
    liveBitViewerDock_->setMinimumHeight(360);

    connect(
        liveBitViewerModeGroup_,
        QOverload<QAbstractButton*, bool>::of(&QButtonGroup::buttonToggled),
        this,
        [this](QAbstractButton*, bool) {
            syncTableBitDisplayMode();
            if (inspectionController_ != nullptr) {
                inspectionController_->scheduleLiveBitViewerRefresh();
            }
        }
    );
    connect(liveBitViewerSizeSpinBox_, &QSpinBox::valueChanged, this, [this](int) {
        if (inspectionController_ != nullptr) {
            inspectionController_->scheduleLiveBitViewerRefresh();
        }
    });

    syncTableBitDisplayMode();
}

void MainWindow::applyInitialWindowLayout() {
    QRect availableGeometry(0, 0, detail::kPreferredMainWindowWidth, detail::kPreferredMainWindowHeight);
    if (const QScreen* primaryScreen = QGuiApplication::primaryScreen(); primaryScreen != nullptr) {
        availableGeometry = primaryScreen->availableGeometry();
    }

    const int targetWidth = qMin(
        availableGeometry.width(),
        qMin(
            detail::kPreferredMainWindowWidth,
            qMax(detail::kMinimumComfortableMainWindowWidth, (availableGeometry.width() * 9) / 10)
        )
    );
    const int targetHeight = qMin(
        availableGeometry.height(),
        qMin(
            detail::kPreferredMainWindowHeight,
            qMax(detail::kMinimumComfortableMainWindowHeight, (availableGeometry.height() * 17) / 20)
        )
    );

    resize(targetWidth, targetHeight);

    if (columnDefinitionsDock_ != nullptr && liveBitViewerDock_ != nullptr) {
        resizeDocks(
            {columnDefinitionsDock_, liveBitViewerDock_},
            {detail::kInitialColumnDefinitionsDockWidth, detail::kInitialLiveBitViewerDockWidth},
            Qt::Horizontal
        );
    }
}

void MainWindow::syncFrameWidthControls() {
    if (frameWidthSpinBox_ == nullptr || frameBitOffsetSpinBox_ == nullptr) {
        return;
    }

    const bool bitModeEnabled = byteTableModel_ != nullptr && byteTableModel_->isBitDisplayMode();
    const bool fixedFramingActive =
        frameLayout_.isFramed() && framingController_ != nullptr && framingController_->fixedFramingControlsEditable();
    const qsizetype widthBits = fixedFramingActive ? frameLayout_.frameMaxLengthBits() : frameLayout_.rawRowWidthBits();
    const int displayWidthValue = bitModeEnabled
        ? static_cast<int>(qBound<qsizetype>(
            static_cast<qsizetype>(1),
            widthBits,
            static_cast<qsizetype>(std::numeric_limits<int>::max())
        ))
        : qMax(1, static_cast<int>((widthBits + 7) / 8));
    const int maximumWidthValue = bitModeEnabled
        ? (
            dataSource_.hasData()
                ? static_cast<int>(qBound<qsizetype>(
                    static_cast<qsizetype>(1),
                    dataSource_.bitCount(),
                    static_cast<qsizetype>(std::numeric_limits<int>::max())
                ))
                : 65536
        )
        : 65536;

    if (frameWidthLabel_ != nullptr) {
        frameWidthLabel_->setText(bitModeEnabled ? QStringLiteral("Frame Width (Bits):") : QStringLiteral("Frame Width:"));
    }

    {
        const QSignalBlocker widthBlocker(frameWidthSpinBox_);
        frameWidthSpinBox_->setRange(1, qMax(1, maximumWidthValue));
        frameWidthSpinBox_->setToolTip(
            bitModeEnabled
                ? QStringLiteral("Raw/fixed frame width in bits")
                : QStringLiteral("Raw/fixed frame width in bytes")
        );
        frameWidthSpinBox_->setValue(displayWidthValue);
    }

    if (!frameLayout_.isFramed()) {
        const QSignalBlocker offsetBlocker(frameBitOffsetSpinBox_);
        frameBitOffsetSpinBox_->setValue(static_cast<int>(qBound<qsizetype>(
            static_cast<qsizetype>(0),
            frameLayout_.rawStartBitOffset(),
            static_cast<qsizetype>(7)
        )));
    }
}

void MainWindow::applyRawLayoutControls() {
    if (byteTableModel_ == nullptr || frameWidthSpinBox_ == nullptr || frameBitOffsetSpinBox_ == nullptr) {
        return;
    }

    const bool bitModeEnabled = byteTableModel_->isBitDisplayMode();
    const int frameWidthValue = qMax(1, frameWidthSpinBox_->value());
    const int bitOffset = qBound(0, frameBitOffsetSpinBox_->value(), 7);
    const qsizetype rowWidthBits = bitModeEnabled
        ? static_cast<qsizetype>(frameWidthValue)
        : static_cast<qsizetype>(frameWidthValue) * 8;

    if (!bitModeEnabled) {
        dataSource_.setBytesPerRow(frameWidthValue);
    }
    frameLayout_.setRawLayout(rowWidthBits, bitOffset);
    byteTableModel_->reload();
    resizeTableColumns();
    updateLoadedFileState();
    if (inspectionController_ != nullptr) {
        inspectionController_->updateSelectionStatus();
        inspectionController_->refreshLiveBitViewer();
    }
}

void MainWindow::updateLoadedFileState() {
    fileInfoLabel_->setText(fileSummaryText());
    if (frameBrowserInfoLabel_ != nullptr) {
        frameBrowserInfoLabel_->setText(
            frameBrowserController_ != nullptr
                ? frameBrowserController_->summaryText()
                : QStringLiteral("Frame grouping is available after framing is active.")
        );
    }
    updateWindowTitle();
    if (framingController_ != nullptr) {
        framingController_->syncControlsFromState();
    }
    syncFrameWidthControls();
    refreshColumnDefinitionsPanel();
    if (frameBrowserController_ != nullptr) {
        frameBrowserController_->refreshPanel();
    }
    if (inspectionController_ != nullptr) {
        inspectionController_->scheduleFrameFieldHintsRefresh();
    }

    const bool hasData = dataSource_.hasData();
    reloadFileAction_->setEnabled(hasData);
    exportCsvAction_->setEnabled(hasData);
    applyFramingAction_->setEnabled(hasData);
    const bool fixedFramingControlsEditable =
        framingController_ != nullptr && framingController_->fixedFramingControlsEditable();
    if (applyFixedFramingAction_ != nullptr) {
        applyFixedFramingAction_->setEnabled(fixedFramingControlsEditable);
    }
    if (bitstreamSyncDiscoveryAction_ != nullptr) {
        bitstreamSyncDiscoveryAction_->setEnabled(hasData);
    }
    if (patternSearchAction_ != nullptr) {
        patternSearchAction_->setEnabled(hasData);
    }
    frameSelectionAction_->setEnabled(hasColumnSelection());
    clearFramingAction_->setEnabled(hasData && frameLayout_.isFramed());
    addColumnDefinitionAction_->setEnabled(hasData);
    defineSelectionColumnAction_->setEnabled(hasColumnSelection());
    combineSelectionAction_->setEnabled(hasColumnSelection());
    if (tableBitSizeSpinBox_ != nullptr) {
        tableBitSizeSpinBox_->setEnabled(hasData);
    }
    if (frameWidthSpinBox_ != nullptr) {
        frameWidthSpinBox_->setEnabled(fixedFramingControlsEditable);
    }
    if (frameBitOffsetSpinBox_ != nullptr) {
        frameBitOffsetSpinBox_->setEnabled(fixedFramingControlsEditable);
    }
    if (setFrameButton_ != nullptr) {
        setFrameButton_->setEnabled(fixedFramingControlsEditable);
    }
    if (frameSortComboBox_ != nullptr) {
        frameSortComboBox_->setEnabled(hasData && frameLayout_.isFramed());
    }
    if (frameGroupingPanel_ != nullptr) {
        frameGroupingPanel_->setFramingActive(hasData && frameLayout_.isFramed());
    }

    if (hasData) {
        if (frameLayout_.isFramed()) {
            statusBar()->showMessage(
                QStringLiteral("Framed view: %1 rows | longest frame %2 bytes")
                    .arg(frameLayout_.rowCount(dataSource_))
                    .arg(frameLayout_.frameMaxLengthBytes())
            );
        } else {
            statusBar()->showMessage(
                QStringLiteral("Loaded %1 bytes across %2 raw rows")
                    .arg(dataSource_.byteCount())
                    .arg(dataSource_.rowCount())
            );
        }
    } else {
        statusBar()->showMessage(QStringLiteral("No file loaded"));
    }
}

void MainWindow::updateWindowTitle() {
    const QString modeSuffix = byteTableModel_ != nullptr && byteTableModel_->isBitDisplayMode()
        ? QStringLiteral(" Bit Mode")
        : QStringLiteral(" Byte Mode");
    if (!dataSource_.hasData()) {
        setWindowTitle(QStringLiteral("Bitabyte C++%1").arg(modeSuffix));
        return;
    }

    const QFileInfo fileInfo(dataSource_.sourceFilePath());
    if (frameLayout_.isFramed()) {
        setWindowTitle(QStringLiteral("Bitabyte C++%1 - %2 [Framed]").arg(modeSuffix, fileInfo.fileName()));
        return;
    }

    setWindowTitle(QStringLiteral("Bitabyte C++%1 - %2").arg(modeSuffix, fileInfo.fileName()));
}

void MainWindow::resizeEvent(QResizeEvent* resizeEvent) {
    QMainWindow::resizeEvent(resizeEvent);

    if (byteTableModel_ != nullptr && byteTableModel_->isBitDisplayMode()) {
        resizeTableColumns();
    }
}

void MainWindow::resizeTableColumns() {
    if (byteTableView_ == nullptr || byteTableModel_ == nullptr) {
        return;
    }

    const int requestedSize = tableBitSizeSpinBox_ != nullptr ? tableBitSizeSpinBox_->value() : 12;
    const int byteFontPixelSize = qMax(10, requestedSize + 2);
    byteTableModel_->setContentFontPixelSize(byteFontPixelSize);

    const int modelColumnCount = byteTableModel_->columnCount();
    const bool hasFrameLengthColumn = byteTableModel_->hasFrameLengthColumn();
    const QFontMetrics headerMetrics(byteTableView_->horizontalHeader()->font());
    const QFontMetrics contentMetrics(byteTableModel_->contentFont());
    auto minimumHeaderWidth = [&](const QString& labelText) {
        return labelText.isEmpty() ? 0 : headerMetrics.horizontalAdvance(labelText) + 16;
    };
    auto contentWidthForCharacters = [&](int characterCount) {
        return contentMetrics.horizontalAdvance(QString(qMax(1, characterCount), QLatin1Char('0'))) + 18;
    };

    if (byteTableModel_->isBitDisplayMode()) {
        const bool digitMode =
            byteTableModel_->bitCellDisplayMode() == models::ByteTableModel::BitCellDisplayMode::Digits;
        const int bitColumnWidth = digitMode ? qMax(6, requestedSize + 2) : requestedSize;
        const int bitRowHeight = digitMode ? qMax(8, requestedSize + 4) : qMax(4, requestedSize + 1);

        if (byteTableView_->verticalHeader() != nullptr) {
            byteTableView_->verticalHeader()->setDefaultSectionSize(bitRowHeight);
        }
        for (int modelColumn = 0; modelColumn < modelColumnCount; ++modelColumn) {
            byteTableView_->setColumnWidth(
                modelColumn,
                hasFrameLengthColumn && modelColumn == 0 ? 44 : bitColumnWidth
            );
        }
        return;
    }

    if (byteTableView_->verticalHeader() != nullptr) {
        byteTableView_->verticalHeader()->setDefaultSectionSize(qMax(22, contentMetrics.height() + 8));
    }

    for (int modelColumn = 0; modelColumn < modelColumnCount; ++modelColumn) {
        int minimumWidth = 36;
        int estimatedContentWidth = minimumWidth;
        if (hasFrameLengthColumn && modelColumn == 0) {
            minimumWidth = qMax(52, minimumHeaderWidth(QStringLiteral("Len")));
            estimatedContentWidth = minimumWidth;
        } else {
            const int visibleColumnIndex = hasFrameLengthColumn ? modelColumn - 1 : modelColumn;
            const features::columns::VisibleByteColumn visibleColumn =
                byteTableModel_->visibleByteColumn(visibleColumnIndex);
            QString displayFormat = visibleColumn.splitDisplayFormat.trimmed().toLower();
            int displayBitWidth = qMax(1, visibleColumn.bitWidth());
            int displayByteCount = qMax(1, (displayBitWidth + 7) / 8);
            if (visibleColumn.definitionIndex >= 0) {
                minimumWidth = 96;
                if (visibleColumn.definitionIndex < columnDefinitions_.size()) {
                    const features::columns::ByteColumnDefinition& columnDefinition =
                        columnDefinitions_.at(visibleColumn.definitionIndex);
                    displayFormat = columnDefinition.displayFormat.trimmed().toLower();
                    displayByteCount = qMax(1, columnDefinition.byteCount());
                    displayBitWidth = columnDefinition.unit == QStringLiteral("bit")
                        ? qMax(1, columnDefinition.totalBits)
                        : displayByteCount * 8;
                }
            } else if (visibleColumn.bitWidth() == 1) {
                minimumWidth = 24;
            } else if (visibleColumn.bitWidth() <= 4) {
                minimumWidth = 34;
            }
            minimumWidth = qMax(
                minimumWidth,
                minimumHeaderWidth(byteTableModel_->bottomHeaderLabelForVisibleColumn(visibleColumnIndex))
            );

            if (displayFormat == QStringLiteral("dec")) {
                displayFormat = QStringLiteral("decimal");
            } else if (displayFormat.isEmpty()) {
                displayFormat = QStringLiteral("hex");
            }

            if (displayFormat == QStringLiteral("binary")) {
                estimatedContentWidth = contentWidthForCharacters(displayBitWidth);
            } else if (displayFormat == QStringLiteral("ascii")) {
                estimatedContentWidth = contentWidthForCharacters(displayByteCount);
            } else if (displayFormat == QStringLiteral("decimal")) {
                const int decimalDigitCount = qMax(
                    1,
                    static_cast<int>(std::ceil(static_cast<double>(displayBitWidth) * 0.30103))
                );
                estimatedContentWidth = contentWidthForCharacters(decimalDigitCount);
            } else {
                const int hexDigitCount = qMax(1, (displayBitWidth + 3) / 4);
                const int trailingBitCharacterCount = displayBitWidth % 4 == 0 ? 0 : 1 + (displayBitWidth % 4);
                estimatedContentWidth = contentWidthForCharacters(2 + hexDigitCount + trailingBitCharacterCount);
            }
        }

        byteTableView_->setColumnWidth(modelColumn, qMax(minimumWidth, estimatedContentWidth));
    }
}

void MainWindow::setBitModeEnabled(bool enabled) {
    if (byteTableModel_ == nullptr) {
        return;
    }

    const auto targetMode = enabled
        ? models::ByteTableModel::DisplayMode::Bit
        : models::ByteTableModel::DisplayMode::Byte;
    if (byteTableModel_->displayMode() == targetMode) {
        return;
    }

    if (!enabled && !frameLayout_.isFramed()) {
        const int byteWidth = qMax(1, static_cast<int>((frameLayout_.rawRowWidthBits() + 7) / 8));
        dataSource_.setBytesPerRow(byteWidth);
        frameLayout_.setRawLayout(static_cast<qsizetype>(byteWidth) * 8, frameLayout_.rawStartBitOffset());
    }

    const QSet<int> selectedColumns = selectedVisibleColumns();
    int selectionStartBit = std::numeric_limits<int>::max();
    int selectionEndBit = std::numeric_limits<int>::min();
    for (int visibleColumnIndex : selectedColumns) {
        const features::columns::VisibleByteColumn visibleColumn = byteTableModel_->visibleByteColumn(visibleColumnIndex);
        selectionStartBit = qMin(selectionStartBit, visibleColumn.absoluteStartBit);
        selectionEndBit = qMax(selectionEndBit, visibleColumn.absoluteEndBit);
    }

    const QModelIndex currentIndex = byteTableView_ != nullptr ? byteTableView_->currentIndex() : QModelIndex{};
    const int targetRow = currentIndex.isValid() ? currentIndex.row() : 0;
    const qsizetype currentStartBit = byteTableModel_->displayStartBitForIndex(currentIndex);

    byteTableModel_->setDisplayMode(targetMode);
    if (byteTableView_ != nullptr) {
        byteTableView_->setShowGrid(true);
        if (byteTableView_->horizontalHeader() != nullptr) {
            byteTableView_->horizontalHeader()->setVisible(!enabled);
        }
        if (byteTableView_->verticalHeader() != nullptr) {
            byteTableView_->verticalHeader()->setVisible(!enabled);
        }
    }
    syncFrameWidthControls();
    resizeTableColumns();

    if (selectionStartBit <= selectionEndBit) {
        selectVisibleColumnsAtRow(visibleColumnsForAbsoluteBitRange(selectionStartBit, selectionEndBit), targetRow);
    } else if (currentStartBit >= 0) {
        const int relativeStartBit = static_cast<int>(currentStartBit - frameLayout_.rowStartBit(dataSource_, targetRow));
        selectVisibleColumnsAtRow(
            visibleColumnsForAbsoluteBitRange(relativeStartBit, relativeStartBit),
            targetRow
        );
    }

    updateLoadedFileState();
    if (inspectionController_ != nullptr) {
        inspectionController_->updateSelectionStatus();
        inspectionController_->scheduleLiveBitViewerRefresh();
        inspectionController_->scheduleFrameFieldHintsRefresh();
    }
    if (byteTableView_ != nullptr) {
        byteTableView_->viewport()->update();
        byteTableView_->horizontalHeader()->viewport()->update();
    }
}

void MainWindow::focusPatternMatch(qsizetype startBit, qsizetype bitCount) {
    if (byteTableModel_ == nullptr || byteTableView_ == nullptr || !dataSource_.hasData() || bitCount <= 0) {
        return;
    }

    const qsizetype matchEndBit = startBit + bitCount - 1;
    const int totalRows = byteTableModel_->rowCount();
    for (int row = 0; row < totalRows; ++row) {
        const qsizetype rowStartBit = frameLayout_.rowStartBit(dataSource_, row);
        const qsizetype rowLengthBits = frameLayout_.rowLengthBits(dataSource_, row);
        if (rowLengthBits <= 0) {
            continue;
        }

        const qsizetype rowEndBit = rowStartBit + rowLengthBits - 1;
        if (matchEndBit < rowStartBit || startBit > rowEndBit) {
            continue;
        }

        const int relativeStartBit = static_cast<int>(qMax(startBit, rowStartBit) - rowStartBit);
        const int relativeEndBit = static_cast<int>(qMin(matchEndBit, rowEndBit) - rowStartBit);
        selectVisibleColumnsAtRow(visibleColumnsForAbsoluteBitRange(relativeStartBit, relativeEndBit), row);
        if (byteTableView_->currentIndex().isValid()) {
            byteTableView_->scrollTo(byteTableView_->currentIndex(), QAbstractItemView::PositionAtCenter);
        }
        byteTableView_->setFocus();
        return;
    }
}

void MainWindow::syncTableBitDisplayMode() {
    if (byteTableModel_ == nullptr || liveBitViewerModeGroup_ == nullptr || liveBitViewerModeGroup_->checkedButton() == nullptr) {
        return;
    }

    const QString modeText = liveBitViewerModeGroup_->checkedButton()->text().trimmed().toLower();
    auto bitCellDisplayMode = models::ByteTableModel::BitCellDisplayMode::Squares;
    if (modeText == QStringLiteral("circles")) {
        bitCellDisplayMode = models::ByteTableModel::BitCellDisplayMode::Circles;
    } else if (modeText == QStringLiteral("digits")) {
        bitCellDisplayMode = models::ByteTableModel::BitCellDisplayMode::Digits;
    }

    byteTableModel_->setBitCellDisplayMode(bitCellDisplayMode);
    if (byteTableView_ != nullptr) {
        byteTableView_->setShowGrid(true);
    }
    if (byteTableModel_->isBitDisplayMode()) {
        resizeTableColumns();
        if (byteTableView_ != nullptr) {
            byteTableView_->viewport()->update();
        }
    }
}

}  // namespace bitabyte::ui
