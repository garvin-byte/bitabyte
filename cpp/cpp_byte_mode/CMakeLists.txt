cmake_minimum_required(VERSION 3.21)

project(bitabyte_cpp_byte_mode LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTORCC ON)

find_package(Qt6 REQUIRED COMPONENTS Widgets Concurrent)

set(BITABYTE_CPP_BYTE_MODE_FILES
    src/app/main.cpp
    src/core/byte_format_utils.h
    src/core/byte_format_utils.cpp
    src/data/byte_data_source.h
    src/data/byte_data_source.cpp
    src/features/bitstream_sync_discovery/bitstream_sync_discovery_types.h
    src/features/bitstream_sync_discovery/bitstream_sync_discovery_internal.h
    src/features/bitstream_sync_discovery/bitstream_sync_discovery_engine.h
    src/features/bitstream_sync_discovery/bitstream_sync_discovery_engine.cpp
    src/features/bitstream_sync_discovery/bitstream_sync_discovery_ranking.cpp
    src/features/bitstream_sync_discovery/bitstream_sync_discovery_formatter.h
    src/features/bitstream_sync_discovery/bitstream_sync_discovery_formatter.cpp
    src/features/bitstream_sync_discovery/bitstream_sync_discovery_worker.h
    src/features/bitstream_sync_discovery/bitstream_sync_discovery_worker.cpp
    src/features/columns/byte_column_definition.h
    src/features/columns/column_color_palette.h
    src/features/columns/column_color_palette.cpp
    src/features/columns/visible_byte_column.h
    src/features/export/byte_table_csv_exporter.h
    src/features/export/byte_table_csv_exporter.cpp
    src/features/frame_browser/frame_grouping.h
    src/features/frame_browser/frame_grouping.cpp
    src/features/framing/frame_layout.h
    src/features/framing/frame_layout.cpp
    src/features/frame_sync/frame_sync_search.h
    src/features/frame_sync/frame_sync_search.cpp
    src/features/inspector/field_inspector_analysis.h
    src/features/inspector/field_inspector_analysis.cpp
    src/models/byte_table_model.h
    src/models/byte_table_model.cpp
    src/models/frame_group_tree_model.h
    src/models/frame_group_tree_model.cpp
    src/ui/byte_table_view.h
    src/ui/byte_table_view.cpp
    src/ui/column_definition_dialog.h
    src/ui/column_definition_dialog.cpp
    src/ui/column_definitions_panel.h
    src/ui/column_definitions_panel.cpp
    src/ui/field_inspector_panel.h
    src/ui/field_inspector_panel.cpp
    src/ui/frame_browser_controller.h
    src/ui/frame_browser_controller.cpp
    src/ui/frame_grouping_panel.h
    src/ui/frame_grouping_panel.cpp
    src/ui/framing_controller.h
    src/ui/framing_controller.cpp
    src/ui/inspection_controller.h
    src/ui/inspection_controller.cpp
    src/ui/bitstream_sync_discovery_dialog.h
    src/ui/bitstream_sync_discovery_dialog.cpp
    src/ui/live_bit_viewer_widget.h
    src/ui/live_bit_viewer_widget.cpp
    src/ui/main_window.h
    src/ui/main_window_internal.h
    src/ui/main_window.cpp
    src/ui/main_window_file_ops.cpp
    src/ui/main_window_columns.cpp
)

qt_add_executable(bitabyte_cpp_byte_mode
    WIN32
    ${BITABYTE_CPP_BYTE_MODE_FILES}
)

target_include_directories(bitabyte_cpp_byte_mode PRIVATE src)

target_link_libraries(bitabyte_cpp_byte_mode PRIVATE Qt6::Widgets Qt6::Concurrent)

if(WIN32)
    get_target_property(BITABYTE_QMAKE_EXECUTABLE Qt6::qmake IMPORTED_LOCATION)
    get_filename_component(BITABYTE_QT_BIN_DIR "${BITABYTE_QMAKE_EXECUTABLE}" DIRECTORY)
    find_program(BITABYTE_WINDEPLOYQT_EXECUTABLE windeployqt HINTS "${BITABYTE_QT_BIN_DIR}")

    if(BITABYTE_WINDEPLOYQT_EXECUTABLE)
        string(REPLACE ";" "\\;" BITABYTE_WINDEPLOYQT_PATH "${BITABYTE_QT_BIN_DIR};$ENV{PATH}")
        add_custom_command(
            TARGET bitabyte_cpp_byte_mode
            POST_BUILD
            COMMAND "${CMAKE_COMMAND}" -E env
                "PATH=${BITABYTE_WINDEPLOYQT_PATH}"
                "${BITABYTE_WINDEPLOYQT_EXECUTABLE}"
                --compiler-runtime
                --no-translations
                "$<TARGET_FILE:bitabyte_cpp_byte_mode>"
            COMMENT "Deploying Qt runtime dependencies"
            VERBATIM
        )
    else()
        message(WARNING "windeployqt was not found; the built executable may be missing Qt runtime DLLs.")
    endif()
endif()

source_group(
    TREE "${CMAKE_CURRENT_SOURCE_DIR}/src"
    PREFIX "src"
    FILES ${BITABYTE_CPP_BYTE_MODE_FILES}
)
