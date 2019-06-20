string(TOUPPER ${PROJECT_NAME} PROJECT_UPPER_NAME)
add_library(${PROJECT_NAME}::${PROJECT_NAME} ALIAS ${PROJECT_NAME})
target_include_directories(${PROJECT_NAME} PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        $<BUILD_INTERFACE:${CMAKE_Fortran_MODULE_DIRECTORY}>
        $<INSTALL_INTERFACE:include>
        )
target_compile_definitions(${PROJECT_NAME} PRIVATE NOMAIN)
if (MOLPRO)
    target_include_directories(${PROJECT_NAME} PRIVATE "${MOLPRO}/src")
endif ()

install(DIRECTORY ${CMAKE_Fortran_MODULE_DIRECTORY}/ DESTINATION include)

install(TARGETS ${PROJECT_NAME} EXPORT ${PROJECT_NAME}Targets LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
        RUNTIME DESTINATION bin
        INCLUDES DESTINATION include
        )
install(EXPORT ${PROJECT_NAME}Targets
        FILE ${PROJECT_NAME}Targets.cmake
        NAMESPACE ${PROJECT_NAME}::
        DESTINATION lib/cmake/${PROJECT_NAME}
        )

include(CMakePackageConfigHelpers)
file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake" "
#include(CMakeFindDependencyMacro)
#find_dependency(Bar 2.0)
include(\"\${CMAKE_CURRENT_LIST_DIR}/${PROJECT_NAME}Targets.cmake\")
")
write_basic_package_version_file("${PROJECT_NAME}ConfigVersion.cmake"
        VERSION ${CMAKE_PROJECT_VERSION}
        COMPATIBILITY SameMajorVersion
        )
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake" "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake"
        DESTINATION lib/cmake/${PROJECT_NAME}
        )

set(CONFIG_CPPFLAGS "-I${CMAKE_INSTALL_PREFIX}/include")
get_target_property(FLAGS ${PROJECT_NAME} INTERFACE_COMPILE_DEFINITIONS)
if (FLAGS)
    foreach (flag ${FLAGS})
        set(CONFIG_CPPFLAGS "${CONFIG_CPPFLAGS} -D${flag}")
    endforeach ()
endif ()
#set(CONFIG_FCFLAGS "${CMAKE_Fortran_MODDIR_FLAG}${CMAKE_INSTALL_PREFIX}/include")
set(CONFIG_FCFLAGS "-I${CMAKE_INSTALL_PREFIX}/include") #TODO should not be hard-wired -I
set(CONFIG_LDFLAGS "-L${CMAKE_INSTALL_PREFIX}/lib")
set(CONFIG_LIBS "-l${PROJECT_NAME}")
configure_file(config.in ${PROJECT_NAME}-config @ONLY)
install(PROGRAMS ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}-config DESTINATION bin)
