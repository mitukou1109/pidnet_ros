if(NOT EXISTS "${VENV_DIR}")
  execute_process(
    COMMAND "${UV_EXECUTABLE}" venv --system-site-packages "${VENV_DIR}"
    RESULT_VARIABLE result
  )

  if(NOT result EQUAL 0)
    message(FATAL_ERROR "Failed to create venv: ${result}")
  endif()
endif()
