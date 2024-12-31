OPTIX_add_sample_executable( optix_icosahedron target_name
  # ${SAMPLES_CUDA_DIR}/helpers.h
  programs.cu
  main.cpp
  utils.h

  OPTIONS -rdc true
  )

target_link_libraries( ${target_name}
  ${CUDA_LIBRARIES}
  )
