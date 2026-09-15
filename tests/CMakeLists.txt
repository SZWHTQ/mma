################################################################################

# Main test
add_executable(test_mma main.cpp)
target_link_libraries(test_mma PRIVATE mma::mma)

# Triangle test
add_executable(test_triangle triangle.cpp)
target_link_libraries(test_triangle PRIVATE mma::mma)

# Pairwise test
add_executable(test_pairwise pairwise.cpp)
target_link_libraries(test_pairwise PRIVATE mma::mma)

# GCMMA test
add_executable(test_gcmma toy.cpp)
target_link_libraries(test_gcmma PRIVATE mma::mma mma::gcmma)

# Reference-faithful GCMMA status, scaling, finite-value, and rollback checks.
add_executable(test_gcmma_reference_repair gcmma_reference_repair.cpp)
target_link_libraries(test_gcmma_reference_repair PRIVATE mma::mma mma::gcmma)

if(BUILD_TESTING)
    add_test(NAME mma.gcmma_reference_repair
             COMMAND test_gcmma_reference_repair)
endif()
