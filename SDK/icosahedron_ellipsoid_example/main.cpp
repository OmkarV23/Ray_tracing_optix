#include <optix.h>
#include <optix_function_table_definition.h>
#include <optix_stack_size.h>
#include <optix_stubs.h>

#include <cuda_runtime.h>

#include <sampleConfig.h>

#include <sutil/CUDAOutputBuffer.h>
#include <sutil/Exception.h>
#include <sutil/sutil.h>

#include "utils.h"

#include <array>
#include <iomanip>
#include <iostream>
#include <string>

#include <sutil/Camera.h>
#include <sutil/Trackball.h>


template <typename T>
struct SbtRecord
{
    __align__( OPTIX_SBT_RECORD_ALIGNMENT ) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    T data;
};



template <typename TriangleMesh>
struct HitgroupRecord
{
__align__( OPTIX_SBT_RECORD_ALIGNMENT ) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
TriangleMesh data;
};


typedef SbtRecord<RayGenData> RayGenSbtRecord;
typedef SbtRecord<MissData>        MissSbtRecord;
typedef HitgroupRecord<HitGroupData>    HitGroupSbtRecord;

void configureCamera( sutil::Camera& cam, const uint32_t width, const uint32_t height )
{
    cam.setEye( {0.0f, 0.0f, 2.0f} );
    cam.setLookat( {0.0f, 0.0f, 0.0f} );
    cam.setUp( {0.0f, 1.0f, 3.0f} );
    cam.setFovY( 45.0f );
    cam.setAspectRatio( (float)width / (float)height );
}

void printUsageAndExit( const char* argv0 )
{
    std::cerr << "Usage  : " << argv0 << " [options]\n";
    std::cerr << "Options: --file | -f <filename>      Specify file for image output\n";
    std::cerr << "         --help | -h                 Print this usage message\n";
    std::cerr << "         --dim=<width>x<height>      Set image dimensions; defaults to 512x384\n";
    exit( 1 );
}


static void context_log_cb( unsigned int level, const char* tag, const char* message, void* /*cbdata */)
{
    std::cerr << "[" << std::setw( 2 ) << level << "][" << std::setw( 12 ) << tag << "]: "
    << message << "\n";
}

int main( int argc, char* argv[] )
{
    std::string outfile;
    int         width  = 1024;
    int         height =  768;

    for( int i = 1; i < argc; ++i )
    {
        const std::string arg( argv[i] );
        if( arg == "--help" || arg == "-h" )
        {
            printUsageAndExit( argv[0] );
        }
        else if( arg == "--file" || arg == "-f" )
        {
            if( i < argc - 1 )
            {
                outfile = argv[++i];
            }
            else
            {
                printUsageAndExit( argv[0] );
            }
        }
        else if( arg.substr( 0, 6 ) == "--dim=" )
        {
            const std::string dims_arg = arg.substr( 6 );
            sutil::parseDimensions( dims_arg.c_str(), width, height );
        }
        else
        {
            std::cerr << "Unknown option '" << arg << "'\n";
            printUsageAndExit( argv[0] );
        }
    }

    try
    {

        ////////////////////// 1st and 2nd step //////////////////////

        //
        // Initialize CUDA and create OptiX context
        //
        OptixDeviceContext context = nullptr;
        {
            // Initialize CUDA
            CUDA_CHECK( cudaFree( 0 ) );

            // Initialize the OptiX API, loading all API entry points
            OPTIX_CHECK( optixInit() );

            // Specify context options
            OptixDeviceContextOptions options = {};
            options.logCallbackFunction       = &context_log_cb;
            options.logCallbackLevel          = 4;
#ifdef DEBUG
            // This may incur significant performance cost and should only be done during development.
            options.validationMode = OPTIX_DEVICE_CONTEXT_VALIDATION_MODE_ALL;
#endif

            // Associate a CUDA context (and therefore a specific GPU) with this
            // device context
            CUcontext cuCtx = 0;  // zero means take the current context
            OPTIX_CHECK( optixDeviceContextCreate( cuCtx, &options, &context ) );
        }

        //////////////////////////////////////////////////////

        ////////////// Acceleration Structures ///////////////

        std::vector<TriangleMesh> meshes;

        {
            std::vector<float3> means = {
                {0.0f, -0.5f, 0.0f}, {0.0f, 0.1f, 0.2f}, {0.5f, -0.3f, 0.2f}, 
                {-0.4f, 0.5f, -0.1f}, {0.2f, -0.4f, 0.5f}, {0.1f, 0.6f, -0.3f}, 
                {-0.3f, 0.0f, 0.6f}, {0.6f, 0.3f, -0.4f}, {-0.5f, -0.6f, 0.0f}, 
                {0.0f, 0.5f, -0.6f}, {0.7f, -0.3f, 0.1f}, {-0.2f, -0.1f, 0.3f}
            };

            std::vector<float3> scales = {
                {0.08f, 0.02f, 0.05f}, {0.05f, 0.08f, 0.03f}, {0.07f, 0.05f, 0.02f}, 
                {0.06f, 0.03f, 0.08f}, {0.04f, 0.07f, 0.05f}, {0.08f, 0.06f, 0.04f}, 
                {0.05f, 0.03f, 0.07f}, {0.06f, 0.08f, 0.03f}, {0.07f, 0.04f, 0.06f}, 
                {0.04f, 0.06f, 0.08f}, {0.03f, 0.05f, 0.07f}, {0.08f, 0.07f, 0.05f}
            };

            std::vector<float3x3> rotations = {
                {{{1.0f, 0.707f, 0.0f}, {0.0f, 0.707f, -0.707f}, {0.0f, 0.707f, 0.707f}}},
                {{{0.0f, 1.0f, 0.0f}, {0.0f, 1.0f, -0.707f}, {0.0f, 0.234f, 0.43f}}},
                {{{0.866f, 0.5f, 0.0f}, {0.0f, 0.866f, -0.5f}, {0.5f, 0.0f, 0.866f}}},
                {{{0.707f, 0.0f, 0.707f}, {0.0f, 1.0f, 0.0f}, {-0.707f, 0.0f, 0.707f}}},
                {{{1.0f, 0.0f, 0.0f}, {0.0f, 0.866f, -0.5f}, {0.0f, 0.5f, 0.866f}}},
                {{{0.866f, -0.5f, 0.0f}, {0.5f, 0.866f, 0.0f}, {0.0f, 0.0f, 1.0f}}},
                {{{0.5f, 0.866f, 0.0f}, {0.0f, 0.0f, 1.0f}, {0.866f, -0.5f, 0.0f}}},
                {{{0.707f, -0.707f, 0.0f}, {0.707f, 0.707f, 0.0f}, {0.0f, 0.0f, 1.0f}}},
                {{{0.5f, 0.5f, 0.707f}, {0.5f, 0.5f, -0.707f}, {0.707f, -0.707f, 0.0f}}},
                {{{1.0f, 0.0f, 0.0f}, {0.0f, 0.707f, -0.707f}, {0.0f, 0.707f, 0.707f}}},
                {{{0.866f, 0.0f, -0.5f}, {0.5f, 0.866f, 0.0f}, {0.0f, 0.5f, 0.866f}}},
                {{{0.707f, -0.707f, 0.0f}, {0.0f, 0.707f, -0.707f}, {1.0f, 0.0f, 0.0f}}}
            };

            std::vector<float> sigmas = {1.0f, 1.2f, 0.9f, 1.1f, 1.3f, 1.0f, 1.1f, 0.8f, 1.2f, 1.0f, 1.15f, 1.05f};
            std::vector<float> alpha_mins = {0.1f, 0.1f, 0.15f, 0.1f, 0.12f, 0.1f, 0.15f, 0.1f, 0.12f, 0.1f, 0.1f, 0.12f};
            std::vector<float3> colors = {
                {0.8f, 0.2f, 0.2f}, {0.2f, 0.8f, 0.2f}, {0.2f, 0.2f, 0.8f}, 
                {0.8f, 0.8f, 0.2f}, {0.8f, 0.2f, 0.8f}, {0.2f, 0.8f, 0.8f}, 
                {0.6f, 0.4f, 0.2f}, {0.4f, 0.6f, 0.2f}, {0.6f, 0.2f, 0.4f}, 
                {0.4f, 0.2f, 0.6f}, {0.5f, 0.5f, 0.5f}, {0.7f, 0.3f, 0.2f}
            };

            for (int i = 0; i < 12; i++) {
                Icosahedron ico;
                ico.generateUnitIcosahedron();
                subdivide(ico.vertices, ico.indices, 2);
                ico.transform(sigmas[i], alpha_mins[i], scales[i], rotations[i], means[i]);
                TriangleMesh mesh;
                mesh.vertex = ico.vertices;
                mesh.index = ico.indices;
                mesh.color = colors[i];
                mesh.MeshId = i;
                meshes.push_back(mesh);
            }

        }

        // Extract geometry data and prepare build inputs
        std::vector<std::vector<float3>> all_vertices;
        std::vector<std::vector<uint3>> all_indices;

        for (auto &m : meshes) {
            all_vertices.push_back(m.vertex);
            all_indices.push_back(m.index);
        }

        std::vector<CUdeviceptr> d_vertices_array(meshes.size());
        std::vector<CUdeviceptr> d_indices_array(meshes.size());
        std::vector<OptixBuildInput> build_inputs(meshes.size());
        std::vector<uint32_t> input_flags(meshes.size(), OPTIX_GEOMETRY_FLAG_NONE);

        // Upload vertex and index data, create build inputs
        for (size_t i = 0; i < meshes.size(); ++i) {
            const size_t vertices_size = sizeof(float3) * all_vertices[i].size();
            CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_vertices_array[i]), vertices_size));
            CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(d_vertices_array[i]),
                                all_vertices[i].data(),
                                vertices_size,
                                cudaMemcpyHostToDevice));

            const size_t indices_size = sizeof(uint3) * all_indices[i].size();
            CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_indices_array[i]), indices_size));
            CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(d_indices_array[i]),
                                all_indices[i].data(),
                                indices_size,
                                cudaMemcpyHostToDevice));

            OptixBuildInput bi = {};
            bi.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
            bi.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
            bi.triangleArray.vertexStrideInBytes = sizeof(float3);
            bi.triangleArray.numVertices = static_cast<uint32_t>(all_vertices[i].size());
            bi.triangleArray.vertexBuffers = &d_vertices_array[i];

            bi.triangleArray.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
            bi.triangleArray.indexStrideInBytes = sizeof(uint3);
            bi.triangleArray.numIndexTriplets = static_cast<uint32_t>(all_indices[i].size());
            bi.triangleArray.indexBuffer = d_indices_array[i];

            bi.triangleArray.flags = &input_flags[i];
            bi.triangleArray.numSbtRecords = 1; // one record per geometry
            bi.triangleArray.sbtIndexOffsetBuffer = 0;
            bi.triangleArray.sbtIndexOffsetSizeInBytes = 0;
            bi.triangleArray.sbtIndexOffsetStrideInBytes = 0;

            build_inputs[i] = bi;
        }

        // Build acceleration structure with multiple geometries
        OptixAccelBuildOptions accel_options = {};
        accel_options.buildFlags = OPTIX_BUILD_FLAG_NONE | OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
        accel_options.operation = OPTIX_BUILD_OPERATION_BUILD;

        OptixAccelBufferSizes gas_buffer_sizes;
        OPTIX_CHECK(optixAccelComputeMemoryUsage(
            context,
            &accel_options,
            build_inputs.data(),
            static_cast<uint32_t>(build_inputs.size()),
            &gas_buffer_sizes
        ));


        CUdeviceptr d_temp_buffer_gas;
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_temp_buffer_gas), gas_buffer_sizes.tempSizeInBytes));
        CUdeviceptr d_gas_output_buffer;
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_gas_output_buffer), gas_buffer_sizes.outputSizeInBytes));

        CUdeviceptr d_compacted_size_buffer;
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_compacted_size_buffer), sizeof(uint64_t)));

        OptixAccelEmitDesc emitProperty = {};
        emitProperty.type = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
        emitProperty.result = d_compacted_size_buffer;

        OptixTraversableHandle gas_handle;

        {
            OPTIX_CHECK(optixAccelBuild(
                context,
                0, // CUDA stream
                &accel_options,
                build_inputs.data(),
                static_cast<uint32_t>(build_inputs.size()),
                d_temp_buffer_gas,
                gas_buffer_sizes.tempSizeInBytes,
                d_gas_output_buffer,
                gas_buffer_sizes.outputSizeInBytes,
                &gas_handle,
                &emitProperty,
                1
            ));
        }
        CUDA_SYNC_CHECK();

        // Compaction
        uint64_t compacted_gas_size;
        CUDA_CHECK(cudaMemcpy(&compacted_gas_size, reinterpret_cast<void*>(d_compacted_size_buffer),
                            sizeof(uint64_t), cudaMemcpyDeviceToHost));

        if (compacted_gas_size < gas_buffer_sizes.outputSizeInBytes) {
            CUdeviceptr d_compacted_gas_buffer;
            CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_compacted_gas_buffer), compacted_gas_size));

            OPTIX_CHECK(optixAccelCompact(
                context,
                0, // CUDA stream
                gas_handle,
                d_compacted_gas_buffer,
                compacted_gas_size,
                &gas_handle
            ));

            CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_gas_output_buffer)));
            d_gas_output_buffer = d_compacted_gas_buffer;
        }

        // Free temporary buffers
        CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_temp_buffer_gas)));
        CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_compacted_size_buffer)));



        //////////////////////////////////////////////////////

        ////////////////////// 3rd step //////////////////////

        //
        // Create module
        //
        OptixModule module = nullptr;
        OptixPipelineCompileOptions pipeline_compile_options = {};
        {
            OptixModuleCompileOptions module_compile_options = {};
#if !defined(NDEBUG)
            // module_compile_options.maxRegisterCount = 50;  // Set to 0 to use default value
            module_compile_options.optLevel   = OPTIX_COMPILE_OPTIMIZATION_LEVEL_0;
            module_compile_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_FULL;
#endif
            pipeline_compile_options.usesMotionBlur        = false;
            pipeline_compile_options.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
            pipeline_compile_options.numPayloadValues      = 3;   //this is set for just rgb values for now
            pipeline_compile_options.numAttributeValues    = 3;
            pipeline_compile_options.exceptionFlags        = OPTIX_EXCEPTION_FLAG_NONE;  // TODO: should be OPTIX_EXCEPTION_FLAG_STACK_OVERFLOW;
            pipeline_compile_options.pipelineLaunchParamsVariableName = "params";
            pipeline_compile_options.usesPrimitiveTypeFlags = OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE;

            size_t      inputSize = 0;
            const char* input = sutil::getInputData( OPTIX_SAMPLE_NAME, OPTIX_SAMPLE_DIR, "programs.cu", inputSize );

            OPTIX_CHECK_LOG( optixModuleCreate(
                        context,
                        &module_compile_options,
                        &pipeline_compile_options,
                        input,
                        inputSize,
                        LOG, &LOG_SIZE,
                        &module
                        ) );
        }

        //////////////////////////////////////////////////////

        ////////////////////// 4th step //////////////////////

        //
        // Create program groups, including NULL miss and hitgroups
        //
        OptixProgramGroup raygen_prog_group   = nullptr;
        OptixProgramGroup miss_prog_group     = nullptr;
        OptixProgramGroup hitgroup_prog_group = nullptr;
        {
            OptixProgramGroupOptions program_group_options   = {}; // Initialize to zeros

            OptixProgramGroupDesc raygen_prog_group_desc  = {}; //
            raygen_prog_group_desc.kind                     = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
            raygen_prog_group_desc.raygen.module            = module;
            raygen_prog_group_desc.raygen.entryFunctionName = "__raygen__rg";
            OPTIX_CHECK_LOG( optixProgramGroupCreate(
                        context,
                        &raygen_prog_group_desc,
                        1,   // num program groups
                        &program_group_options,
                        LOG, &LOG_SIZE,
                        &raygen_prog_group
                        ) );

            OptixProgramGroupDesc miss_prog_group_desc = {};
            miss_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
            miss_prog_group_desc.miss.module = module;
            miss_prog_group_desc.miss.entryFunctionName = "__miss__ms";
            OPTIX_CHECK_LOG( optixProgramGroupCreate(
                        context,
                        &miss_prog_group_desc,
                        1,   // num program groups
                        &program_group_options,
                        LOG, &LOG_SIZE,
                        &miss_prog_group
                        ) );

            OptixProgramGroupDesc hitgroup_prog_group_desc = {};
            hitgroup_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
            hitgroup_prog_group_desc.hitgroup.moduleCH = module;
            hitgroup_prog_group_desc.hitgroup.entryFunctionNameCH = "__closesthit__ch";
            hitgroup_prog_group_desc.hitgroup.moduleAH = module;
            hitgroup_prog_group_desc.hitgroup.entryFunctionNameAH = "__anyhit__ah";
            OPTIX_CHECK_LOG( optixProgramGroupCreate(
                        context,
                        &hitgroup_prog_group_desc,
                        1,   // num program groups
                        &program_group_options,
                        LOG, &LOG_SIZE,
                        &hitgroup_prog_group
                        ) );
        }

        //////////////////////////////////////////////////////

        ////////////////////// 5th step //////////////////////

        //
        // Link pipeline
        //
        OptixPipeline pipeline = nullptr;
        {
            const uint32_t    max_trace_depth  = 1;  // this is the limit for the trace depth
                                                    // parameter that defines how many times a ray can recursively spawn new rays (e.g., reflection or refraction rays)
            OptixProgramGroup program_groups[] = { raygen_prog_group , miss_prog_group, hitgroup_prog_group };

            OptixPipelineLinkOptions pipeline_link_options = {};
            pipeline_link_options.maxTraceDepth            = max_trace_depth;
            OPTIX_CHECK_LOG( optixPipelineCreate(
                        context,
                        &pipeline_compile_options,
                        &pipeline_link_options,
                        program_groups,
                        sizeof( program_groups ) / sizeof( program_groups[0] ),
                        LOG, &LOG_SIZE,
                        &pipeline
                        ) );

            OptixStackSizes stack_sizes = {};
            for( auto& prog_group : program_groups )
            {
                OPTIX_CHECK( optixUtilAccumulateStackSizes( prog_group, &stack_sizes, pipeline ) );
            }

            uint32_t direct_callable_stack_size_from_traversal;
            uint32_t direct_callable_stack_size_from_state;
            uint32_t continuation_stack_size;
            OPTIX_CHECK( optixUtilComputeStackSizes( &stack_sizes, max_trace_depth,
                                                     0,  // maxCCDepth
                                                     0,  // maxDCDEpth
                                                     &direct_callable_stack_size_from_traversal,
                                                     &direct_callable_stack_size_from_state, &continuation_stack_size ) );
            OPTIX_CHECK( optixPipelineSetStackSize( pipeline, direct_callable_stack_size_from_traversal,
                                                    direct_callable_stack_size_from_state, continuation_stack_size,
                                                    1  // maxTraversableDepth
                                                    ) );
        }

        //////////////////////////////////////////////////////

        ////////////////////// 6th step //////////////////////

        //
        // Set up shader binding table
        //
        OptixShaderBindingTable sbt = {};
        {
            
            // setting SBT data for raygen

            CUdeviceptr  raygen_record;
            const size_t raygen_record_size = sizeof( RayGenSbtRecord );
            CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &raygen_record ), raygen_record_size ) );
            RayGenSbtRecord rg_sbt;
            // rg_sbt.data = nullptr;
            OPTIX_CHECK( optixSbtRecordPackHeader( raygen_prog_group, &rg_sbt ) );
            CUDA_CHECK( cudaMemcpy(
                        reinterpret_cast<void*>( raygen_record ),
                        &rg_sbt,
                        raygen_record_size,
                        cudaMemcpyHostToDevice
                        ) );

            // setting SBT data for miss

            CUdeviceptr miss_record;
            size_t      miss_record_size = sizeof( MissSbtRecord );
            CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &miss_record ), miss_record_size ) );
            MissSbtRecord ms_sbt;
            ms_sbt.data.bg_color = {0.0f, 0.0f, 0.0f};
            // ms_sbt.data = nullptr;
            OPTIX_CHECK( optixSbtRecordPackHeader( miss_prog_group, &ms_sbt ) );
            CUDA_CHECK( cudaMemcpy(
                        reinterpret_cast<void*>( miss_record ),
                        &ms_sbt,
                        miss_record_size,
                        cudaMemcpyHostToDevice
                        ) );

            // setting SBT data for hitgroup

            std::vector<HitGroupSbtRecord> hitgroup_records(meshes.size());

            for (size_t i = 0; i < meshes.size(); ++i) {
                HitGroupSbtRecord hg_sbt;
                OPTIX_CHECK(optixSbtRecordPackHeader(hitgroup_prog_group, &hg_sbt));
                hg_sbt.data.vertex = (float3*)d_vertices_array[i];
                hg_sbt.data.index = (uint3*)d_indices_array[i];
                hg_sbt.data.color = meshes[i].color;
                hg_sbt.data.MeshId = static_cast<unsigned int>(i);
                hitgroup_records[i] = hg_sbt;
            }


            CUdeviceptr d_hitgroup_records;
            CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_hitgroup_records), hitgroup_records.size() * sizeof(HitGroupSbtRecord)));
            CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(d_hitgroup_records),
                                hitgroup_records.data(),
                                hitgroup_records.size() * sizeof(HitGroupSbtRecord),
                                cudaMemcpyHostToDevice));         
            

            sbt.raygenRecord                = raygen_record;
            sbt.missRecordBase              = miss_record;
            sbt.missRecordStrideInBytes     = sizeof( MissSbtRecord );
            sbt.missRecordCount             = 1;
            sbt.hitgroupRecordBase          = d_hitgroup_records;
            sbt.hitgroupRecordStrideInBytes = sizeof(HitGroupSbtRecord);
            sbt.hitgroupRecordCount         = static_cast<uint32_t>(hitgroup_records.size());
        }

        //////////////////////////////////////////////////////

        sutil::CUDAOutputBuffer<uchar4> output_buffer( sutil::CUDAOutputBufferType::CUDA_DEVICE, width, height );

        //
        // launch
        //
        {
            CUstream stream;
            CUDA_CHECK( cudaStreamCreate( &stream ) );

            sutil::Camera cam;
            configureCamera( cam, width, height );

            Params params;
            params.image       = output_buffer.map();
            params.image_width = width;
            params.image_height = height;
            params.handle = gas_handle;
            params.cam_eye = cam.eye();
            cam.UVWFrame( params.cam_u, params.cam_v, params.cam_w );


            CUdeviceptr d_param;
            CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &d_param ), sizeof( Params ) ) );
            CUDA_CHECK( cudaMemcpy(
                        reinterpret_cast<void*>( d_param ),
                        &params, sizeof( params ),
                        cudaMemcpyHostToDevice
                        ) );

            OPTIX_CHECK( optixLaunch( pipeline, stream, d_param, sizeof( Params ), &sbt, width, height, /*depth=*/1 ) );
            CUDA_SYNC_CHECK();

            output_buffer.unmap();
            CUDA_CHECK( cudaFree( reinterpret_cast<void*>( d_param ) ) );
        }

        //
        // Display results
        //
        {
            sutil::ImageBuffer buffer;
            buffer.data         = output_buffer.getHostPointer();
            buffer.width        = width;
            buffer.height       = height;
            buffer.pixel_format = sutil::BufferImageFormat::UNSIGNED_BYTE4;
            if( outfile.empty() )
                sutil::displayBufferWindow( argv[0], buffer );
            else
                sutil::saveImage( outfile.c_str(), buffer, false );
        }

        //
        // Cleanup
        //
        {
            CUDA_CHECK( cudaFree( reinterpret_cast<void*>( sbt.raygenRecord       ) ) );
            CUDA_CHECK( cudaFree( reinterpret_cast<void*>( sbt.missRecordBase     ) ) );
            CUDA_CHECK( cudaFree( reinterpret_cast<void*>( sbt.hitgroupRecordBase ) ) );
            CUDA_CHECK( cudaFree( reinterpret_cast<void*>( d_gas_output_buffer    ) ) );

            for (size_t i = 0; i < meshes.size(); ++i) {
                CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_vertices_array[i])));
                CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_indices_array[i])));
            }

            OPTIX_CHECK( optixPipelineDestroy( pipeline ) );
            OPTIX_CHECK( optixProgramGroupDestroy( hitgroup_prog_group ) );
            OPTIX_CHECK( optixProgramGroupDestroy( miss_prog_group ) );
            OPTIX_CHECK( optixProgramGroupDestroy( raygen_prog_group ) );
            OPTIX_CHECK( optixModuleDestroy( module ) );

            OPTIX_CHECK( optixDeviceContextDestroy( context ) );
        }
    }
    catch( std::exception& e )
    {
        std::cerr << "Caught exception: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
