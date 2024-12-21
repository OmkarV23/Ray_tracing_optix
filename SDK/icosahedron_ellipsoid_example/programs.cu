#include <optix.h>

#include "utils.h"
#include <cuda/helpers.h>

extern "C" {
__constant__ Params params;
}

static __forceinline__ __device__ void setPayload( float3 p )
{   
    // Custom payload, each payload for each channel (r, g, b)

    optixSetPayload_0( __float_as_uint( p.x ) );
    optixSetPayload_1( __float_as_uint( p.y ) );
    optixSetPayload_2( __float_as_uint( p.z ) );
}

static __forceinline__ __device__ void computeRay( uint3 idx, uint3 dim, float3& origin, float3& direction )
{
    const float3 U = params.cam_u;
    const float3 V = params.cam_v;
    const float3 W = params.cam_w;
    const float2 d = 2.0f * make_float2(
            static_cast<float>( idx.x ) / static_cast<float>( dim.x ),
            static_cast<float>( idx.y ) / static_cast<float>( dim.y )
            ) - 1.0f;

    origin    = params.cam_eye;
    direction = normalize( d.x * U + d.y * V + W );
}

extern "C"
__global__ void __raygen__rg()
{
    // Lookup our location within the launch grid
    const uint3 idx = optixGetLaunchIndex();
    const uint3 dim = optixGetLaunchDimensions();

    // Map our launch idx to a screen location and create a ray from the camera
    // location through the screen
    float3 ray_origin, ray_direction;
    computeRay( idx, dim, ray_origin, ray_direction );


   // Trace the ray against our scene hierarchy
    unsigned int p0, p1, p2; // Payloads
    optixTrace(
            params.handle,
            ray_origin,
            ray_direction,
            0.0f,                // Min intersection distance
            1e16f,               // Max intersection distance
            0.0f,                // rayTime -- used for motion blur
            OptixVisibilityMask( 255 ), // Specify always visible
            OPTIX_RAY_FLAG_NONE,
            0,                   // SBT offset   -- See SBT discussion
            1,                   // SBT stride   -- See SBT discussion
            0,                   // missSBTIndex -- See SBT discussion
            p0, p1, p2 );
    float3 result;
    result.x = __uint_as_float( p0 );
    result.y = __uint_as_float( p1 );
    result.z = __uint_as_float( p2 );

    // Record results in our output raster
    params.image[idx.y * params.image_width + idx.x] = make_color( result );
}


extern "C" __global__ void __miss__ms()
{
    MissData* miss_data  = reinterpret_cast<MissData*>( optixGetSbtDataPointer() );
    setPayload(  miss_data->bg_color );
}


extern "C" __global__ void __anyhit__ah()
{
    const HitGroupData &mesh = *(reinterpret_cast<HitGroupData*>( optixGetSbtDataPointer() ));

    const int primIdx = optixGetPrimitiveIndex();

    const uint3 index = mesh.index[primIdx];

    // const int meshid = optixGetInstanceId();
    const int meshid = mesh.MeshId;

    const float3 v0 = mesh.vertex[index.x];
    const float3 v1 = mesh.vertex[index.y];
    const float3 v2 = mesh.vertex[index.z];

    const float3 n = normalize( cross( v1 - v0, v2 - v0 ) );
    const float3 rayDir = optixGetWorldRayDirection();
    const float3 hitPoint = optixGetWorldRayOrigin() + optixGetRayTmax() * rayDir;
    const float3 L = normalize( hitPoint - params.cam_eye );

    const float3 color = mesh.color;
    const float3 ambient = 0.1f * color;
    const float3 diffuse = fmaxf( 0.0f, dot( n, L ) ) * color;
    const float3 result = ambient + diffuse;

    setPayload( result );
}

extern "C" __global__ void __closesthit__ch()
{
    // set baricentric coordinates
    float2 barycentrics = optixGetTriangleBarycentrics();
    setPayload( make_float3( barycentrics, 1.0f ) );
}