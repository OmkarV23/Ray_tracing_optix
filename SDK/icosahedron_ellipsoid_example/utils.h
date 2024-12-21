#pragma once

#include <vector>
#include <cuda_runtime.h>
#include <cmath>
#include "sutil/vec_math.h"
#include <map>

struct Params
{
    uchar4* image;
    unsigned int image_width;
    unsigned int image_height;
    float3 cam_eye;
    float3 cam_u, cam_v, cam_w;
    OptixTraversableHandle handle;
};

struct RayGenData
{
    // float r,g,b;
};

struct MissData
{
    float3 bg_color;
};

struct HitGroupData
{
    float3  color;
    float3 *vertex;
    uint3 *index;
    int MeshId;
};

struct TriangleMesh
{
    std::vector<float3> vertex;  // array of vertices
    std::vector<uint3>   index;   // each triangle = 3 vertex indices
    float3              color;
    int MeshId;
};

struct float3x3
{
    float m[3][3];
};

struct Icosahedron
{
    std::vector<float3> vertices;
    std::vector<uint3> indices;

    // Generate a unit icosahedron
    void generateUnitIcosahedron()
    {
        const float t = (1.0f + sqrtf(5.0f)) / 2.0f; // Golden ratio

        // Define 12 vertices of an icosahedron
        vertices = {
            {-1,  t,  0}, { 1,  t,  0}, {-1, -t,  0}, { 1, -t,  0},
            { 0, -1,  t}, { 0,  1,  t}, { 0, -1, -t}, { 0,  1, -t},
            { t,  0, -1}, { t,  0,  1}, {-t,  0, -1}, {-t,  0,  1}
        };

        // Normalize to unit sphere
        for (auto& v : vertices)
        {
            float length = sqrtf(v.x * v.x + v.y * v.y + v.z * v.z);
            v.x /= length;
            v.y /= length;
            v.z /= length;
        }

        // Define the 20 triangular faces
        indices = {
            {0, 11, 5}, {0, 5, 1}, {0, 1, 7}, {0, 7, 10}, {0, 10, 11},
            {1, 5, 9}, {5, 11, 4}, {11, 10, 2}, {10, 7, 6}, {7, 1, 8},
            {3, 9, 4}, {3, 4, 2}, {3, 2, 6}, {3, 6, 8}, {3, 8, 9},
            {4, 9, 5}, {2, 4, 11}, {6, 2, 10}, {8, 6, 7}, {9, 8, 1}
        };
    }

    // Transform vertices based on the given parameters
    void transform(const float sigma, const float alpha_min, const float3& scale, const float3x3& rotation, const float3& mean)
    {
        float factor = sqrtf(2 * logf(sigma / alpha_min));
        for (auto& v : vertices)
        {
            // Scale, rotate, and translate
            float3 scaled_v = make_float3(scale.x * v.x, scale.y * v.y, scale.z * v.z);
            float3 rotated_v = make_float3(
                rotation.m[0][0] * scaled_v.x + rotation.m[0][1] * scaled_v.y + rotation.m[0][2] * scaled_v.z,
                rotation.m[1][0] * scaled_v.x + rotation.m[1][1] * scaled_v.y + rotation.m[1][2] * scaled_v.z,
                rotation.m[2][0] * scaled_v.x + rotation.m[2][1] * scaled_v.y + rotation.m[2][2] * scaled_v.z
            );
            v = make_float3(factor * rotated_v.x + mean.x, factor * rotated_v.y + mean.y, factor * rotated_v.z + mean.z);
        }
    }

};

void subdivide(std::vector<float3>& vertices, std::vector<uint3>& indices, int subdivisions) {
    std::map<std::pair<int, int>, int> midpoint_cache;

    auto getMidpoint = [&](int v1, int v2) {
        if (v1 > v2) std::swap(v1, v2);
        auto edge = std::make_pair(v1, v2);
        if (midpoint_cache.count(edge)) return midpoint_cache[edge];

        float3 midpoint = make_float3(
            (vertices[v1].x + vertices[v2].x) / 2,
            (vertices[v1].y + vertices[v2].y) / 2,
            (vertices[v1].z + vertices[v2].z) / 2
        );
        midpoint = normalize(midpoint); // Project onto the unit sphere
        vertices.push_back(midpoint);
        return midpoint_cache[edge] = vertices.size() - 1;
    };

    for (int i = 0; i < subdivisions; ++i) {
        std::vector<uint3> new_indices;

        for (const auto& tri : indices) {
            int a = getMidpoint(tri.x, tri.y);
            int b = getMidpoint(tri.y, tri.z);
            int c = getMidpoint(tri.z, tri.x);

            new_indices.push_back(make_uint3(tri.x, a, c));
            new_indices.push_back(make_uint3(tri.y, b, a));
            new_indices.push_back(make_uint3(tri.z, c, b));
            new_indices.push_back(make_uint3(a, b, c));

        }

        indices = new_indices;
    }
}