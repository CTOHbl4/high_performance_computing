#define PI 3.14159265
#define ENVIRONMENT_RADIUS 100000.0
#define MIN_T 0.001

struct Ray {
    float3 origin;
    float3 direction;
};

struct HitRecord {
    float3 color;
};

uniform uint2 size;
StructuredBuffer<float> cameras;
uniform uint spheresNum;
StructuredBuffer<float> spheres;
StructuredBuffer<float> functionValues;
StructuredBuffer<float> radiuses;
RWTexture2D<float> output;

bool hitSphere(Ray ray, float3 sphereCenter, float sphereRadius, float mint, inout float maxt, inout HitRecord rec) {
    float3 oc = ray.origin - sphereCenter;
    float a = dot(ray.direction, ray.direction);
    float b = dot(oc, ray.direction);
    float c = dot(oc, oc) - sphereRadius * sphereRadius;
    float discriminant = b * b - a * c;
    
    if (discriminant < 0) {
        return false;
    }
    
    float sqrtd = sqrt(discriminant);

    float root = (-b - sqrtd) / a;
    if (root < mint || root > maxt) {
        root = (-b + sqrtd) / a;
        if (root < mint || root > maxt) {
            return false;
        }
    }
    
    maxt = root;
    return true;
}

void hitScene(Ray ray, inout HitRecord rec) {
    HitRecord tempRec;
    float closest = 2.0 * ENVIRONMENT_RADIUS;
    float mint = MIN_T;
    float value = 0.5;
    float curRadius = 0.0;
    bool foundHit = false;

    for (uint i = 0; i < spheresNum; i++) {
        uint baseIdx = i * 3;
        float3 sphereCenter = float3(spheres[baseIdx], spheres[baseIdx + 1], spheres[baseIdx + 2]);
        value = functionValues[i];
        curRadius = radiuses[i];

        if (hitSphere(ray, sphereCenter, curRadius, mint, closest, tempRec)) {
            rec = tempRec;
            rec.color = float3(1.0 - abs(value), 0.5 - value / 2.0, 0.5 + value / 2.0);
            foundHit = true;
        }
    }

    if (!foundHit) {
        rec.color = float3(0.9, 0.9, 0.9);
    }
}

[shader("compute")]
void pathsTrace(uint3 tid : SV_DispatchThreadID) {

    float3 cameraPos = float3(cameras[0], cameras[1], cameras[2]);
    float3 w = float3(cameras[3], cameras[4], cameras[5]);
    float3 uvec = float3(cameras[6], cameras[7], cameras[8]);
    float3 vvec = float3(cameras[9], cameras[10], cameras[11]);
    float tanfovdiv2 = cameras[12];
    float aspect = cameras[13];

    HitRecord rec;
    Ray ray;

    float u = ((tid.x + 0.5) / size.x * 2.0 - 1.0) * tanfovdiv2 * aspect;
    float v = ((tid.y + 0.5) / size.y * 2.0 - 1.0) * tanfovdiv2;
    float3 rayDir = normalize(u * uvec + v * vvec + w);

    ray.origin = cameraPos;
    ray.direction = rayDir;

    hitScene(ray, rec);

    uint2 idx = uint2(tid.x * 3, tid.y);
    output[idx] = rec.color.r;
    output[uint2(idx.x + 1, idx.y)] = rec.color.g;
    output[uint2(idx.x + 2, idx.y)] = rec.color.b;
}