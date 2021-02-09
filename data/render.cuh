#pragma once

class vec {
public:
    float x;
    float y;
    float z;

    __device__ vec(float _x = 0.0f, float _y = 0.0f, float _z = 0.0f)
        : x(_x), y(_y), z(_z) {
        
    }

    __device__ float normSqr() const {
        return x * x + y * y + z * z;
    }

    __device__ float norm() const {
        return norm3df(x, y, z);
    }

    __device__ vec unit() const {
        const float k = rnorm3df(x, y, z);
        return vec(
            x * k,
            y * k,
            z * k
        );
    }

    
    __device__ vec& operator+=(const vec& v) {
        x += v.x;
        y += v.y;
        z += v.z;
        return *this;
    }
    __device__ vec& operator-=(const vec& v) {
        x -= v.x;
        y -= v.y;
        z -= v.z;
        return *this;
    }

    __device__ vec operator-() const {
        return vec(
            -x,
            -y,
            -z
        );
    }
};


__device__ vec operator+(const vec& u, const vec& v) {
    return vec(
        u.x + v.x,
        u.y + v.y,
        u.z + v.z
    );
}
__device__ vec operator-(const vec& u, const vec& v) {
    return vec(
        u.x + v.x,
        u.y + v.y,
        u.z + v.z
    );
}

__device__ vec operator*(const vec& v, float k) {
    return vec(
        v.x * k,
        v.y * k,
        v.z * k
    );
}
__device__ vec operator*(float k, const vec& v) {
    return vec(
        v.x * k,
        v.y * k,
        v.z * k
    );
}

__device__ float operator*(const vec& u, const vec& v) {
    return u.x * v.x + u.y * v.y + u.z * v.z;
}

__device__ float wrapNearZero(float v) {
    const float l = 1.0f;
    const float r = 2.0f;
    if (v < -l) {
        return v + l;
    } else if (v > l) {
        return v - l;
    } else {
        v *= r;
        return (v - roundf(v)) / r;
    }
}


__device__ float signedDistance(vec v, float power) {
    auto z = v;
    auto dr = 1.0f;
    auto r = 0.0f;

    // TODO: Inigo Quilez has a crazily optimized version of the Mandelbulb
    // that uses only polynomials and a single inverse sqrt
    // See https://www.iquilezles.org/www/articles/mandelbulb/mandelbulb.htm

    constexpr int N = 24;
    for (int i = 0; i < N; ++i) {
        r = z.norm();
        if (r > 2.0f) {
            break;
        }

        const auto theta = acos(z.z / r) * power;
        const auto phi = atan2f(z.y, z.x) * power;
        const auto zr = powf(r, power);
        dr = powf(r, power - 1.0f) * power * dr + 1.0f;
        float st, ct, sp, cp;
        sincosf(theta, &st, &ct);
        sincosf(phi, &sp, &cp);
        z = zr * vec(
            st * cp,
            st * sp,
            ct
        );
        z += v;
    }

    return 0.5f * logf(r) * r / dr;
}

__device__ float phong(vec v, vec l) {
    return max(0.0f, v * l);
}

__device__ vec bounce(vec v, vec n) {
    return v - (2.0f * (v * n) * n);
}

template<int N>
__device__ vec rayMarch(vec p, vec d, float power) {
    const float minDist = 1e-4f;
    const auto lightDir = vec(0.5f, -1.0f, -0.3f).unit();
    bool hit = false;
    int stepsBeforeHit = 0;
    float distSinceHit = 0.0f;
    float smallestLightTanAngle = 100.0f;
    auto colour = vec(0.2f, 0.4f, 1.0f);

    // NOTE: because of numeric instability, marching by the full distance estimate at each
    // step can cause rays to end up inside of obstacles.
    const auto stepSize = 0.5f;

    for (int i = 0; i < N; ++i) {
        const float dist = signedDistance(p, power);
        if (hit) {
            distSinceHit += abs(dist);
            const float lightTanAngle = abs(dist) / distSinceHit;
            smallestLightTanAngle = min(smallestLightTanAngle, lightTanAngle);
        } else {
            if (dist < minDist) {
                hit = true;
                colour = vec(1.0f, 1.0f, 1.0f); // TODO: more interesting colours
                d = lightDir;
            }
            ++stepsBeforeHit;
        }
        p += dist * d * stepSize;
    }
    const auto smallestLightAngle = atanf(smallestLightTanAngle);
    const auto maxShadowRad = 0.2f;
    const auto shadeRatio = min(smallestLightAngle, maxShadowRad) / maxShadowRad;
    const auto aoRatio = static_cast<float>(stepsBeforeHit) / static_cast<float>(N);
    return colour * 0.5f * (shadeRatio + aoRatio);
}

template<int N>
__global__ void renderKernel(float* out_data, size_t sizeX, size_t sizeY, float angle, float power) {
    const int ix = blockIdx.x * blockDim.x + threadIdx.x;
    const int iy = blockIdx.y * blockDim.y + threadIdx.y;
    if (ix >= sizeX || iy >= sizeY){
        return;
    }

    const float screenX = -1.0f + 2.0f * static_cast<float>(ix) / static_cast<float>(sizeX);
    const float screenY = -1.0f + 2.0f * static_cast<float>(iy) / static_cast<float>(sizeY);

    const auto cameraDist = 5.0f;

    float sinAngle;
    float cosAngle;
    sincosf(angle, &sinAngle, &cosAngle);

    const auto cameraPos = vec(
        cameraDist * cosAngle,
        0.0f,
        cameraDist * sinAngle
    );
    const auto cameraDir = vec(
        -cosAngle,
        0.0f,
        -sinAngle
    );

    const auto fovFactor = 0.2f;

    const auto cameraDisplacement = vec(
        sinAngle * screenX,
        screenY,
        -cosAngle * screenX
    );

    const auto cameraSensorPos = cameraPos + cameraDisplacement;
    const auto cameraSensorDir = cameraDir + fovFactor * cameraDisplacement;

    const auto colour = rayMarch<N>(cameraSensorPos, cameraSensorDir, power);

    const auto base = 4 * (ix + iy * sizeX);
    out_data[base + 0] = colour.x;
    out_data[base + 1] = colour.y;
    out_data[base + 2] = colour.z;
}
