///
/// pygcode_viewer - Camera Utilities
/// Simple camera from position/target to view/projection matrices
///

#ifndef PYGCODE_CAMERA_HPP
#define PYGCODE_CAMERA_HPP

#include <array>
#include <cmath>

#include "Types.hpp"

namespace pygcode {

/// Simple 3D vector operations
struct Vec3f {
    float x, y, z;
    
    Vec3f() : x(0), y(0), z(0) {}
    Vec3f(float x_, float y_, float z_) : x(x_), y(y_), z(z_) {}
    Vec3f(const std::array<float, 3>& arr) : x(arr[0]), y(arr[1]), z(arr[2]) {}
    
    Vec3f operator+(const Vec3f& other) const {
        return Vec3f(x + other.x, y + other.y, z + other.z);
    }
    
    Vec3f operator-(const Vec3f& other) const {
        return Vec3f(x - other.x, y - other.y, z - other.z);
    }
    
    Vec3f operator*(float s) const {
        return Vec3f(x * s, y * s, z * s);
    }
    
    float dot(const Vec3f& other) const {
        return x * other.x + y * other.y + z * other.z;
    }
    
    Vec3f cross(const Vec3f& other) const {
        return Vec3f(
            y * other.z - z * other.y,
            z * other.x - x * other.z,
            x * other.y - y * other.x
        );
    }
    
    float length() const {
        return std::sqrt(x * x + y * y + z * z);
    }
    
    Vec3f normalized() const {
        float len = length();
        if (len > 0) {
            return *this * (1.0f / len);
        }
        return *this;
    }
    
    std::array<float, 3> to_array() const {
        return {x, y, z};
    }
};

/// Camera parameters
struct CameraParams {
    Vec3f position{100.0f, 100.0f, 100.0f};
    Vec3f target{0.0f, 0.0f, 50.0f};
    Vec3f up{0.0f, 0.0f, 1.0f};
    float fov = 45.0f;  // Field of view in degrees
    float near_plane = 0.1f;
    float far_plane = 10000.0f;
    
    CameraParams() = default;
    
    CameraParams(const std::array<float, 3>& pos,
                 const std::array<float, 3>& tgt,
                 const std::array<float, 3>& up_vec = {0, 0, 1},
                 float fov_deg = 45.0f)
        : position(pos), target(tgt), up(up_vec), fov(fov_deg) {}
};

/// Camera class for computing view and projection matrices
class Camera {
public:
    Camera();
    explicit Camera(const CameraParams& params);
    
    /// Set camera parameters
    void set_params(const CameraParams& params);
    
    /// Get current parameters
    const CameraParams& get_params() const { return m_params; }
    
    /// Set position
    void set_position(float x, float y, float z);
    void set_position(const Vec3f& pos);
    
    /// Set target (look-at point)
    void set_target(float x, float y, float z);
    void set_target(const Vec3f& target);
    
    /// Set up vector
    void set_up(float x, float y, float z);
    void set_up(const Vec3f& up);
    
    /// Set field of view (degrees)
    void set_fov(float fov_degrees);
    
    /// Set near/far planes
    void set_clip_planes(float near_plane, float far_plane);
    
    /// Compute view matrix (camera space)
    libvgcode::Mat4x4 get_view_matrix() const;
    
    /// Compute projection matrix (perspective)
    libvgcode::Mat4x4 get_projection_matrix(float aspect_ratio) const;
    libvgcode::Mat4x4 get_projection_matrix(int width, int height) const;
    
    /// Get camera position for shader use
    libvgcode::Vec3 get_position_vec3() const;
    
    /// Auto-fit camera to view a bounding box
    void fit_to_bbox(const std::array<float, 6>& bbox, float padding = 1.2f);
    
private:
    CameraParams m_params;
};

/// Matrix utilities
namespace MatrixUtils {
    /// Create identity matrix
    libvgcode::Mat4x4 identity();
    
    /// Create look-at matrix (view matrix)
    libvgcode::Mat4x4 look_at(const Vec3f& eye, const Vec3f& target, const Vec3f& up);
    
    /// Create perspective projection matrix
    libvgcode::Mat4x4 perspective(float fov_radians, float aspect, float near_plane, float far_plane);
    
    /// Multiply two 4x4 matrices
    libvgcode::Mat4x4 multiply(const libvgcode::Mat4x4& a, const libvgcode::Mat4x4& b);
}

} // namespace pygcode

#endif // PYGCODE_CAMERA_HPP
