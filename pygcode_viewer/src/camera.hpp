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

/// Camera intrinsic matrix parameters
struct CameraIntrinsics {
    float fx = 0.0f;  // Focal length x (pixels)
    float fy = 0.0f;  // Focal length y (pixels)
    float cx = 0.0f;  // Principal point x (pixels)
    float cy = 0.0f;  // Principal point y (pixels)
    int image_width = 0;   // Original calibration image width
    int image_height = 0;  // Original calibration image height
    bool is_valid = false;
    
    CameraIntrinsics() = default;
    
    CameraIntrinsics(float fx_, float fy_, float cx_, float cy_, int width, int height)
        : fx(fx_), fy(fy_), cx(cx_), cy(cy_), 
          image_width(width), image_height(height), is_valid(true) {}
    
    /// Create from 3x3 camera matrix (row-major)
    static CameraIntrinsics from_matrix(const std::array<std::array<float, 3>, 3>& matrix,
                                        int width, int height) {
        return CameraIntrinsics(
            matrix[0][0],  // fx
            matrix[1][1],  // fy
            matrix[0][2],  // cx
            matrix[1][2],  // cy
            width, height
        );
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
    
    // Optional intrinsic matrix for realistic camera projection
    CameraIntrinsics intrinsics;
    bool use_intrinsics = false;
    
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
    
    /// Set camera intrinsics from calibration data
    void set_intrinsics(const CameraIntrinsics& intrinsics);
    void set_intrinsics(float fx, float fy, float cx, float cy, int width, int height);
    
    /// Enable/disable use of intrinsic matrix for projection
    void set_use_intrinsics(bool use);
    bool get_use_intrinsics() const { return m_params.use_intrinsics; }
    
    /// Get intrinsics
    const CameraIntrinsics& get_intrinsics() const { return m_params.intrinsics; }
    
    /// Compute view matrix (camera space)
    libvgcode::Mat4x4 get_view_matrix() const;
    
    /// Compute projection matrix (perspective)
    /// If intrinsics are set and enabled, uses intrinsic matrix
    /// Otherwise uses FOV-based perspective
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
    
    /// Create perspective projection matrix from FOV
    libvgcode::Mat4x4 perspective(float fov_radians, float aspect, float near_plane, float far_plane);
    
    /// Create perspective projection matrix from camera intrinsics
    /// This converts calibrated camera intrinsics to OpenGL projection matrix
    /// fx, fy: focal lengths in pixels
    /// cx, cy: principal point in pixels
    /// width, height: render target size (may differ from calibration image size)
    /// calib_width, calib_height: original calibration image size
    libvgcode::Mat4x4 perspective_from_intrinsics(
        float fx, float fy, float cx, float cy,
        int width, int height,
        int calib_width, int calib_height,
        float near_plane, float far_plane
    );
    
    /// Multiply two 4x4 matrices
    libvgcode::Mat4x4 multiply(const libvgcode::Mat4x4& a, const libvgcode::Mat4x4& b);
}

} // namespace pygcode

#endif // PYGCODE_CAMERA_HPP
