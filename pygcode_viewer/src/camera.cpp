///
/// pygcode_viewer - Camera Implementation
///

#include "camera.hpp"

#include <algorithm>

namespace pygcode {

//=============================================================================
// Camera
//=============================================================================

Camera::Camera() = default;

Camera::Camera(const CameraParams& params) : m_params(params) {}

void Camera::set_params(const CameraParams& params) {
    m_params = params;
}

void Camera::set_position(float x, float y, float z) {
    m_params.position = Vec3f(x, y, z);
}

void Camera::set_position(const Vec3f& pos) {
    m_params.position = pos;
}

void Camera::set_target(float x, float y, float z) {
    m_params.target = Vec3f(x, y, z);
}

void Camera::set_target(const Vec3f& target) {
    m_params.target = target;
}

void Camera::set_up(float x, float y, float z) {
    m_params.up = Vec3f(x, y, z);
}

void Camera::set_up(const Vec3f& up) {
    m_params.up = up;
}

void Camera::set_fov(float fov_degrees) {
    m_params.fov = fov_degrees;
}

void Camera::set_clip_planes(float near_plane, float far_plane) {
    m_params.near_plane = near_plane;
    m_params.far_plane = far_plane;
}

void Camera::set_intrinsics(const CameraIntrinsics& intrinsics) {
    m_params.intrinsics = intrinsics;
    m_params.use_intrinsics = intrinsics.is_valid;
}

void Camera::set_intrinsics(float fx, float fy, float cx, float cy, int width, int height) {
    m_params.intrinsics = CameraIntrinsics(fx, fy, cx, cy, width, height);
    m_params.use_intrinsics = true;
}

void Camera::set_use_intrinsics(bool use) {
    m_params.use_intrinsics = use && m_params.intrinsics.is_valid;
}

libvgcode::Mat4x4 Camera::get_view_matrix() const {
    return MatrixUtils::look_at(m_params.position, m_params.target, m_params.up);
}

libvgcode::Mat4x4 Camera::get_projection_matrix(float aspect_ratio) const {
    constexpr float PI = 3.14159265358979323846f;
    float fov_radians = m_params.fov * PI / 180.0f;
    return MatrixUtils::perspective(fov_radians, aspect_ratio, m_params.near_plane, m_params.far_plane);
}

libvgcode::Mat4x4 Camera::get_projection_matrix(int width, int height) const {
    // If intrinsics are enabled and valid, use them
    if (m_params.use_intrinsics && m_params.intrinsics.is_valid) {
        const auto& intr = m_params.intrinsics;
        return MatrixUtils::perspective_from_intrinsics(
            intr.fx, intr.fy, intr.cx, intr.cy,
            width, height,
            intr.image_width, intr.image_height,
            m_params.near_plane, m_params.far_plane
        );
    }
    
    // Otherwise use standard FOV-based perspective
    float aspect = static_cast<float>(width) / static_cast<float>(height);
    return get_projection_matrix(aspect);
}

libvgcode::Vec3 Camera::get_position_vec3() const {
    return {m_params.position.x, m_params.position.y, m_params.position.z};
}

void Camera::fit_to_bbox(const std::array<float, 6>& bbox, float padding) {
    // Calculate bounding box center and size
    Vec3f min_pt(bbox[0], bbox[1], bbox[2]);
    Vec3f max_pt(bbox[3], bbox[4], bbox[5]);
    
    Vec3f center = (min_pt + max_pt) * 0.5f;
    Vec3f size = max_pt - min_pt;
    
    // Calculate distance to fit the model in view
    float max_size = std::max({size.x, size.y, size.z}) * padding;
    constexpr float PI = 3.14159265358979323846f;
    float fov_radians = m_params.fov * PI / 180.0f;
    float distance = max_size / (2.0f * std::tan(fov_radians / 2.0f));
    
    // Position camera at isometric-ish angle
    Vec3f offset(1.0f, 1.0f, 0.8f);
    offset = offset.normalized() * distance;
    
    m_params.target = center;
    m_params.position = center + offset;
    
    // Adjust clip planes
    m_params.near_plane = distance * 0.01f;
    m_params.far_plane = distance * 10.0f;
}

//=============================================================================
// MatrixUtils
//=============================================================================

namespace MatrixUtils {

libvgcode::Mat4x4 identity() {
    return {
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0,
        0, 0, 0, 1
    };
}

libvgcode::Mat4x4 look_at(const Vec3f& eye, const Vec3f& target, const Vec3f& up) {
    // Calculate forward direction
    Vec3f f = (target - eye).normalized();
    
    // Handle gimbal lock: when forward is nearly parallel to the up vector,
    // the cross product becomes degenerate. Choose an alternative up vector.
    Vec3f working_up = up;
    float parallel_check = std::abs(f.dot(up));
    
    if (parallel_check > 0.99f) {
        // Forward and up are nearly parallel - use an alternative up vector
        // If looking along Y, use Z; if looking along Z, use Y; otherwise use Z
        if (std::abs(f.z) > 0.9f) {
            // Looking up/down along Z - use Y as up
            working_up = Vec3f(0.0f, 1.0f, 0.0f);
        } else if (std::abs(f.y) > 0.9f) {
            // Looking along Y - use Z as up
            working_up = Vec3f(0.0f, 0.0f, 1.0f);
        } else {
            // Looking along X - use Z as up
            working_up = Vec3f(0.0f, 0.0f, 1.0f);
        }
    }
    
    // Calculate right and up vectors
    Vec3f r = f.cross(working_up);
    float r_len = r.length();
    if (r_len < 0.001f) {
        // Still degenerate - force a valid right vector
        r = Vec3f(1.0f, 0.0f, 0.0f);
    } else {
        r = r * (1.0f / r_len);
    }
    Vec3f u = r.cross(f);  // Up (recalculated, always orthogonal)
    
    // Create view matrix (column-major order)
    // This is the inverse of the camera transform
    libvgcode::Mat4x4 m;
    
    // Column 0
    m[0] = r.x;
    m[1] = u.x;
    m[2] = -f.x;
    m[3] = 0.0f;
    
    // Column 1
    m[4] = r.y;
    m[5] = u.y;
    m[6] = -f.y;
    m[7] = 0.0f;
    
    // Column 2
    m[8] = r.z;
    m[9] = u.z;
    m[10] = -f.z;
    m[11] = 0.0f;
    
    // Column 3 (translation)
    m[12] = -r.dot(eye);
    m[13] = -u.dot(eye);
    m[14] = f.dot(eye);
    m[15] = 1.0f;
    
    return m;
}

libvgcode::Mat4x4 perspective(float fov_radians, float aspect, float near_plane, float far_plane) {
    float tan_half_fov = std::tan(fov_radians / 2.0f);
    
    libvgcode::Mat4x4 m = {};  // Zero initialize
    
    m[0] = 1.0f / (aspect * tan_half_fov);
    m[5] = 1.0f / tan_half_fov;
    m[10] = -(far_plane + near_plane) / (far_plane - near_plane);
    m[11] = -1.0f;
    m[14] = -(2.0f * far_plane * near_plane) / (far_plane - near_plane);
    
    return m;
}

libvgcode::Mat4x4 perspective_from_intrinsics(
    float fx, float fy, float cx, float cy,
    int width, int height,
    int calib_width, int calib_height,
    float near_plane, float far_plane
) {
    // Scale intrinsics if render size differs from calibration size
    float scale_x = static_cast<float>(width) / static_cast<float>(calib_width);
    float scale_y = static_cast<float>(height) / static_cast<float>(calib_height);
    
    float fx_scaled = fx * scale_x;
    float fy_scaled = fy * scale_y;
    float cx_scaled = cx * scale_x;
    float cy_scaled = cy * scale_y;
    
    float w = static_cast<float>(width);
    float h = static_cast<float>(height);
    float n = near_plane;
    float f = far_plane;
    
    // OpenGL projection matrix from camera intrinsics
    // This maps from camera coordinates to NDC (Normalized Device Coordinates)
    // 
    // The standard pinhole camera model projects 3D point (X, Y, Z) to 2D:
    //   x = fx * X/Z + cx
    //   y = fy * Y/Z + cy
    //
    // OpenGL NDC maps pixel coordinates to [-1, 1]:
    //   ndc_x = 2*x/width - 1 = 2*fx*X/(Z*width) + (2*cx/width - 1)
    //   ndc_y = 1 - 2*y/height = -2*fy*Y/(Z*height) + (1 - 2*cy/height)
    //
    // Note: OpenGL has Y pointing up, while image coordinates have Y pointing down
    
    libvgcode::Mat4x4 m = {};  // Zero initialize
    
    // Column 0
    m[0] = 2.0f * fx_scaled / w;
    m[1] = 0.0f;
    m[2] = 0.0f;
    m[3] = 0.0f;
    
    // Column 1
    m[4] = 0.0f;
    m[5] = 2.0f * fy_scaled / h;  // Positive because we flip Y in column 2
    m[6] = 0.0f;
    m[7] = 0.0f;
    
    // Column 2
    // Principal point offset and depth mapping
    // The signs account for OpenGL's coordinate system
    m[8] = (w - 2.0f * cx_scaled) / w;   // Maps cx to center
    m[9] = (2.0f * cy_scaled - h) / h;   // Maps cy to center, flips Y
    m[10] = -(f + n) / (f - n);          // Depth mapping
    m[11] = -1.0f;                        // Perspective divide
    
    // Column 3
    m[12] = 0.0f;
    m[13] = 0.0f;
    m[14] = -2.0f * f * n / (f - n);     // Depth offset
    m[15] = 0.0f;
    
    return m;
}

libvgcode::Mat4x4 multiply(const libvgcode::Mat4x4& a, const libvgcode::Mat4x4& b) {
    libvgcode::Mat4x4 result = {};
    
    for (int col = 0; col < 4; ++col) {
        for (int row = 0; row < 4; ++row) {
            float sum = 0.0f;
            for (int k = 0; k < 4; ++k) {
                sum += a[k * 4 + row] * b[col * 4 + k];
            }
            result[col * 4 + row] = sum;
        }
    }
    
    return result;
}

} // namespace MatrixUtils

} // namespace pygcode
