///
/// pygcode_viewer - Bed Renderer
/// Renders a simple bed outline for visualization
///

#ifndef PYGCODE_BED_RENDERER_HPP
#define PYGCODE_BED_RENDERER_HPP

#include <array>
#include <string>

namespace pygcode {

/// Bed configuration
struct BedConfig {
    // Bed size in mm
    float size_x = 250.0f;  // Prusa MK4 default
    float size_y = 210.0f;
    
    // Bed origin offset (where 0,0 is relative to bed corner)
    float origin_x = 0.0f;
    float origin_y = 0.0f;
    
    // Bed color (RGBA, 0-1 range)
    std::array<float, 4> outline_color = {0.3f, 0.3f, 0.3f, 1.0f};  // Light gray
    std::array<float, 4> grid_color = {0.2f, 0.2f, 0.2f, 1.0f};     // Darker gray
    
    // Visibility
    bool show_outline = true;
    bool show_grid = false;
    float grid_spacing = 10.0f;  // Grid line spacing in mm
    
    // Line width
    float line_width = 2.0f;
    
    // Z position of bed
    float z_position = 0.0f;
};

/// Simple bed outline renderer
class BedRenderer {
public:
    BedRenderer();
    ~BedRenderer();
    
    /// Initialize the renderer (must be called with valid OpenGL context)
    bool init();
    
    /// Set bed configuration
    void set_config(const BedConfig& config);
    
    /// Get current configuration
    const BedConfig& get_config() const { return m_config; }
    
    /// Render the bed outline
    /// view_matrix and projection_matrix should be 4x4 column-major matrices
    void render(const float* view_matrix, const float* projection_matrix);
    
    /// Cleanup OpenGL resources
    void cleanup();
    
private:
    void render_outline(const float* view_matrix, const float* projection_matrix);
    void render_grid(const float* view_matrix, const float* projection_matrix);
    
    BedConfig m_config;
    
    // OpenGL resources
    unsigned int m_shader_program = 0;
    unsigned int m_vao = 0;
    unsigned int m_vbo = 0;
    
    bool m_initialized = false;
};

} // namespace pygcode

#endif // PYGCODE_BED_RENDERER_HPP
