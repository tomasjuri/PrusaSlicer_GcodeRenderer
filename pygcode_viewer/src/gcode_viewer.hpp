///
/// pygcode_viewer - GCode Viewer
/// Main viewer class that ties together all components
///

#ifndef PYGCODE_GCODE_VIEWER_HPP
#define PYGCODE_GCODE_VIEWER_HPP

#include <string>
#include <memory>
#include <array>
#include <map>

#include "gcode_parser.hpp"
#include "camera.hpp"
#include "config_parser.hpp"

namespace libvgcode {
class Viewer;
}

namespace pygcode {

class HeadlessContext;
class FramebufferRenderer;

/// Main GCode Viewer class
class GCodeViewerCore {
public:
    GCodeViewerCore();
    ~GCodeViewerCore();
    
    // No copy
    GCodeViewerCore(const GCodeViewerCore&) = delete;
    GCodeViewerCore& operator=(const GCodeViewerCore&) = delete;
    
    /// Load a GCODE file
    void load(const std::string& filepath);
    
    /// Check if GCODE is loaded
    bool is_loaded() const { return m_loaded; }
    
    /// Get bounding box (min_x, min_y, min_z, max_x, max_y, max_z)
    std::array<float, 6> get_bounding_box() const;
    
    /// Get layer count
    size_t get_layer_count() const;
    
    /// Get estimated print time in seconds
    float get_estimated_time() const;
    
    /// Set the layer range to display
    void set_layer_range(size_t min_layer, size_t max_layer);
    
    /// Set the view type (FeatureType, Height, Width, Speed, etc.)
    void set_view_type(const std::string& view_type);
    
    /// Apply configuration
    void apply_config(const std::string& config_json);
    
    /// Render to PNG file
    void render_to_file(
        const std::string& output_path,
        int width,
        int height,
        const std::map<std::string, std::array<float, 3>>& camera_params,
        const std::string& config_json
    );
    
    /// Simplified render interface with camera dict
    void render_to_file(
        const std::string& output_path,
        int width,
        int height,
        float cam_pos_x, float cam_pos_y, float cam_pos_z,
        float cam_target_x, float cam_target_y, float cam_target_z,
        const std::string& config_json
    );
    
private:
    void ensure_context();
    void init_viewer();
    
    std::unique_ptr<HeadlessContext> m_context;
    std::unique_ptr<FramebufferRenderer> m_framebuffer;
    std::unique_ptr<libvgcode::Viewer> m_viewer;
    
    GCodeParseResult m_parse_result;
    Camera m_camera;
    ViewConfig m_config;
    
    bool m_loaded = false;
    bool m_viewer_initialized = false;
    int m_current_width = 0;
    int m_current_height = 0;
};

} // namespace pygcode

#endif // PYGCODE_GCODE_VIEWER_HPP
