///
/// pygcode_viewer - GCode Viewer Implementation
///

#include "gcode_viewer.hpp"
#include "headless_context.hpp"
#include "framebuffer_renderer.hpp"
#include "Viewer.hpp"
#include "GCodeInputData.hpp"

#include <stdexcept>

namespace pygcode {

GCodeViewerCore::GCodeViewerCore() {
    m_config.init_defaults();
    // Default bed config for Prusa MK4
    m_bed_config.size_x = 250.0f;
    m_bed_config.size_y = 210.0f;
    m_bed_config.show_outline = true;
    m_bed_config.show_grid = false;
}

GCodeViewerCore::~GCodeViewerCore() {
    // Cleanup in correct order
    if (m_viewer) {
        m_viewer->shutdown();
    }
    m_viewer.reset();
    m_bed_renderer.reset();
    m_framebuffer.reset();
    if (m_context) {
        m_context->release();
    }
    m_context.reset();
}

void GCodeViewerCore::ensure_context() {
    if (!m_context || !m_context->is_valid()) {
        m_context = HeadlessContext::create_context();
        if (!m_context->create(1920, 1080)) {
            throw std::runtime_error("Failed to create headless OpenGL context");
        }
    }
    m_context->make_current();
}

void GCodeViewerCore::init_viewer() {
    if (!m_viewer_initialized) {
        ensure_context();
        
        m_viewer = std::make_unique<libvgcode::Viewer>();
        m_viewer->init(m_context->get_version_string());
        
        // Initialize bed renderer
        m_bed_renderer = std::make_unique<BedRenderer>();
        m_bed_renderer->init();
        m_bed_renderer->set_config(m_bed_config);
        
        m_viewer_initialized = true;
    }
}

void GCodeViewerCore::set_bed_config(const BedConfig& config) {
    m_bed_config = config;
    if (m_bed_renderer) {
        m_bed_renderer->set_config(config);
    }
}

void GCodeViewerCore::load(const std::string& filepath) {
    // Parse GCODE file
    GCodeParser parser;
    m_parse_result = parser.parse(filepath);
    
    // Ensure OpenGL context and viewer
    init_viewer();
    
    // Load data into viewer
    m_viewer->load(std::move(m_parse_result.data));
    
    // Auto-fit camera
    m_camera.fit_to_bbox(m_parse_result.bounding_box);
    
    m_loaded = true;
}

std::array<float, 6> GCodeViewerCore::get_bounding_box() const {
    if (!m_loaded) {
        return {0, 0, 0, 0, 0, 0};
    }
    return m_parse_result.bounding_box;
}

size_t GCodeViewerCore::get_layer_count() const {
    if (!m_loaded || !m_viewer) {
        return 0;
    }
    return m_viewer->get_layers_count();
}

float GCodeViewerCore::get_estimated_time() const {
    if (!m_loaded || !m_viewer) {
        return 0.0f;
    }
    return m_viewer->get_estimated_time();
}

void GCodeViewerCore::set_layer_range(size_t min_layer, size_t max_layer) {
    if (!m_loaded || !m_viewer) {
        throw std::runtime_error("No GCODE loaded. Call load() first.");
    }
    
    size_t layer_count = m_viewer->get_layers_count();
    if (layer_count == 0) return;
    
    // Clamp to valid range
    min_layer = std::min(min_layer, layer_count - 1);
    max_layer = std::min(max_layer, layer_count - 1);
    if (max_layer < min_layer) max_layer = min_layer;
    
    // Set layers view range
    m_viewer->set_layers_view_range(min_layer, max_layer);
    
    // Set view visible range to show all data up to max layer
    // The enabled range reflects what's available in the current layer range
    auto enabled_range = m_viewer->get_view_enabled_range();
    m_viewer->set_view_visible_range(enabled_range[0], enabled_range[1]);
    
    // Store in config
    m_config.layer_min = min_layer;
    m_config.layer_max = max_layer;
    m_config.has_layer_range = true;
}

void GCodeViewerCore::set_view_type(const std::string& view_type) {
    if (!m_loaded || !m_viewer) {
        throw std::runtime_error("No GCODE loaded. Call load() first.");
    }
    
    libvgcode::EViewType type = ConfigParser::parse_view_type(view_type);
    m_viewer->set_view_type(type);
    m_config.view_type = type;
}

void GCodeViewerCore::apply_config(const std::string& config_json) {
    if (!m_loaded || !m_viewer) {
        throw std::runtime_error("No GCODE loaded. Call load() first.");
    }
    
    if (!config_json.empty()) {
        m_config = ConfigParser::parse_json(config_json);
    }
    ConfigParser::apply_config(*m_viewer, m_config);
}

void GCodeViewerCore::render_to_file(
    const std::string& output_path,
    int width,
    int height,
    const std::map<std::string, std::array<float, 3>>& camera_params,
    const std::string& config_json
) {
    // Extract camera parameters
    float pos_x = 100, pos_y = 100, pos_z = 100;
    float target_x = 0, target_y = 0, target_z = 50;
    
    auto it = camera_params.find("position");
    if (it != camera_params.end()) {
        pos_x = it->second[0];
        pos_y = it->second[1];
        pos_z = it->second[2];
    }
    
    it = camera_params.find("target");
    if (it != camera_params.end()) {
        target_x = it->second[0];
        target_y = it->second[1];
        target_z = it->second[2];
    }
    
    render_to_file(output_path, width, height,
                   pos_x, pos_y, pos_z,
                   target_x, target_y, target_z,
                   config_json);
}

void GCodeViewerCore::render_to_file(
    const std::string& output_path,
    int width,
    int height,
    float cam_pos_x, float cam_pos_y, float cam_pos_z,
    float cam_target_x, float cam_target_y, float cam_target_z,
    const std::string& config_json
) {
    // Call the version with empty intrinsics
    CameraIntrinsics empty_intrinsics;
    render_to_file(output_path, width, height,
                   cam_pos_x, cam_pos_y, cam_pos_z,
                   cam_target_x, cam_target_y, cam_target_z,
                   empty_intrinsics, config_json);
}

void GCodeViewerCore::render_to_file(
    const std::string& output_path,
    int width,
    int height,
    float cam_pos_x, float cam_pos_y, float cam_pos_z,
    float cam_target_x, float cam_target_y, float cam_target_z,
    const CameraIntrinsics& intrinsics,
    const std::string& config_json
) {
    if (!m_loaded) {
        throw std::runtime_error("No GCODE loaded. Call load() first.");
    }
    
    ensure_context();
    
    // Parse config if provided
    if (!config_json.empty()) {
        m_config = ConfigParser::parse_json(config_json);
    }
    
    // Apply configuration to viewer
    ConfigParser::apply_config(*m_viewer, m_config);
    
    // Set up camera
    m_camera.set_position(cam_pos_x, cam_pos_y, cam_pos_z);
    m_camera.set_target(cam_target_x, cam_target_y, cam_target_z);
    
    // Apply intrinsics if valid
    if (intrinsics.is_valid) {
        m_camera.set_intrinsics(intrinsics);
    } else {
        m_camera.set_use_intrinsics(false);
    }
    
    // Use config dimensions if not specified
    if (width <= 0) width = m_config.width;
    if (height <= 0) height = m_config.height;
    
    // Set up or resize framebuffer
    if (!m_framebuffer) {
        m_framebuffer = std::make_unique<FramebufferRenderer>();
    }
    
    if (m_current_width != width || m_current_height != height) {
        if (!m_framebuffer->init(width, height)) {
            throw std::runtime_error("Failed to create framebuffer");
        }
        m_current_width = width;
        m_current_height = height;
    }
    
    // Get matrices
    auto view_matrix = m_camera.get_view_matrix();
    auto projection_matrix = m_camera.get_projection_matrix(width, height);
    
    // Render
    bool success = m_framebuffer->render_to_file(
        output_path,
        m_config.background_color,
        [this, &view_matrix, &projection_matrix]() {
            // Render bed first (under the model)
            if (m_bed_renderer && m_bed_config.show_outline) {
                m_bed_renderer->render(view_matrix.data(), projection_matrix.data());
            }
            // Render the GCODE
            m_viewer->render(view_matrix, projection_matrix);
        }
    );
    
    if (!success) {
        throw std::runtime_error("Failed to render to file: " + output_path);
    }
}

} // namespace pygcode
