///
/// pygcode_viewer - GCode Parser
/// Simplified GCODE parser that creates libvgcode-compatible data
///

#ifndef PYGCODE_GCODE_PARSER_HPP
#define PYGCODE_GCODE_PARSER_HPP

#include <string>
#include <vector>
#include <array>
#include <map>
#include <cstdint>

#include "GCodeInputData.hpp"
#include "PathVertex.hpp"
#include "Types.hpp"

namespace pygcode {

/// Result of parsing a GCODE file
struct GCodeParseResult {
    libvgcode::GCodeInputData data;
    
    // Metadata
    float estimated_time = 0.0f;
    size_t layer_count = 0;
    std::array<float, 6> bounding_box = {0, 0, 0, 0, 0, 0}; // min_x, min_y, min_z, max_x, max_y, max_z
    
    // Statistics
    size_t move_count = 0;
    size_t extrusion_count = 0;
    size_t travel_count = 0;
};

/// Simplified GCODE parser
class GCodeParser {
public:
    GCodeParser();
    
    /// Parse a GCODE file and return the result
    GCodeParseResult parse(const std::string& filepath);
    
    /// Set tool colors for multi-extruder support
    void set_tool_colors(const std::vector<std::array<uint8_t, 3>>& colors);
    
private:
    struct ParserState {
        // Current position
        float x = 0.0f, y = 0.0f, z = 0.0f;
        float e = 0.0f;  // Extruder position (absolute)
        
        // Current settings
        float feedrate = 0.0f;
        float width = 0.4f;
        float height = 0.2f;
        float fan_speed = 0.0f;
        float temperature = 200.0f;
        
        // Current tool/extruder
        uint8_t extruder_id = 0;
        uint8_t color_id = 0;
        
        // Current role
        libvgcode::EGCodeExtrusionRole role = libvgcode::EGCodeExtrusionRole::None;
        
        // Current layer
        uint32_t layer_id = 0;
        
        // Gcode line number
        uint32_t gcode_id = 0;
        
        // Accumulated time
        float time = 0.0f;
        
        // Previous position for time calculation
        float prev_x = 0.0f, prev_y = 0.0f, prev_z = 0.0f;
        
        // Extrusion mode: true = relative (M83), false = absolute (M82)
        bool relative_e = false;
        
        // Position mode: true = relative (G91), false = absolute (G90)
        bool relative_xyz = false;
    };
    
    void parse_line(const std::string& line, ParserState& state, GCodeParseResult& result);
    void parse_gcode_command(const std::string& cmd, const std::map<char, float>& params, 
                             ParserState& state, GCodeParseResult& result);
    void parse_comment(const std::string& comment, ParserState& state);
    
    libvgcode::EGCodeExtrusionRole parse_role(const std::string& role_str);
    
    void add_vertex(const ParserState& state, libvgcode::EMoveType type, GCodeParseResult& result);
    void update_bounding_box(float x, float y, float z, GCodeParseResult& result);
    float calculate_move_time(const ParserState& state, float new_x, float new_y, float new_z);
    
    /// Linearize an arc (G2/G3) into line segments
    /// @param clockwise true for G2 (clockwise), false for G3 (counterclockwise)
    /// @param target_x, target_y Target position
    /// @param i_offset, j_offset Offset from current position to arc center
    /// @param total_e Total extrusion for the arc
    /// @param has_e Whether extrusion was specified
    void linearize_arc(bool clockwise, float target_x, float target_y, 
                       float i_offset, float j_offset, float total_e, bool has_e,
                       ParserState& state, GCodeParseResult& result);
    
    std::vector<std::array<uint8_t, 3>> m_tool_colors;
};

} // namespace pygcode

#endif // PYGCODE_GCODE_PARSER_HPP
