///
/// pygcode_viewer - GCode Parser Implementation
///

#include "gcode_parser.hpp"

#include <fstream>
#include <sstream>
#include <algorithm>
#include <cctype>
#include <cmath>
#include <limits>

namespace pygcode {

GCodeParser::GCodeParser() {
    // Default tool colors
    m_tool_colors = {
        {255, 128, 0},   // Orange
        {0, 255, 0},     // Green
        {0, 0, 255},     // Blue
        {255, 0, 255},   // Magenta
        {0, 255, 255},   // Cyan
    };
}

void GCodeParser::set_tool_colors(const std::vector<std::array<uint8_t, 3>>& colors) {
    m_tool_colors = colors;
}

GCodeParseResult GCodeParser::parse(const std::string& filepath) {
    GCodeParseResult result;
    ParserState state;
    
    // Initialize bounding box to invalid values
    result.bounding_box = {
        std::numeric_limits<float>::max(),
        std::numeric_limits<float>::max(),
        std::numeric_limits<float>::max(),
        std::numeric_limits<float>::lowest(),
        std::numeric_limits<float>::lowest(),
        std::numeric_limits<float>::lowest()
    };
    
    // Set default tool colors
    for (const auto& color : m_tool_colors) {
        result.data.tools_colors.push_back({color[0], color[1], color[2]});
    }
    
    // Open file
    std::ifstream file(filepath);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open GCODE file: " + filepath);
    }
    
    std::string line;
    while (std::getline(file, line)) {
        state.gcode_id++;
        parse_line(line, state, result);
    }
    
    // Finalize result
    result.layer_count = state.layer_id + 1;
    result.estimated_time = state.time;
    
    // Fix bounding box if no moves
    if (result.move_count == 0) {
        result.bounding_box = {0, 0, 0, 0, 0, 0};
    }
    
    return result;
}

void GCodeParser::parse_line(const std::string& line, ParserState& state, GCodeParseResult& result) {
    if (line.empty()) return;
    
    std::string trimmed = line;
    // Trim whitespace
    size_t start = trimmed.find_first_not_of(" \t\r\n");
    if (start == std::string::npos) return;
    trimmed = trimmed.substr(start);
    
    // Check for comment
    size_t comment_pos = trimmed.find(';');
    std::string command_part;
    std::string comment_part;
    
    if (comment_pos != std::string::npos) {
        command_part = trimmed.substr(0, comment_pos);
        comment_part = trimmed.substr(comment_pos + 1);
        parse_comment(comment_part, state);
    } else {
        command_part = trimmed;
    }
    
    // Skip if no command
    if (command_part.empty() || command_part[0] == ';') return;
    
    // Trim command part
    size_t end = command_part.find_last_not_of(" \t\r\n");
    if (end != std::string::npos) {
        command_part = command_part.substr(0, end + 1);
    }
    if (command_part.empty()) return;
    
    // Parse command and parameters
    std::istringstream iss(command_part);
    std::string cmd;
    iss >> cmd;
    
    // Convert command to uppercase
    std::transform(cmd.begin(), cmd.end(), cmd.begin(), ::toupper);
    
    // Parse parameters
    std::map<char, float> params;
    std::string param;
    while (iss >> param) {
        if (!param.empty() && param.size() > 1) {
            char key = std::toupper(param[0]);
            try {
                float value = std::stof(param.substr(1));
                params[key] = value;
            } catch (...) {
                // Ignore invalid parameters
            }
        }
    }
    
    parse_gcode_command(cmd, params, state, result);
}

void GCodeParser::parse_gcode_command(const std::string& cmd, const std::map<char, float>& params,
                                       ParserState& state, GCodeParseResult& result) {
    if (cmd == "G0" || cmd == "G1") {
        // Movement command
        float new_x, new_y, new_z;
        float e_value = 0.0f;
        bool has_e = params.count('E') > 0;
        
        if (state.relative_xyz) {
            new_x = state.x + (params.count('X') ? params.at('X') : 0.0f);
            new_y = state.y + (params.count('Y') ? params.at('Y') : 0.0f);
            new_z = state.z + (params.count('Z') ? params.at('Z') : 0.0f);
        } else {
            new_x = params.count('X') ? params.at('X') : state.x;
            new_y = params.count('Y') ? params.at('Y') : state.y;
            new_z = params.count('Z') ? params.at('Z') : state.z;
        }
        
        if (has_e) {
            e_value = params.at('E');
        }
        
        if (params.count('F')) {
            state.feedrate = params.at('F') / 60.0f; // Convert from mm/min to mm/s
        }
        
        // Calculate time
        state.time += calculate_move_time(state, new_x, new_y, new_z);
        
        // Determine move type based on extrusion mode
        libvgcode::EMoveType move_type = libvgcode::EMoveType::Noop;
        bool has_movement = (new_x != state.x || new_y != state.y || new_z != state.z);
        
        if (has_e) {
            float delta_e;
            if (state.relative_e) {
                // Relative mode: E value is the delta
                delta_e = e_value;
            } else {
                // Absolute mode: E value is absolute position
                delta_e = e_value - state.e;
            }
            
            if (delta_e < -0.001f) {
                // Retraction (with or without movement)
                move_type = libvgcode::EMoveType::Retract;
            } else if (delta_e > 0.001f && has_movement) {
                // Extrusion: positive E delta WITH movement
                move_type = libvgcode::EMoveType::Extrude;
                result.extrusion_count++;
            } else if (delta_e > 0.001f && !has_movement) {
                // Unretract: positive E delta WITHOUT movement (prime/deretract)
                move_type = libvgcode::EMoveType::Unretract;
            } else if (has_movement) {
                // No extrusion change but has movement = travel
                move_type = libvgcode::EMoveType::Travel;
                result.travel_count++;
            }
            
            // Update E position
            if (state.relative_e) {
                state.e += e_value;
            } else {
                state.e = e_value;
            }
        } else if (has_movement) {
            // No E parameter but has movement = travel
            move_type = libvgcode::EMoveType::Travel;
            result.travel_count++;
        }
        
        // Update position state
        state.prev_x = state.x;
        state.prev_y = state.y;
        state.prev_z = state.z;
        state.x = new_x;
        state.y = new_y;
        state.z = new_z;
        
        // Add vertex
        if (move_type != libvgcode::EMoveType::Noop) {
            add_vertex(state, move_type, result);
            result.move_count++;
        }
        
    } else if (cmd == "G2" || cmd == "G3") {
        // Arc movement: G2 = clockwise, G3 = counterclockwise
        bool clockwise = (cmd == "G2");
        
        // Get target position (defaults to current if not specified)
        float target_x, target_y;
        if (state.relative_xyz) {
            target_x = state.x + (params.count('X') ? params.at('X') : 0.0f);
            target_y = state.y + (params.count('Y') ? params.at('Y') : 0.0f);
        } else {
            target_x = params.count('X') ? params.at('X') : state.x;
            target_y = params.count('Y') ? params.at('Y') : state.y;
        }
        
        // Get arc center offset (I, J are always relative to current position)
        float i_offset = params.count('I') ? params.at('I') : 0.0f;
        float j_offset = params.count('J') ? params.at('J') : 0.0f;
        
        // Get extrusion value
        float e_value = 0.0f;
        bool has_e = params.count('E') > 0;
        if (has_e) {
            e_value = params.at('E');
        }
        
        // Update feedrate if specified
        if (params.count('F')) {
            state.feedrate = params.at('F') / 60.0f; // Convert from mm/min to mm/s
        }
        
        // Skip if no arc center offset specified (invalid arc)
        if (std::abs(i_offset) > 0.0001f || std::abs(j_offset) > 0.0001f) {
            // Linearize the arc into line segments
            linearize_arc(clockwise, target_x, target_y, i_offset, j_offset, e_value, has_e, state, result);
        }
        
    } else if (cmd == "G28") {
        // Home
        state.x = 0;
        state.y = 0;
        state.z = 0;
        
    } else if (cmd == "G90") {
        // Absolute positioning
        state.relative_xyz = false;
        
    } else if (cmd == "G91") {
        // Relative positioning
        state.relative_xyz = true;
        
    } else if (cmd == "G92") {
        // Set position
        if (params.count('X')) state.x = params.at('X');
        if (params.count('Y')) state.y = params.at('Y');
        if (params.count('Z')) state.z = params.at('Z');
        if (params.count('E')) state.e = params.at('E');
        
    } else if (cmd == "M82") {
        // Absolute extrusion mode
        state.relative_e = false;
        
    } else if (cmd == "M83") {
        // Relative extrusion mode
        state.relative_e = true;
        
    } else if (cmd == "M104" || cmd == "M109") {
        // Set extruder temperature
        if (params.count('S')) {
            state.temperature = params.at('S');
        }
        
    } else if (cmd == "M106") {
        // Set fan speed
        if (params.count('S')) {
            state.fan_speed = params.at('S') / 255.0f * 100.0f; // Convert to percentage
        }
        
    } else if (cmd == "M107") {
        // Fan off
        state.fan_speed = 0.0f;
        
    } else if (cmd[0] == 'T' && cmd.size() > 1) {
        // Tool change
        try {
            state.extruder_id = std::stoi(cmd.substr(1));
            
            // Add tool change marker
            add_vertex(state, libvgcode::EMoveType::ToolChange, result);
        } catch (...) {}
    }
}

void GCodeParser::parse_comment(const std::string& comment, ParserState& state) {
    std::string trimmed = comment;
    size_t start = trimmed.find_first_not_of(" \t");
    if (start != std::string::npos) {
        trimmed = trimmed.substr(start);
    }
    
    // PrusaSlicer TYPE comment
    if (trimmed.find("TYPE:") == 0) {
        std::string type_str = trimmed.substr(5);
        // Trim
        size_t end = type_str.find_first_of(" \t\r\n");
        if (end != std::string::npos) {
            type_str = type_str.substr(0, end);
        }
        state.role = parse_role(type_str);
        
    } else if (trimmed.find("LAYER_CHANGE") == 0 || trimmed.find("LAYER:") == 0) {
        state.layer_id++;
        
    } else if (trimmed.find("WIDTH:") == 0) {
        try {
            state.width = std::stof(trimmed.substr(6));
        } catch (...) {}
        
    } else if (trimmed.find("HEIGHT:") == 0) {
        try {
            state.height = std::stof(trimmed.substr(7));
        } catch (...) {}
        
    } else if (trimmed.find("COLOR_CHANGE") == 0) {
        state.color_id++;
        
    } else if (trimmed.find("WIPE_START") == 0) {
        // Wipe starts - handled in moves
        
    } else if (trimmed.find("WIPE_END") == 0) {
        // Wipe ends
    }
}

libvgcode::EGCodeExtrusionRole GCodeParser::parse_role(const std::string& role_str) {
    if (role_str == "Perimeter" || role_str == "perimeter")
        return libvgcode::EGCodeExtrusionRole::Perimeter;
    if (role_str == "External perimeter" || role_str == "external perimeter")
        return libvgcode::EGCodeExtrusionRole::ExternalPerimeter;
    if (role_str == "Overhang perimeter" || role_str == "overhang perimeter")
        return libvgcode::EGCodeExtrusionRole::OverhangPerimeter;
    if (role_str == "Internal infill" || role_str == "internal infill" || role_str == "Sparse infill")
        return libvgcode::EGCodeExtrusionRole::InternalInfill;
    if (role_str == "Solid infill" || role_str == "solid infill")
        return libvgcode::EGCodeExtrusionRole::SolidInfill;
    if (role_str == "Top solid infill" || role_str == "top solid infill")
        return libvgcode::EGCodeExtrusionRole::TopSolidInfill;
    if (role_str == "Ironing")
        return libvgcode::EGCodeExtrusionRole::Ironing;
    if (role_str == "Bridge infill" || role_str == "bridge infill")
        return libvgcode::EGCodeExtrusionRole::BridgeInfill;
    if (role_str == "Gap fill" || role_str == "gap fill")
        return libvgcode::EGCodeExtrusionRole::GapFill;
    if (role_str == "Skirt" || role_str == "skirt" || role_str == "Skirt/Brim")
        return libvgcode::EGCodeExtrusionRole::Skirt;
    if (role_str == "Support material" || role_str == "support material")
        return libvgcode::EGCodeExtrusionRole::SupportMaterial;
    if (role_str == "Support material interface" || role_str == "support material interface")
        return libvgcode::EGCodeExtrusionRole::SupportMaterialInterface;
    if (role_str == "Wipe tower" || role_str == "wipe tower")
        return libvgcode::EGCodeExtrusionRole::WipeTower;
    if (role_str == "Custom")
        return libvgcode::EGCodeExtrusionRole::Custom;
    
    return libvgcode::EGCodeExtrusionRole::None;
}

void GCodeParser::add_vertex(const ParserState& state, libvgcode::EMoveType type, GCodeParseResult& result) {
    libvgcode::PathVertex vertex;
    
    vertex.position = {state.x, state.y, state.z};
    vertex.height = state.height;
    vertex.width = state.width;
    vertex.feedrate = state.feedrate;
    vertex.actual_feedrate = state.feedrate;
    vertex.fan_speed = state.fan_speed;
    vertex.temperature = state.temperature;
    vertex.role = state.role;
    vertex.type = type;
    vertex.gcode_id = state.gcode_id;
    vertex.layer_id = state.layer_id;
    vertex.extruder_id = state.extruder_id;
    vertex.color_id = state.color_id;
    vertex.times = {state.time, state.time};
    
    // Calculate mm3_per_mm for extrusions
    if (type == libvgcode::EMoveType::Extrude) {
        vertex.mm3_per_mm = state.width * state.height;
    }
    
    result.data.vertices.push_back(vertex);
    
    // Update bounding box for extrusions and travels
    if (type == libvgcode::EMoveType::Extrude) {
        update_bounding_box(state.x, state.y, state.z, result);
    }
}

void GCodeParser::update_bounding_box(float x, float y, float z, GCodeParseResult& result) {
    result.bounding_box[0] = std::min(result.bounding_box[0], x);
    result.bounding_box[1] = std::min(result.bounding_box[1], y);
    result.bounding_box[2] = std::min(result.bounding_box[2], z);
    result.bounding_box[3] = std::max(result.bounding_box[3], x);
    result.bounding_box[4] = std::max(result.bounding_box[4], y);
    result.bounding_box[5] = std::max(result.bounding_box[5], z);
}

float GCodeParser::calculate_move_time(const ParserState& state, float new_x, float new_y, float new_z) {
    float dx = new_x - state.x;
    float dy = new_y - state.y;
    float dz = new_z - state.z;
    float distance = std::sqrt(dx*dx + dy*dy + dz*dz);
    
    if (state.feedrate > 0 && distance > 0) {
        return distance / state.feedrate;
    }
    return 0.0f;
}

void GCodeParser::linearize_arc(bool clockwise, float target_x, float target_y,
                                 float i_offset, float j_offset, float total_e, bool has_e,
                                 ParserState& state, GCodeParseResult& result) {
    // Arc center is at current position + offset
    float center_x = state.x + i_offset;
    float center_y = state.y + j_offset;
    
    // Calculate radii from center to start and end points
    float start_dx = state.x - center_x;
    float start_dy = state.y - center_y;
    float end_dx = target_x - center_x;
    float end_dy = target_y - center_y;
    
    float radius = std::sqrt(start_dx * start_dx + start_dy * start_dy);
    
    // Skip degenerate arcs
    if (radius < 0.001f) {
        return;
    }
    
    // Calculate start and end angles
    float start_angle = std::atan2(start_dy, start_dx);
    float end_angle = std::atan2(end_dy, end_dx);
    
    // Calculate angular distance
    float angular_distance;
    if (clockwise) {
        // G2: clockwise - angle decreases
        angular_distance = start_angle - end_angle;
        if (angular_distance <= 0.0f) {
            angular_distance += 2.0f * static_cast<float>(M_PI);
        }
    } else {
        // G3: counterclockwise - angle increases
        angular_distance = end_angle - start_angle;
        if (angular_distance <= 0.0f) {
            angular_distance += 2.0f * static_cast<float>(M_PI);
        }
    }
    
    // Calculate arc length
    float arc_length = radius * angular_distance;
    
    // Adaptive segment sizing: smaller segments for tighter curves
    // Segment length between 0.5mm and 2mm, based on radius
    float segment_length = std::max(0.5f, std::min(2.0f, radius * 0.1f));
    int num_segments = std::max(1, static_cast<int>(std::ceil(arc_length / segment_length)));
    
    // Limit segments to prevent excessive vertex generation
    num_segments = std::min(num_segments, 100);
    
    // Calculate delta values per segment
    float angle_step = angular_distance / static_cast<float>(num_segments);
    if (clockwise) {
        angle_step = -angle_step;
    }
    
    // Calculate extrusion per segment
    float delta_e_per_segment = 0.0f;
    float start_e = state.e;
    if (has_e) {
        float total_delta_e;
        if (state.relative_e) {
            total_delta_e = total_e;
        } else {
            total_delta_e = total_e - state.e;
        }
        delta_e_per_segment = total_delta_e / static_cast<float>(num_segments);
    }
    
    // Generate intermediate points along the arc
    float current_angle = start_angle;
    for (int i = 1; i <= num_segments; ++i) {
        current_angle += angle_step;
        
        // Calculate new position on arc
        float new_x, new_y;
        if (i == num_segments) {
            // Last segment - use exact target position to avoid rounding errors
            new_x = target_x;
            new_y = target_y;
        } else {
            new_x = center_x + radius * std::cos(current_angle);
            new_y = center_y + radius * std::sin(current_angle);
        }
        
        // Calculate time for this segment
        state.time += calculate_move_time(state, new_x, new_y, state.z);
        
        // Update position state
        state.prev_x = state.x;
        state.prev_y = state.y;
        state.prev_z = state.z;
        state.x = new_x;
        state.y = new_y;
        
        // Handle extrusion
        libvgcode::EMoveType move_type = libvgcode::EMoveType::Noop;
        if (has_e) {
            if (delta_e_per_segment > 0.001f) {
                // Extrusion move
                move_type = libvgcode::EMoveType::Extrude;
                result.extrusion_count++;
            } else if (delta_e_per_segment < -0.001f) {
                // Retraction (unlikely during arc but handle it)
                move_type = libvgcode::EMoveType::Retract;
            } else {
                // No extrusion change but has movement = travel
                move_type = libvgcode::EMoveType::Travel;
                result.travel_count++;
            }
            
            // Update E position
            if (state.relative_e) {
                state.e += delta_e_per_segment;
            } else {
                state.e = start_e + delta_e_per_segment * static_cast<float>(i);
            }
        } else {
            // No E parameter = travel
            move_type = libvgcode::EMoveType::Travel;
            result.travel_count++;
        }
        
        // Add vertex for this segment
        if (move_type != libvgcode::EMoveType::Noop) {
            add_vertex(state, move_type, result);
            result.move_count++;
        }
    }
}

} // namespace pygcode
